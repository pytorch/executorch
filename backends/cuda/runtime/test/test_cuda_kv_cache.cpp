/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/cuda/runtime/backend_options.h>
#include <executorch/backends/cuda/runtime/cuda_kv_cache.h>
#include <executorch/backends/cuda/runtime/cuda_kv_pool.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/cache/cell_cache.h>
#include <executorch/runtime/core/evalue.h>
#include <gtest/gtest.h>
#include <array>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace cu = ::executorch::backends::cuda;
namespace cache = ::executorch::extension::llm::cache;
namespace aoti = ::executorch::backends::aoti;
namespace slim = ::executorch::backends::aoti::slim;
namespace slimc10 = ::executorch::backends::aoti::slim::c10;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

namespace executorch::backends::cuda {

struct CudaKVPoolTestPeer {
  CudaKVPoolTestPeer()
      : previous_hook(CudaKVPool::private_fence_for_testing_),
        previous(active) {
    active = this;
    CudaKVPool::private_fence_for_testing_ = fence;
  }
  ~CudaKVPoolTestPeer() {
    CudaKVPool::private_fence_for_testing_ = previous_hook;
    active = previous;
  }
  int device_fences{0};
  bool fail_stream_fence{false};
  bool fail_device_fence{false};

 private:
  static cudaError_t fence(cudaStream_t stream, bool all_streams) {
    const auto result =
        all_streams ? cudaDeviceSynchronize() : cudaStreamSynchronize(stream);
    active->device_fences += all_streams;
    bool& fail =
        all_streams ? active->fail_device_fence : active->fail_stream_fence;
    if (result == cudaSuccess && fail) {
      fail = false;
      return cudaErrorUnknown;
    }
    return result;
  }
  CudaKVPool::FenceFunction previous_hook;
  CudaKVPoolTestPeer* previous;
  static thread_local CudaKVPoolTestPeer* active;
};

thread_local CudaKVPoolTestPeer* CudaKVPoolTestPeer::active = nullptr;

} // namespace executorch::backends::cuda

namespace {

struct Binding {
  void* data{nullptr};
  slimc10::ScalarType dtype{slimc10::ScalarType::Undefined};
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
};

struct FakeContainer {
  std::vector<std::string> internal_names;
  std::vector<std::string> fqns;
  std::unordered_map<std::string, Binding> bound;
  int updates{0};
  // Compiled metadata the fake reports for a constant index. Unset entries
  // report what the most recent make_cache() geometry and config declare.
  std::unordered_map<size_t, int32_t> dtype_override{};
  size_t last_update_pairs{0};
  // Compiled dtype and bytes by FQN, for constants make_cache() cannot
  // describe: storage built straight on a CudaKVPool, side buffers, and
  // programs compiled for another geometry than the cache's.
  std::unordered_map<std::string, std::pair<int32_t, size_t>> declared{};
  // Compiled shape by FQN, as the serialized FQN-weight metadata carries it.
  std::unordered_map<std::string, std::vector<int64_t>> shapes{};
};

// Declares `fqn` as compiled with `dtype` and contiguous `sizes`.
void declare(
    FakeContainer& container,
    const std::string& fqn,
    slimc10::ScalarType dtype,
    std::initializer_list<int64_t> sizes) {
  size_t bytes = slimc10::elementSize(dtype);
  for (const int64_t size : sizes) {
    bytes *= static_cast<size_t>(size);
  }
  container.declared[fqn] = {static_cast<int32_t>(dtype), bytes};
  container.shapes[fqn] = sizes;
}

// What the program under test was "compiled" with, for the fake's defaults.
struct Compiled {
  cache::CacheGeometry geometry;
  cache::CacheConfig config;
};
Compiled& compiled() {
  static Compiled value;
  return value;
}

std::shared_ptr<cache::Cache> make_cache(
    const cache::CacheGeometry& geometry,
    const cache::CacheConfig& cfg) {
  compiled() = Compiled{geometry, cfg};
  return cu::make_cuda_sequence_kv_cache(geometry, cfg);
}

// Rows the compiled program declares for one layer's K or V: the capacity,
// or window + max_write - 1 for a ring.
int64_t declared_rows(
    const cache::LayerGeometry& layer,
    const cache::CacheConfig& cfg) {
  return layer.policy.kind == cache::LayerPolicy::Kind::Ring
      ? layer.policy.window + cfg.max_write.value_or(1) - 1
      : cfg.capacity;
}

// Bytes the compiled program declares for one layer's K or V: BSHD at the
// declared rows.
size_t declared_bytes(
    const cache::LayerGeometry& layer,
    const cache::CacheConfig& cfg) {
  return static_cast<size_t>(declared_rows(layer, cfg)) * layer.n_kv_heads *
      layer.head_dim *
      slimc10::elementSize(static_cast<slimc10::ScalarType>(cfg.kv_dtype));
}

size_t layer_of(const std::string& fqn) {
  const std::string prefix = "__et_offgraph_kv_layer_";
  return std::stoul(fqn.substr(prefix.size()));
}

// The shapes the serialized metadata carries for the container's off-graph
// constants: what declare() recorded, else what the most recent make_cache()
// geometry and config declare for a layer.
std::unordered_map<std::string, std::vector<int64_t>> compiled_shapes(
    const FakeContainer& container) {
  std::unordered_map<std::string, std::vector<int64_t>> shapes;
  const auto& layers = compiled().geometry.layers;
  for (const std::string& fqn : container.fqns) {
    const auto declared = container.shapes.find(fqn);
    if (declared != container.shapes.end()) {
      shapes[fqn] = declared->second;
    } else if (
        fqn.rfind("__et_offgraph_kv_layer_", 0) == 0 &&
        layer_of(fqn) < layers.size()) {
      const auto& layer = layers[layer_of(fqn)];
      shapes[fqn] = {
          1,
          declared_rows(layer, compiled().config),
          layer.n_kv_heads,
          layer.head_dim};
    }
  }
  return shapes;
}

Error get_num_constants(
    aoti::AOTInductorModelContainerHandle container,
    size_t* count) {
  *count = reinterpret_cast<FakeContainer*>(container)->fqns.size();
  return Error::Ok;
}

Error get_constant_name(
    aoti::AOTInductorModelContainerHandle container,
    size_t index,
    const char** name) {
  *name = reinterpret_cast<FakeContainer*>(container)
              ->internal_names.at(index)
              .c_str();
  return Error::Ok;
}

Error get_constant_fqn(
    aoti::AOTInductorModelContainerHandle container,
    size_t index,
    const char** fqn) {
  *fqn = reinterpret_cast<FakeContainer*>(container)->fqns.at(index).c_str();
  return Error::Ok;
}

Error get_constant_dtype(
    aoti::AOTInductorModelContainerHandle container,
    size_t index,
    int32_t* dtype) {
  auto* fake = reinterpret_cast<FakeContainer*>(container);
  const auto it = fake->dtype_override.find(index);
  if (it != fake->dtype_override.end()) {
    *dtype = it->second;
    return Error::Ok;
  }
  const auto declared = fake->declared.find(fake->fqns.at(index));
  *dtype = declared != fake->declared.end() ? declared->second.first
                                            : compiled().config.kv_dtype;
  return Error::Ok;
}

Error get_constant_data_size(
    aoti::AOTInductorModelContainerHandle container,
    size_t index,
    size_t* data_size) {
  auto* fake = reinterpret_cast<FakeContainer*>(container);
  const std::string& fqn = fake->fqns.at(index);
  const auto declared = fake->declared.find(fqn);
  if (declared != fake->declared.end()) {
    *data_size = declared->second.second;
    return Error::Ok;
  }
  *data_size = declared_bytes(
      compiled().geometry.layers.at(layer_of(fqn)), compiled().config);
  return Error::Ok;
}

Error update_pairs(
    aoti::AOTInductorModelContainerHandle container,
    const aoti::AOTInductorConstantMapEntry* pairs,
    size_t count,
    bool,
    bool) {
  auto* fake = reinterpret_cast<FakeContainer*>(container);
  ++fake->updates;
  fake->last_update_pairs = count;
  for (size_t index = 0; index < count; ++index) {
    auto* tensor = reinterpret_cast<slim::SlimTensor*>(pairs[index].handle);
    fake->bound[pairs[index].name] = Binding{
        tensor->data_ptr(),
        tensor->dtype(),
        std::vector<int64_t>(tensor->sizes().begin(), tensor->sizes().end()),
        std::vector<int64_t>(
            tensor->strides().begin(), tensor->strides().end())};
  }
  return Error::Ok;
}

cu::CudaDelegateHandle make_handle(FakeContainer& container) {
  cu::CudaDelegateHandle handle;
  handle.container_handle =
      reinterpret_cast<aoti::AOTInductorModelContainerHandle>(&container);
  handle.get_num_constants = get_num_constants;
  handle.get_constant_name = get_constant_name;
  handle.get_constant_original_fqn = get_constant_fqn;
  handle.get_constant_dtype = get_constant_dtype;
  handle.get_constant_data_size = get_constant_data_size;
  handle.update_user_managed_constant_buffer_pairs = update_pairs;
  handle.offgraph_kv_sizes = compiled_shapes(container);
  return handle;
}

bool has_cuda_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

constexpr int kHeads = 2;
constexpr int kDim = 8;
constexpr int kRow = kHeads * kDim; // elements in one sequence row

cache::CacheGeometry flat_and_ring(int window) {
  cache::CacheGeometry geometry;
  geometry.layers = {
      {{cache::LayerPolicy::Kind::Flat, 0}, kHeads, kDim},
      {{cache::LayerPolicy::Kind::Ring, window}, kHeads, kDim},
  };
  return geometry;
}

FakeContainer flat_and_ring_container() {
  return FakeContainer{
      {"flat_k", "flat_v", "ring_k", "ring_v"},
      {"__et_offgraph_kv_layer_0_k",
       "__et_offgraph_kv_layer_0_v",
       "__et_offgraph_kv_layer_1_k",
       "__et_offgraph_kv_layer_1_v"},
      {},
      0};
}

std::vector<uint16_t> iota_rows(int rows) {
  std::vector<uint16_t> values(static_cast<size_t>(rows) * kRow);
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<uint16_t>(index + 1);
  }
  return values;
}

std::vector<uint16_t> read_rows(void* device, int rows) {
  std::vector<uint16_t> values(static_cast<size_t>(rows) * kRow);
  EXPECT_EQ(
      cudaMemcpy(
          values.data(),
          device,
          values.size() * sizeof(uint16_t),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  return values;
}

class CudaKVCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!has_cuda_device()) {
      GTEST_SKIP() << "CUDA device required";
    }
  }

  static cache::CacheConfig config(int capacity, int initial, int max_write) {
    cache::CacheConfig cfg;
    cfg.capacity = capacity;
    cfg.initial_capacity = initial;
    cfg.max_write = max_write;
    cfg.kv_dtype = static_cast<int>(slimc10::ScalarType::BFloat16);
    return cfg;
  }
};

} // namespace

TEST_F(CudaKVCacheTest, LayersGrowGeometricallyAndKeepHistory) {
  constexpr int kWindow = 4;
  constexpr int kMaxWrite = 9;
  auto cache_ptr = make_cache(flat_and_ring(kWindow), config(32, 4, kMaxWrite));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  auto container = flat_and_ring_container();
  auto handle = make_handle(container);
  ASSERT_TRUE(kv.note_handle(&handle).get());
  // Nothing is allocated until the first step asks for it.
  EXPECT_EQ(kv.metrics().allocated_bytes, 0);

  // A ring starts as small as a flat layer: it cannot wrap before it is full.
  ASSERT_EQ(kv.prepare_step(3, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  constexpr int kRingRows = kWindow + kMaxWrite - 1;
  constexpr int64_t kRowBytes = kRow * static_cast<int64_t>(sizeof(uint16_t));
  EXPECT_EQ(kv.metrics().flat_capacity, 4);
  EXPECT_EQ(kv.metrics().growth_count, 0);
  EXPECT_EQ(kv.metrics().allocated_bytes, 2 * (4 + 4) * kRowBytes);

  // Stand in for the kernels: the step writes rows [0, 3) of both layers.
  const auto history = iota_rows(3);
  for (const char* name : {"flat_k", "ring_k"}) {
    ASSERT_EQ(
        cudaMemcpy(
            container.bound[name].data,
            history.data(),
            history.size() * sizeof(uint16_t),
            cudaMemcpyHostToDevice),
        cudaSuccess);
  }
  void* first_flat = container.bound["flat_k"].data;
  void* first_ring = container.bound["ring_k"].data;
  ASSERT_EQ(kv.commit_step(3), Error::Ok);

  // 3 + 2 > 4: doubles to 8, moves both layers, keeps what was written.
  ASSERT_EQ(kv.prepare_step(2, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  EXPECT_EQ(kv.metrics().flat_capacity, 8);
  EXPECT_EQ(kv.metrics().growth_count, 1);
  EXPECT_NE(container.bound["flat_k"].data, first_flat);
  EXPECT_NE(container.bound["ring_k"].data, first_ring);
  EXPECT_EQ(read_rows(container.bound["flat_k"].data, 3), history);
  EXPECT_EQ(read_rows(container.bound["ring_k"].data, 3), history);
  ASSERT_EQ(kv.commit_step(2), Error::Ok);

  // A step wider than doubling asks for: grows straight to what it needs. The
  // ring stops at its slots and no longer moves.
  ASSERT_EQ(kv.prepare_step(9, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  EXPECT_EQ(kv.metrics().flat_capacity, 16);
  EXPECT_EQ(kv.metrics().allocated_bytes, 2 * (16 + kRingRows) * kRowBytes);
  void* full_ring = container.bound["ring_k"].data;
  ASSERT_EQ(kv.commit_step(9), Error::Ok);
  ASSERT_EQ(kv.prepare_step(9, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  EXPECT_EQ(kv.metrics().flat_capacity, 32);
  EXPECT_EQ(kv.metrics().growth_count, 3);
  EXPECT_EQ(kv.metrics().allocated_bytes, 2 * (32 + kRingRows) * kRowBytes);
  EXPECT_EQ(container.bound["ring_k"].data, full_ring);
  ASSERT_EQ(kv.commit_step(9), Error::Ok);

  // Filling to exactly the declared capacity needs no growth; one more token
  // past it is refused, not grown into.
  ASSERT_EQ(kv.prepare_step(9, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(kv.metrics().growth_count, 3);
  ASSERT_EQ(kv.commit_step(9), Error::Ok);
  EXPECT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::InvalidArgument);
  EXPECT_EQ(kv.metrics().logical_length, 32);

  // A reset keeps the grown storage, so it neither regrows nor moves.
  cache_ptr->as<cache::SequenceControl>()->clear();
  ASSERT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(kv.metrics().logical_length, 0);
  EXPECT_EQ(kv.metrics().flat_capacity, 32);
  EXPECT_EQ(kv.metrics().growth_count, 3);
  kv.forget_handle(&handle);
}

TEST_F(CudaKVCacheTest, MetricsClampFlatCapacityAtTheFlatLayers) {
  // The ring's ceiling (4 + 8 - 1 = 11) dominates the flat capacity, so
  // doubling from 6 rows lands the shared count at 11 while the flat layer
  // caps at 8. metrics must report what the flat layers hold.
  auto cache_ptr = make_cache(flat_and_ring(4), config(8, 3, 8));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  auto container = flat_and_ring_container();
  auto handle = make_handle(container);
  ASSERT_TRUE(kv.note_handle(&handle).get());
  for (int token = 0; token < 7; ++token) {
    ASSERT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::Ok);
    ASSERT_EQ(kv.commit_step(1), Error::Ok);
  }
  EXPECT_EQ(kv.metrics().logical_length, 7);
  EXPECT_EQ(kv.metrics().flat_capacity, 8);
  EXPECT_EQ(kv.metrics().growth_count, 2);
  kv.forget_handle(&handle);
}

TEST_F(CudaKVCacheTest, BindsTheDeclaredShapeOverTheCurrentAllocation) {
  constexpr int kWindow = 4;
  constexpr int kMaxWrite = 4;
  auto cache_ptr = make_cache(flat_and_ring(kWindow), config(64, 8, kMaxWrite));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  auto container = flat_and_ring_container();
  auto handle = make_handle(container);
  ASSERT_TRUE(kv.note_handle(&handle).get());

  ASSERT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);

  // The program's own declaration: BSHD at the maximum rows, although only
  // 8 flat rows are allocated. The ring is fixed at window + max_write - 1.
  const std::vector<int64_t> flat_sizes{1, 64, kHeads, kDim};
  const std::vector<int64_t> flat_strides{64 * kRow, kRow, kDim, 1};
  const std::vector<int64_t> ring_sizes{
      1, kWindow + kMaxWrite - 1, kHeads, kDim};
  EXPECT_EQ(container.bound["flat_k"].sizes, flat_sizes);
  EXPECT_EQ(container.bound["flat_k"].strides, flat_strides);
  EXPECT_EQ(container.bound["flat_v"].sizes, flat_sizes);
  EXPECT_EQ(container.bound["ring_k"].sizes, ring_sizes);
  EXPECT_EQ(kv.metrics().flat_capacity, 8);

  // Rebinding is cached until storage moves.
  const int updates = container.updates;
  ASSERT_EQ(kv.commit_step(1), Error::Ok);
  ASSERT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  EXPECT_EQ(container.updates, updates);
  kv.forget_handle(&handle);
}

TEST_F(CudaKVCacheTest, GrowthIsOrderedOnTheStepStream) {
  cudaStream_t stream = nullptr;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  cache::CacheGeometry geometry;
  geometry.layers = {{{cache::LayerPolicy::Kind::Flat, 0}, kHeads, kDim}};
  auto cache_ptr = make_cache(geometry, config(1 << 16, 4, 4096));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  FakeContainer container{
      {"flat_k", "flat_v"},
      {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"},
      {},
      0};
  auto handle = make_handle(container);
  ASSERT_TRUE(kv.note_handle(&handle).get());

  // Enough rows that an unordered copy would plausibly overtake the write.
  constexpr int kRows = 4096;
  ASSERT_EQ(kv.prepare_step(kRows, stream), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  const auto history = iota_rows(kRows);
  // The "kernel" writes asynchronously on the step stream, exactly as the
  // compiled program would; nothing waits for it before the next step grows.
  ASSERT_EQ(
      cudaMemcpyAsync(
          container.bound["flat_k"].data,
          history.data(),
          history.size() * sizeof(uint16_t),
          cudaMemcpyHostToDevice,
          stream),
      cudaSuccess);
  ASSERT_EQ(kv.commit_step(kRows), Error::Ok);

  ASSERT_EQ(kv.prepare_step(1, stream), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  ASSERT_EQ(kv.metrics().growth_count, 1);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(read_rows(container.bound["flat_k"].data, kRows), history);
  kv.forget_handle(&handle);
  cache_ptr.reset();
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(CudaKVCacheTest, FailedGrowthLeavesTheCacheAsItWas) {
  // Layer 0 grows first and fits; layer 1 is sized so its growth cannot be
  // allocated on any device (tens of GB per row times tens of thousands of
  // rows), so the failure lands after one layer has already grown.
  constexpr int kHugeDim = 1 << 22;
  cache::CacheGeometry geometry;
  geometry.layers = {
      {{cache::LayerPolicy::Kind::Flat, 0}, kHeads, kDim},
      {{cache::LayerPolicy::Kind::Flat, 0}, 1, kHugeDim},
  };
  auto cache_ptr = make_cache(geometry, config(1 << 16, 4, 1 << 16));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  FakeContainer container{
      {"small_k", "small_v", "huge_k", "huge_v"},
      {"__et_offgraph_kv_layer_0_k",
       "__et_offgraph_kv_layer_0_v",
       "__et_offgraph_kv_layer_1_k",
       "__et_offgraph_kv_layer_1_v"},
      {},
      0};
  auto handle = make_handle(container);
  ASSERT_TRUE(kv.note_handle(&handle).get());

  ASSERT_EQ(kv.prepare_step(3, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  const auto history = iota_rows(3);
  ASSERT_EQ(
      cudaMemcpy(
          container.bound["small_k"].data,
          history.data(),
          history.size() * sizeof(uint16_t),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(kv.commit_step(3), Error::Ok);
  const auto before = kv.metrics();
  void* small_k = container.bound["small_k"].data;
  void* huge_k = container.bound["huge_k"].data;
  const int updates = container.updates;

  EXPECT_EQ(
      kv.prepare_step(60000, cudaStreamPerThread),
      Error::MemoryAllocationFailed);

  // Nothing moved: the bindings still point at live storage, which still holds
  // the history, and capacity and accounting are unchanged.
  const auto after = kv.metrics();
  EXPECT_EQ(after.flat_capacity, before.flat_capacity);
  EXPECT_EQ(after.growth_count, 0);
  EXPECT_EQ(after.allocated_bytes, before.allocated_bytes);
  EXPECT_EQ(after.logical_length, 3);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  EXPECT_EQ(container.updates, updates);
  EXPECT_EQ(container.bound["small_k"].data, small_k);
  EXPECT_EQ(container.bound["huge_k"].data, huge_k);
  EXPECT_EQ(read_rows(small_k, 3), history);

  // And the cache is still usable: a step that fits runs, and a modest growth
  // succeeds.
  ASSERT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.commit_step(1), Error::Ok);
  ASSERT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(kv.metrics().flat_capacity, 8);
  EXPECT_EQ(kv.metrics().growth_count, 1);
  kv.forget_handle(&handle);
}

TEST_F(CudaKVCacheTest, FailedFirstAllocationCanBeRetried) {
  cache::CacheGeometry geometry;
  geometry.layers = {
      {{cache::LayerPolicy::Kind::Flat, 0}, kHeads, kDim},
      {{cache::LayerPolicy::Kind::Flat, 0}, 1, 1 << 22},
  };
  auto cache_ptr = make_cache(geometry, config(1 << 16, 4, 1 << 16));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  FakeContainer container{
      {"small_k", "small_v", "huge_k", "huge_v"},
      {"__et_offgraph_kv_layer_0_k",
       "__et_offgraph_kv_layer_0_v",
       "__et_offgraph_kv_layer_1_k",
       "__et_offgraph_kv_layer_1_v"},
      {},
      0};
  auto handle = make_handle(container);
  ASSERT_TRUE(kv.note_handle(&handle).get());

  // A first step too wide to allocate frees what it did allocate...
  EXPECT_EQ(
      kv.prepare_step(60000, cudaStreamPerThread),
      Error::MemoryAllocationFailed);
  EXPECT_EQ(kv.metrics().allocated_bytes, 0);
  // ...and a narrower one then allocates normally.
  ASSERT_EQ(kv.prepare_step(3, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  EXPECT_EQ(kv.metrics().flat_capacity, 4);
  kv.forget_handle(&handle);
}

TEST_F(CudaKVCacheTest, StepOnAnotherStreamWaitsForThePreviousOne) {
  cudaStream_t first = nullptr;
  cudaStream_t second = nullptr;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking), cudaSuccess);
  cache::CacheGeometry geometry;
  geometry.layers = {{{cache::LayerPolicy::Kind::Flat, 0}, kHeads, kDim}};
  auto cache_ptr = make_cache(geometry, config(1 << 16, 4, 4096));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  FakeContainer container{
      {"flat_k", "flat_v"},
      {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"},
      {},
      0};
  auto handle = make_handle(container);
  ASSERT_TRUE(kv.note_handle(&handle).get());

  constexpr int kRows = 1024;
  ASSERT_EQ(kv.prepare_step(kRows, first), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  // The previous step's "kernel" is held back on its stream, then writes.
  const size_t bytes = static_cast<size_t>(kRows) * kRow * sizeof(uint16_t);
  uint16_t* pinned = nullptr;
  ASSERT_EQ(cudaMallocHost(&pinned, bytes), cudaSuccess);
  const auto history = iota_rows(kRows);
  std::copy(history.begin(), history.end(), pinned);
  ASSERT_EQ(
      cudaLaunchHostFunc(
          first,
          [](void*) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
          },
          nullptr),
      cudaSuccess);
  ASSERT_EQ(
      cudaMemcpyAsync(
          container.bound["flat_k"].data,
          pinned,
          bytes,
          cudaMemcpyHostToDevice,
          first),
      cudaSuccess);
  ASSERT_EQ(kv.commit_step(kRows), Error::Ok);

  // The next step runs on another stream and grows. Its copy must wait for the
  // held-back write rather than copy what was there before it.
  ASSERT_EQ(kv.prepare_step(1, second), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  ASSERT_EQ(kv.metrics().growth_count, 1);
  ASSERT_EQ(cudaStreamSynchronize(second), cudaSuccess);
  EXPECT_EQ(read_rows(container.bound["flat_k"].data, kRows), history);

  ASSERT_EQ(cudaStreamSynchronize(first), cudaSuccess);
  ASSERT_EQ(cudaFreeHost(pinned), cudaSuccess);
  kv.forget_handle(&handle);
  cache_ptr.reset();
  ASSERT_EQ(cudaStreamDestroy(first), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(second), cudaSuccess);
}

TEST_F(CudaKVCacheTest, CacheNotMatchingTheCompiledRingIsRejected) {
  // Compiled with max_write 8: the ring constants span window + 8 - 1 rows.
  // A cache installed with max_write 1 would allocate window rows and let a
  // one-token step at position `window` write past them.
  constexpr int kWindow = 16;
  auto cache_ptr = make_cache(flat_and_ring(kWindow), config(64, 4, 1));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  auto container = flat_and_ring_container();
  for (const char* fqn :
       {"__et_offgraph_kv_layer_1_k", "__et_offgraph_kv_layer_1_v"}) {
    declare(
        container,
        fqn,
        slimc10::ScalarType::BFloat16,
        {1, kWindow + 8 - 1, kHeads, kDim});
  }
  auto handle = make_handle(container);

  EXPECT_EQ(kv.note_handle(&handle).error(), Error::InvalidProgram);
  EXPECT_EQ(container.updates, 0);
}

TEST_F(CudaKVCacheTest, RingWithinTheSamePaddedBytesIsRejected) {
  // One bf16 head of 16: a ring compiled at window 2 spans 64 bytes, and one
  // installed at window 1 allocates 32, which AOTI's 64-byte rounding also
  // reports as 64. A step at position 1 would index the second row, past the
  // allocation, so only the shape may decide.
  cache::CacheGeometry geometry;
  geometry.layers = {{{cache::LayerPolicy::Kind::Ring, 1}, 1, 16}};
  auto cache_ptr = make_cache(geometry, config(64, 4, 1));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  FakeContainer container{
      {"ring_k", "ring_v"},
      {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"},
      {},
      0};
  for (const char* fqn :
       {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"}) {
    declare(container, fqn, slimc10::ScalarType::BFloat16, {1, 2, 1, 16});
  }
  ASSERT_EQ(container.declared["__et_offgraph_kv_layer_0_k"].second, 64u);
  auto handle = make_handle(container);

  EXPECT_EQ(kv.note_handle(&handle).error(), Error::InvalidProgram);
  EXPECT_EQ(container.updates, 0);
}

TEST_F(CudaKVCacheTest, CacheNotMatchingTheCompiledCapacityIsRejected) {
  auto cache_ptr = make_cache(flat_and_ring(4), config(64, 4, 4));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  auto container = flat_and_ring_container();
  for (const char* fqn :
       {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"}) {
    declare(
        container, fqn, slimc10::ScalarType::BFloat16, {1, 128, kHeads, kDim});
  }
  auto handle = make_handle(container);

  EXPECT_EQ(kv.note_handle(&handle).error(), Error::InvalidProgram);
}

TEST_F(CudaKVCacheTest, CacheNotMatchingTheCompiledDtypeIsRejected) {
  // One-byte storage for a program whose kernels read bf16 would under-
  // allocate by half; the dtype check refuses it before any binding.
  cache::CacheGeometry geometry;
  geometry.layers = {{{cache::LayerPolicy::Kind::Flat, 0}, kHeads, kDim}};
  auto cfg = config(32, 4, 4);
  cfg.kv_dtype = static_cast<int>(slimc10::ScalarType::Byte);
  auto cache_ptr = make_cache(geometry, cfg);
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  FakeContainer container{
      {"flat_k", "flat_v"},
      {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"},
      {},
      0};
  const auto bf16 = static_cast<int32_t>(slimc10::ScalarType::BFloat16);
  container.dtype_override = {{0, bf16}, {1, bf16}};
  auto handle = make_handle(container);

  EXPECT_EQ(kv.note_handle(&handle).error(), Error::InvalidProgram);
}

TEST_F(CudaKVCacheTest, ReloadingAHandleBindsEachConstantOnce) {
  auto cache_ptr = make_cache(flat_and_ring(4), config(64, 4, 4));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  auto container = flat_and_ring_container();
  auto handle = make_handle(container);
  ASSERT_TRUE(kv.note_handle(&handle).get());
  ASSERT_TRUE(kv.note_handle(&handle).get());

  ASSERT_EQ(kv.prepare_step(2, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);

  EXPECT_EQ(container.updates, 1);
  EXPECT_EQ(container.last_update_pairs, 4);
  kv.forget_handle(&handle);
}

TEST_F(CudaKVCacheTest, CommitAdmitsTheStepOnEveryLayer) {
  // The flat layer alone would accept an 8-token step; the ring, sized for
  // max_write 4, cannot, and commit must refuse it as prepare would.
  auto cache_ptr = make_cache(flat_and_ring(4), config(64, 4, 4));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  auto container = flat_and_ring_container();
  auto handle = make_handle(container);
  ASSERT_TRUE(kv.note_handle(&handle).get());
  ASSERT_EQ(kv.prepare_step(2, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.commit_step(2), Error::Ok);

  EXPECT_EQ(kv.commit_step(8), Error::InvalidArgument);
  EXPECT_EQ(kv.metrics().logical_length, 2);
  kv.forget_handle(&handle);
}

TEST_F(CudaKVCacheTest, CacheOutlivesTheStreamItLastSteppedOn) {
  cudaStream_t stream = nullptr;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  auto cache_ptr = make_cache(flat_and_ring(4), config(64, 4, 4));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  auto container = flat_and_ring_container();
  auto handle = make_handle(container);
  ASSERT_TRUE(kv.note_handle(&handle).get());
  ASSERT_EQ(kv.prepare_step(2, stream), Error::Ok);
  ASSERT_EQ(kv.commit_step(2), Error::Ok);
  kv.forget_handle(&handle);

  // The caller's stream goes first; tearing the cache down must not use it.
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  cache_ptr.reset();
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(OffGraphKVRequirementTest, ReadFromTheProgramNotTheCaller) {
  // A program lowered off-graph must not run without a cache even when the
  // caller passed no key; one that is not must not be asked for one.
  auto offgraph = flat_and_ring_container();
  auto offgraph_handle = make_handle(offgraph);
  const auto needs = cu::requires_offgraph_kv_storage(offgraph_handle);
  ASSERT_TRUE(needs.ok());
  EXPECT_TRUE(needs.get());

  FakeContainer in_graph{{"weight"}, {"tok_embeddings.weight"}, {}, 0};
  auto in_graph_handle = make_handle(in_graph);
  const auto plain = cu::requires_offgraph_kv_storage(in_graph_handle);
  ASSERT_TRUE(plain.ok());
  EXPECT_FALSE(plain.get());
}

TEST_F(CudaKVCacheTest, RingLayersRequireMaxWrite) {
  auto cfg = config(32, 4, 4);
  cfg.max_write.reset();
  EXPECT_EQ(make_cache(flat_and_ring(4), cfg), nullptr);

  cache::CacheGeometry flat_only;
  flat_only.layers = {{{cache::LayerPolicy::Kind::Flat, 0}, kHeads, kDim}};
  EXPECT_NE(make_cache(flat_only, cfg), nullptr);
}

TEST_F(CudaKVCacheTest, ProgramWithoutStorageIsNotServed) {
  cache::CacheGeometry geometry;
  geometry.layers = {{{cache::LayerPolicy::Kind::Flat, 0}, kHeads, kDim}};
  auto cache_ptr = make_cache(geometry, config(32, 4, 4));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  FakeContainer embedding{{"weight"}, {"tok_embeddings.weight"}, {}, 0};
  auto handle = make_handle(embedding);
  const auto serves = kv.note_handle(&handle);
  ASSERT_EQ(serves.error(), Error::Ok);
  EXPECT_FALSE(serves.get());
  EXPECT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  EXPECT_EQ(embedding.updates, 0);
}

TEST_F(CudaKVCacheTest, SupportedDenseDtypesControlStorageAndBindings) {
  for (const auto dtype : {
           slimc10::ScalarType::Byte,
           slimc10::ScalarType::Char,
           slimc10::ScalarType::Short,
           slimc10::ScalarType::Int,
           slimc10::ScalarType::Long,
           slimc10::ScalarType::Half,
           slimc10::ScalarType::Float,
           slimc10::ScalarType::Bool,
           slimc10::ScalarType::BFloat16,
       }) {
    SCOPED_TRACE(slimc10::toString(dtype));
    auto cfg = config(8, 4, 4);
    cfg.kv_dtype = static_cast<int>(dtype);
    cache::CacheGeometry geometry;
    geometry.layers = {{{cache::LayerPolicy::Kind::Flat, 0}, kHeads, kDim}};
    auto cache_ptr = make_cache(geometry, cfg);
    ASSERT_NE(cache_ptr, nullptr);
    auto& kv = *cache_ptr->as<cu::CudaKVCache>();
    FakeContainer container{
        {"flat_k", "flat_v"},
        {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"},
        {},
        0};
    auto handle = make_handle(container);
    ASSERT_TRUE(kv.note_handle(&handle).get());

    ASSERT_EQ(kv.prepare_step(4, cudaStreamPerThread), Error::Ok);
    ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
    EXPECT_EQ(container.bound["flat_k"].dtype, dtype);
    EXPECT_EQ(container.bound["flat_v"].dtype, dtype);
    const int64_t element_size =
        static_cast<int64_t>(slimc10::elementSize(dtype));
    EXPECT_EQ(kv.metrics().allocated_bytes, 2 * 4 * kRow * element_size);

    // Growth carries the rows across byte-for-byte whatever the dtype.
    std::vector<uint8_t> values(4 * kRow * element_size);
    for (size_t index = 0; index < values.size(); ++index) {
      values[index] = static_cast<uint8_t>(index % 251 + 1);
    }
    ASSERT_EQ(
        cudaMemcpy(
            container.bound["flat_k"].data,
            values.data(),
            values.size(),
            cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(kv.commit_step(4), Error::Ok);
    ASSERT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::Ok);
    ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
    EXPECT_EQ(kv.metrics().flat_capacity, 8);
    std::vector<uint8_t> copied(values.size());
    ASSERT_EQ(
        cudaMemcpy(
            copied.data(),
            container.bound["flat_k"].data,
            copied.size(),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(copied, values);
    kv.forget_handle(&handle);
  }
}

TEST(OffGraphKVStepWidthTest, ParsesInputIndexAndDim) {
  const auto parsed = cu::parse_offgraph_kv_step_width("12:3");
  ASSERT_EQ(parsed.error(), Error::Ok);
  EXPECT_EQ(parsed.get().input, 12);
  EXPECT_EQ(parsed.get().dim, 3);

  for (const char* bad : {"", "1", ":0", "1:", "a:0", "1:-1", "-1:0", "1:0x"}) {
    SCOPED_TRACE(bad);
    EXPECT_EQ(
        cu::parse_offgraph_kv_step_width(bad).error(), Error::InvalidArgument);
  }
}

TEST(OffGraphKVStepWidthTest, ReadsTheExtentOfTheNamedInput) {
  namespace etensor = ::executorch::runtime::etensor;
  etensor::TensorImpl::SizesType sizes[] = {1, 7};
  etensor::TensorImpl::DimOrderType dim_order[] = {0, 1};
  int64_t data[7] = {};
  etensor::TensorImpl impl(
      etensor::ScalarType::Long, 2, sizes, data, dim_order);
  EValue scalar(static_cast<int64_t>(3));
  EValue tensor{etensor::Tensor(&impl)};
  EValue* inputs[] = {&scalar, &tensor};
  const ::executorch::runtime::Span<EValue*> span(inputs, 2);

  const auto width = cu::read_offgraph_kv_step_width({1, 1}, span);
  ASSERT_EQ(width.error(), Error::Ok);
  EXPECT_EQ(width.get(), 7);

  // Wrong input, a non-tensor, a dim past the rank, and an empty step.
  EXPECT_EQ(
      cu::read_offgraph_kv_step_width({2, 0}, span).error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      cu::read_offgraph_kv_step_width({0, 0}, span).error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      cu::read_offgraph_kv_step_width({1, 2}, span).error(),
      Error::InvalidArgument);
  sizes[1] = 0;
  etensor::TensorImpl empty(
      etensor::ScalarType::Long, 2, sizes, data, dim_order);
  EValue empty_tensor{etensor::Tensor(&empty)};
  EValue* empty_inputs[] = {&scalar, &empty_tensor};
  EXPECT_EQ(
      cu::read_offgraph_kv_step_width(
          {1, 1}, ::executorch::runtime::Span<EValue*>(empty_inputs, 2))
          .error(),
      Error::InvalidArgument);
}

TEST_F(CudaKVCacheTest, GrowthDropsACapturedGraphOnlyWhenStorageMoves) {
  cache::CacheGeometry geometry;
  geometry.layers = {{{cache::LayerPolicy::Kind::Flat, 0}, kHeads, kDim}};
  auto cache_ptr = make_cache(geometry, config(32, 4, 4));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  FakeContainer container{
      {"flat_k", "flat_v"},
      {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"},
      {},
      0};
  auto handle = make_handle(container);
  ASSERT_TRUE(kv.note_handle(&handle).get());

  // A decode graph captured against the initial storage.
  ASSERT_EQ(kv.prepare_step(3, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  auto& graph = handle.cuda_graph_state;
  graph.phase = cu::CudaGraphPhase::Replay;
  void* static_input = nullptr;
  ASSERT_EQ(cudaMalloc(&static_input, 16), cudaSuccess);
  graph.static_input_ptrs = {static_input};
  graph.static_input_nbytes = {16};
  graph.static_input_sizes = {{4}};
  ASSERT_EQ(kv.commit_step(3), Error::Ok);

  // Fits the current storage: the graph keeps replaying.
  ASSERT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(kv.metrics().growth_count, 0);
  EXPECT_EQ(graph.phase, cu::CudaGraphPhase::Replay);
  EXPECT_EQ(graph.static_input_ptrs.size(), 1);
  ASSERT_EQ(kv.commit_step(1), Error::Ok);

  // Grows: the graph points at freed storage, so the growth itself drops it
  // -- freeing what the capture pinned -- and this very call captures again,
  // after one eager step that absorbs AOTI's constant fold.
  ASSERT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.metrics().growth_count, 1);
  EXPECT_EQ(graph.phase, cu::CudaGraphPhase::Warmup);
  EXPECT_EQ(graph.warmup_remaining, 1);
  EXPECT_TRUE(graph.static_input_ptrs.empty());
  EXPECT_TRUE(graph.static_input_nbytes.empty());
  EXPECT_TRUE(graph.static_input_sizes.empty());
  EXPECT_EQ(graph.graph_exec, nullptr);

  // Once recaptured, later steps that fit replay it untouched.
  graph.phase = cu::CudaGraphPhase::Replay;
  ASSERT_EQ(kv.commit_step(1), Error::Ok);
  ASSERT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(graph.phase, cu::CudaGraphPhase::Replay);
  kv.forget_handle(&handle);
}

TEST_F(CudaKVCacheTest, GrowthJustBeforeFirstCaptureKeepsAnEagerStep) {
  // A handle whose warmup has run out captures on its next call. A growth
  // now rebinds its constants, which must first be folded eagerly, so it gets
  // one eager step back -- while a longer outstanding warmup is left alone.
  cache::CacheGeometry geometry;
  geometry.layers = {{{cache::LayerPolicy::Kind::Flat, 0}, kHeads, kDim}};
  auto cache_ptr = make_cache(geometry, config(64, 4, 16));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  auto names = [] {
    return FakeContainer{
        {"flat_k", "flat_v"},
        {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"},
        {},
        0};
  };
  auto about_container = names();
  auto warming_container = names();
  auto about_to_capture = make_handle(about_container);
  auto still_warming = make_handle(warming_container);
  ASSERT_TRUE(kv.note_handle(&about_to_capture).get());
  ASSERT_TRUE(kv.note_handle(&still_warming).get());
  about_to_capture.cuda_graph_state.phase = cu::CudaGraphPhase::Warmup;
  about_to_capture.cuda_graph_state.warmup_remaining = 0;
  still_warming.cuda_graph_state.phase = cu::CudaGraphPhase::Warmup;
  still_warming.cuda_graph_state.warmup_remaining = 2;

  ASSERT_EQ(kv.prepare_step(4, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.commit_step(4), Error::Ok);
  ASSERT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.metrics().growth_count, 1);

  EXPECT_EQ(
      about_to_capture.cuda_graph_state.phase, cu::CudaGraphPhase::Warmup);
  EXPECT_EQ(about_to_capture.cuda_graph_state.warmup_remaining, 1);
  EXPECT_EQ(still_warming.cuda_graph_state.warmup_remaining, 2);
  kv.forget_handle(&about_to_capture);
  kv.forget_handle(&still_warming);
}

TEST_F(CudaKVCacheTest, RetiringAGraphFreesWhatItsLastLaunchAllocated) {
  // A captured program allocates its outputs inside the graph. Dropping the
  // graph to recapture must free them; AutoFreeOnLaunch only would on a next
  // launch of the same graph, which never comes.
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  auto graph_mem_in_use = [device] {
    size_t used = 0;
    EXPECT_EQ(
        cudaDeviceGetGraphMemAttribute(
            device, cudaGraphMemAttrUsedMemCurrent, &used),
        cudaSuccess);
    return used;
  };
  cudaStream_t stream = nullptr;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  // Only memory a graph still holds survives a trim, so measure after one.
  ASSERT_EQ(cudaDeviceGraphMemTrim(device), cudaSuccess);
  const size_t before = graph_mem_in_use();

  constexpr size_t kBytes = 4 << 20;
  cu::CudaGraphState graph;
  void* output = nullptr;
  ASSERT_EQ(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed),
      cudaSuccess);
  ASSERT_EQ(cudaMallocAsync(&output, kBytes, stream), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(output, 0, kBytes, stream), cudaSuccess);
  void* scratch = nullptr;
  ASSERT_EQ(cudaMallocAsync(&scratch, kBytes, stream), cudaSuccess);
  ASSERT_EQ(cudaFreeAsync(scratch, stream), cudaSuccess);
  ASSERT_EQ(cudaStreamEndCapture(stream, &graph.graph), cudaSuccess);
  ASSERT_EQ(
      cudaGraphInstantiate(
          &graph.graph_exec,
          graph.graph,
          cudaGraphInstantiateFlagAutoFreeOnLaunch),
      cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(graph.graph_exec, stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_GE(graph_mem_in_use(), before + kBytes);
  graph.note_graph_allocations();
  // Only the allocation the graph leaves outstanding is its to free.
  ASSERT_EQ(graph.graph_allocations.size(), 1);
  EXPECT_EQ(graph.graph_allocations[0], output);

  graph.release();
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaDeviceGraphMemTrim(device), cudaSuccess);
  EXPECT_EQ(graph_mem_in_use(), before);
  EXPECT_TRUE(graph.graph_allocations.empty());
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(CudaKVCacheTest, GrowthDuringAnotherMethodDropsItsGraph) {
  // Prefill runs eagerly and is where most growth happens, while decode's
  // graph sits idle. The growth must reach decode too.
  cache::CacheGeometry geometry;
  geometry.layers = {{{cache::LayerPolicy::Kind::Flat, 0}, kHeads, kDim}};
  auto cache_ptr = make_cache(geometry, config(64, 4, 16));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  auto names = [] {
    return FakeContainer{
        {"flat_k", "flat_v"},
        {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"},
        {},
        0};
  };
  auto prefill_container = names();
  auto decode_container = names();
  auto prefill = make_handle(prefill_container);
  auto decode = make_handle(decode_container);
  ASSERT_TRUE(kv.note_handle(&prefill).get());
  ASSERT_TRUE(kv.note_handle(&decode).get());

  ASSERT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&decode), Error::Ok);
  decode.cuda_graph_state.phase = cu::CudaGraphPhase::Replay;
  void* captured_storage = decode_container.bound["flat_k"].data;
  ASSERT_EQ(kv.commit_step(1), Error::Ok);

  // Prefill grows the cache; decode is not running.
  ASSERT_EQ(kv.prepare_step(16, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.metrics().growth_count, 1);
  EXPECT_EQ(decode.cuda_graph_state.phase, cu::CudaGraphPhase::Warmup);
  EXPECT_EQ(decode.cuda_graph_state.warmup_remaining, 1);
  // A handle that never captured is left as it was.
  EXPECT_EQ(prefill.cuda_graph_state.phase, cu::CudaGraphPhase::Disabled);
  ASSERT_EQ(kv.rebind_for_execute(&prefill), Error::Ok);
  ASSERT_EQ(kv.commit_step(16), Error::Ok);

  // decode's next run rebinds away from the storage it captured.
  ASSERT_EQ(kv.prepare_step(1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&decode), Error::Ok);
  EXPECT_NE(decode_container.bound["flat_k"].data, captured_storage);
  kv.forget_handle(&prefill);
  kv.forget_handle(&decode);
}

namespace {

// One growable layer, one fixed layer and two side buffers: everything the
// pool distinguishes, and nothing about what the rows mean.
class CudaKVPoolTest : public CudaKVCacheTest {
 protected:
  static constexpr int kDeclaredRows = 64;
  static constexpr int kFixedRows = 6;

  std::unique_ptr<cu::CudaKVPool> make_pool(int initial_rows) {
    return std::make_unique<cu::CudaKVPool>(
        std::vector<cu::CudaKVPool::Layer>{
            {kHeads, kDim, kDeclaredRows, /*growable=*/true},
            {kHeads, kDim, kFixedRows, /*growable=*/false},
        },
        std::vector<cu::CudaKVPool::SideBuffer>{
            {"__et_offgraph_kv_cells", slimc10::ScalarType::Long, {16}},
            {"__et_offgraph_kv_mask_w0",
             slimc10::ScalarType::Bool,
             {1, 1, 16, kDeclaredRows}},
        },
        slimc10::ScalarType::BFloat16,
        initial_rows);
  }

  // What a program lowered for make_pool() declares, for `fqns`.
  static void declare_pool(FakeContainer& container) {
    const auto bf16 = slimc10::ScalarType::BFloat16;
    for (const char* fqn :
         {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"}) {
      declare(container, fqn, bf16, {1, kDeclaredRows, kHeads, kDim});
    }
    for (const char* fqn :
         {"__et_offgraph_kv_layer_1_k", "__et_offgraph_kv_layer_1_v"}) {
      declare(container, fqn, bf16, {1, kFixedRows, kHeads, kDim});
    }
    declare(
        container, "__et_offgraph_kv_cells", slimc10::ScalarType::Long, {16});
    declare(
        container,
        "__et_offgraph_kv_mask_w0",
        slimc10::ScalarType::Bool,
        {1, 1, 16, kDeclaredRows});
  }

  static FakeContainer full_container() {
    FakeContainer container{
        {"grow_k", "grow_v", "fixed_k", "fixed_v", "cells", "mask"},
        {"__et_offgraph_kv_layer_0_k",
         "__et_offgraph_kv_layer_0_v",
         "__et_offgraph_kv_layer_1_k",
         "__et_offgraph_kv_layer_1_v",
         "__et_offgraph_kv_cells",
         "__et_offgraph_kv_mask_w0"},
        {},
        0};
    declare_pool(container);
    return container;
  }
};

} // namespace

namespace {

class CudaPrivatePoolTest : public CudaKVCacheTest {
 protected:
  static constexpr int kOwners = 64;
  static constexpr int kColumns = kOwners + 1;
  static constexpr int kFlatRows = 32;
  static constexpr int kRingRows = 12;
  static constexpr int kWindow = 5;

  std::unique_ptr<cu::CudaKVPool> make_private_pool(bool huge_stage = false) {
    return std::make_unique<cu::CudaKVPool>(
        std::vector<cu::CudaKVPool::Layer>{
            {kHeads, kDim, kFlatRows * kColumns, true, 0},
            {kHeads, kDim, kRingRows * kColumns, true, kWindow}},
        std::vector<cu::CudaKVPool::SideBuffer>{
            {"__et_offgraph_kv_private_layout",
             slimc10::ScalarType::Int,
             {kColumns, 2}},
            {"__et_offgraph_kv_stage_k",
             slimc10::ScalarType::BFloat16,
             {1,
              huge_stage ? 1 << 20 : 8,
              huge_stage ? 1 : kHeads,
              huge_stage ? 1 << 20 : kDim},
             true},
            {"__et_offgraph_kv_stage_v",
             slimc10::ScalarType::BFloat16,
             {1, huge_stage ? 1 << 20 : 8, kHeads, kDim},
             true}},
        slimc10::ScalarType::BFloat16,
        2 * kColumns,
        kColumns,
        kColumns,
        true);
  }

  static FakeContainer private_container(bool huge_stage = false) {
    FakeContainer container{
        {"flat_k",
         "flat_v",
         "ring_k",
         "ring_v",
         "layout",
         "stage_k",
         "stage_v"},
        {"__et_offgraph_kv_layer_0_k",
         "__et_offgraph_kv_layer_0_v",
         "__et_offgraph_kv_layer_1_k",
         "__et_offgraph_kv_layer_1_v",
         "__et_offgraph_kv_private_layout",
         "__et_offgraph_kv_stage_k",
         "__et_offgraph_kv_stage_v"},
        {},
        0};
    for (const char* suffix : {"k", "v"}) {
      declare(
          container,
          cu::offgraph_kv_layer_fqn(0, suffix),
          slimc10::ScalarType::BFloat16,
          {1, kFlatRows * kColumns, kHeads, kDim});
      declare(
          container,
          cu::offgraph_kv_layer_fqn(1, suffix),
          slimc10::ScalarType::BFloat16,
          {1, kRingRows * kColumns, kHeads, kDim});
    }
    declare(
        container,
        "__et_offgraph_kv_private_layout",
        slimc10::ScalarType::Int,
        {kColumns, 2});
    for (const char* fqn :
         {"__et_offgraph_kv_stage_k", "__et_offgraph_kv_stage_v"}) {
      declare(
          container,
          fqn,
          slimc10::ScalarType::BFloat16,
          huge_stage
              ? (std::string(fqn) == "__et_offgraph_kv_stage_k"
                     ? std::initializer_list<int64_t>{1, 1 << 20, 1, 1 << 20}
                     : std::initializer_list<int64_t>{1, 1 << 20, kHeads, kDim})
              : std::initializer_list<int64_t>{1, 8, kHeads, kDim});
    }
    return container;
  }

  static void write_row(
      cu::CudaKVPool& pool,
      size_t layer,
      size_t owner,
      int row,
      uint16_t tag) {
    const auto& span = pool.private_layout(layer).spans.at(owner);
    ASSERT_LT(row, span.capacity);
    for (bool value : {false, true}) {
      std::vector<uint16_t> payload(kHeads * kDim);
      for (size_t i = 0; i < payload.size(); ++i) {
        payload[i] = tag + i + (value ? 10000 : 0);
      }
      auto* dst = static_cast<uint16_t*>(pool.layer_data(layer, value)) +
          (span.base + row) * kHeads * kDim;
      ASSERT_EQ(
          cudaMemcpy(
              dst,
              payload.data(),
              payload.size() * sizeof(uint16_t),
              cudaMemcpyHostToDevice),
          cudaSuccess);
    }
  }

  static void expect_row(
      cu::CudaKVPool& pool,
      size_t layer,
      size_t owner,
      int row,
      uint16_t tag) {
    const auto& span = pool.private_layout(layer).spans.at(owner);
    ASSERT_LT(row, span.capacity);
    for (bool value : {false, true}) {
      std::vector<uint16_t> actual(kHeads * kDim);
      const auto* src =
          static_cast<const uint16_t*>(pool.layer_data(layer, value)) +
          (span.base + row) * kHeads * kDim;
      ASSERT_EQ(
          cudaMemcpy(
              actual.data(),
              src,
              actual.size() * sizeof(uint16_t),
              cudaMemcpyDeviceToHost),
          cudaSuccess);
      for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i], tag + i + (value ? 10000 : 0));
      }
    }
  }
};

} // namespace

TEST_F(
    CudaPrivatePoolTest,
    Heterogeneous64OwnersKeepTheirOwnCapacityAndPayload) {
  auto pool = make_private_pool();
  auto container = private_container();
  auto handle = make_handle(container);
  ASSERT_TRUE(pool->note_handle(&handle).get());
  ASSERT_EQ(pool->validate(), Error::Ok);
  std::vector<std::vector<int64_t>> rows(2, std::vector<int64_t>(kOwners, 0));
  std::vector<int64_t> live(kOwners, 0);
  rows[0][0] = rows[1][0] = 1;
  ASSERT_EQ(
      pool->prepare_private(rows, live, 1, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(pool->private_layout(0).rows, 3);
  EXPECT_EQ(pool->private_layout(1).rows, 3);
  for (auto& layer : rows) {
    std::fill(layer.begin(), layer.end(), 1);
  }
  live[0] = 1;
  ASSERT_EQ(
      pool->prepare_private(rows, live, 1, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(pool->private_layout(0).rows, 2 * kOwners + 1);
  for (size_t owner = 0; owner < kOwners; ++owner) {
    write_row(*pool, 0, owner, 0, owner * 100);
    write_row(*pool, 1, owner, 0, owner * 100 + 50);
  }
  ASSERT_EQ(pool->mark_step_done(cudaStreamPerThread), Error::Ok);
  std::fill(live.begin(), live.end(), 1);
  rows[0][17] = 17;
  rows[1][17] = kWindow;
  ASSERT_EQ(
      pool->prepare_private(rows, live, 1, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(pool->private_layout(0).spans[17].capacity, 17);
  EXPECT_EQ(pool->private_layout(1).spans[17].capacity, kWindow);
  for (size_t owner = 0; owner < kOwners; ++owner) {
    if (owner != 17) {
      EXPECT_EQ(pool->private_layout(0).spans[owner].capacity, 2);
      EXPECT_EQ(pool->private_layout(1).spans[owner].capacity, 2);
    }
    expect_row(*pool, 0, owner, 0, owner * 100);
    expect_row(*pool, 1, owner, 0, owner * 100 + 50);
  }
  EXPECT_EQ(pool->private_layout(0).rows, 4 * kOwners + 1);
  pool->forget_handle(&handle);
}

TEST_F(
    CudaPrivatePoolTest,
    CompactionRetainsWrappedRingRowsAndStableStageAddresses) {
  auto pool = make_private_pool();
  auto container = private_container();
  auto handle = make_handle(container);
  ASSERT_TRUE(pool->note_handle(&handle).get());
  std::vector<std::vector<int64_t>> rows(2, std::vector<int64_t>(kOwners, 0));
  rows[0][0] = rows[0][1] = 11;
  rows[1][0] = rows[1][1] = kWindow;
  std::vector<int64_t> live(kOwners, 0);
  ASSERT_EQ(
      pool->prepare_private(rows, live, 8, cudaStreamPerThread), Error::Ok);
  for (int row = 0; row < kWindow; ++row) {
    write_row(*pool, 1, 1, row, 200 + row);
  }
  const int64_t before = pool->allocated_bytes();
  void* stage_k = pool->side_buffer(1);
  void* stage_v = pool->side_buffer(2);
  ASSERT_EQ(pool->mark_step_done(cudaStreamPerThread), Error::Ok);
  rows[0][0] = rows[1][0] = 0;
  live[1] = 11;
  ASSERT_EQ(
      pool->prepare_private(rows, live, 0, cudaStreamPerThread, true),
      Error::Ok);
  EXPECT_EQ(pool->private_layout(0).spans[0].capacity, 0);
  EXPECT_EQ(pool->private_layout(1).spans[1].base, 0);
  EXPECT_LT(pool->allocated_bytes(), before);
  EXPECT_EQ(pool->side_buffer(1), stage_k);
  EXPECT_EQ(pool->side_buffer(2), stage_v);
  for (int row = 0; row < kWindow; ++row) {
    expect_row(*pool, 1, 1, row, 200 + row);
  }
  pool->forget_handle(&handle);
}

TEST_F(
    CudaPrivatePoolTest,
    StageGrowthMovesOnlyStageAndInvalidatesReplayBindings) {
  auto pool = make_private_pool();
  auto container = private_container();
  auto handle = make_handle(container);
  ASSERT_TRUE(pool->note_handle(&handle).get());
  std::vector<std::vector<int64_t>> rows(2, std::vector<int64_t>(kOwners, 0));
  rows[0][0] = 10;
  rows[1][0] = kWindow;
  std::vector<int64_t> live(kOwners, 0);
  ASSERT_EQ(
      pool->prepare_private(rows, live, 1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(pool->bind(&handle), Error::Ok);
  void* key = pool->layer_data(0, false);
  void* layout = pool->side_buffer(0);
  void* stage = pool->side_buffer(1);
  EXPECT_EQ(pool->side_buffer_bytes(1), 2 * kHeads * kDim * sizeof(uint16_t));
  handle.cuda_graph_state.phase = cu::CudaGraphPhase::Replay;
  handle.cuda_graph_state.warmup_remaining = 0;
  ASSERT_EQ(
      pool->prepare_private(rows, live, 5, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(pool->layer_data(0, false), key);
  EXPECT_EQ(pool->side_buffer(0), layout);
  EXPECT_NE(pool->side_buffer(1), stage);
  EXPECT_EQ(pool->side_buffer_bytes(1), 5 * kHeads * kDim * sizeof(uint16_t));
  EXPECT_EQ(handle.cuda_graph_state.phase, cu::CudaGraphPhase::Warmup);
  ASSERT_EQ(pool->bind(&handle), Error::Ok);
  EXPECT_EQ(container.bound["layout"].data, layout);
  EXPECT_EQ(container.bound["stage_k"].data, pool->side_buffer(1));
  stage = pool->side_buffer(1);
  handle.cuda_graph_state.warmup_remaining = 0;
  ASSERT_EQ(
      pool->prepare_private(rows, live, 4, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(pool->side_buffer(1), stage);
  EXPECT_EQ(handle.cuda_graph_state.warmup_remaining, 0);
  pool->forget_handle(&handle);
}

TEST_F(CudaPrivatePoolTest, StaggeredAdmissionsAmortizeGraphInvalidation) {
  auto pool = make_private_pool();
  auto fake = private_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(pool->note_handle(&handle).get());
  std::vector<std::vector<int64_t>> rows(2, std::vector<int64_t>(kOwners, 0));
  std::vector<int64_t> live(kOwners, 0);
  int moves = 0;
  for (int owner = 0; owner < kOwners; ++owner) {
    const bool allocated = pool->allocated();
    void* old_key = allocated ? pool->layer_data(0, false) : nullptr;
    const auto previous = allocated ? pool->private_layout(0).spans
                                    : std::vector<cu::PrivateRowSpan>{};
    rows[0][owner] = rows[1][owner] = 1;
    handle.cuda_graph_state.phase = cu::CudaGraphPhase::Replay;
    ASSERT_EQ(
        pool->prepare_private(rows, live, 1, cudaStreamPerThread), Error::Ok);
    ASSERT_EQ(pool->bind(&handle), Error::Ok);
    const bool moved = old_key != pool->layer_data(0, false);
    moves += moved;
    if (allocated) {
      EXPECT_EQ(
          handle.cuda_graph_state.phase,
          moved ? cu::CudaGraphPhase::Warmup : cu::CudaGraphPhase::Replay);
    }
    for (int old = 0; old < owner; ++old) {
      EXPECT_EQ(pool->private_layout(0).spans[old].base, previous[old].base);
      expect_row(*pool, 0, old, 0, old * 100);
      expect_row(*pool, 1, old, 0, old * 100 + 50);
    }
    write_row(*pool, 0, owner, 0, owner * 100);
    write_row(*pool, 1, owner, 0, owner * 100 + 50);
    live[owner] = 1;
    ASSERT_EQ(pool->mark_step_done(cudaStreamPerThread), Error::Ok);
  }
  EXPECT_LE(moves, 7);
  pool->forget_handle(&handle);
}

TEST_F(CudaPrivatePoolTest, PrivateRelocationWithinArenaPreservesReplay) {
  auto pool = make_private_pool();
  auto fake = private_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(pool->note_handle(&handle).get());
  std::vector<std::vector<int64_t>> rows(2, std::vector<int64_t>(kOwners, 0));
  std::vector<int64_t> live(kOwners, 0);
  for (int owner = 0; owner < 3; ++owner) {
    rows[0][owner] = rows[1][owner] = 1;
  }
  ASSERT_EQ(
      pool->prepare_private(rows, live, 1, cudaStreamPerThread), Error::Ok);
  for (int owner = 0; owner < 3; ++owner) {
    write_row(*pool, 0, owner, 0, owner * 100);
    write_row(*pool, 1, owner, 0, owner * 100 + 50);
    live[owner] = 1;
  }
  ASSERT_EQ(pool->mark_step_done(cudaStreamPerThread), Error::Ok);
  rows[0][3] = rows[1][3] = 1;
  ASSERT_EQ(
      pool->prepare_private(rows, live, 1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(pool->bind(&handle), Error::Ok);
  write_row(*pool, 0, 3, 0, 300);
  write_row(*pool, 1, 3, 0, 350);
  live[3] = 1;
  ASSERT_EQ(pool->mark_step_done(cudaStreamPerThread), Error::Ok);
  const auto previous = pool->private_layout(0).spans;
  void* key = pool->layer_data(0, false);
  void* ring = pool->layer_data(1, false);
  const int updates = fake.updates;
  handle.cuda_graph_state.phase = cu::CudaGraphPhase::Replay;
  rows[0][0] = rows[1][0] = 3;
  ASSERT_EQ(
      pool->prepare_private(rows, live, 1, cudaStreamPerThread), Error::Ok);
  EXPECT_NE(pool->private_layout(0).spans[0].base, previous[0].base);
  EXPECT_EQ(pool->private_layout(0).spans[0].capacity, 4);
  EXPECT_EQ(pool->layer_data(0, false), key);
  EXPECT_EQ(pool->layer_data(1, false), ring);
  EXPECT_EQ(handle.cuda_graph_state.phase, cu::CudaGraphPhase::Replay);
  ASSERT_EQ(pool->bind(&handle), Error::Ok);
  EXPECT_EQ(fake.updates, updates);
  for (int owner = 0; owner < 4; ++owner) {
    if (owner > 0) {
      EXPECT_EQ(
          pool->private_layout(0).spans[owner].base, previous[owner].base);
    }
    expect_row(*pool, 0, owner, 0, owner * 100);
    expect_row(*pool, 1, owner, 0, owner * 100 + 50);
  }
  pool->forget_handle(&handle);
}

TEST_F(
    CudaPrivatePoolTest,
    InvalidPreparationLeavesCommittedPointersAndLayoutsUsable) {
  auto pool = make_private_pool();
  auto container = private_container();
  auto handle = make_handle(container);
  ASSERT_TRUE(pool->note_handle(&handle).get());
  std::vector<std::vector<int64_t>> rows(2, std::vector<int64_t>(kOwners, 0));
  rows[0][0] = rows[1][0] = 1;
  std::vector<int64_t> live(kOwners, 0);
  ASSERT_EQ(
      pool->prepare_private(rows, live, 1, cudaStreamPerThread), Error::Ok);
  const int64_t bytes = pool->allocated_bytes();
  void* key = pool->layer_data(0, false);
  void* stage = pool->side_buffer(1);
  rows[1][7] = -1;
  EXPECT_EQ(
      pool->prepare_private(rows, live, 1, cudaStreamPerThread),
      Error::InvalidArgument);
  rows[1][7] = 0;
  EXPECT_EQ(
      pool->prepare_private(rows, live, 9, cudaStreamPerThread),
      Error::InvalidArgument);
  EXPECT_EQ(pool->allocated_bytes(), bytes);
  EXPECT_EQ(pool->layer_data(0, false), key);
  EXPECT_EQ(pool->side_buffer(1), stage);
  EXPECT_EQ(pool->private_layout(0).spans[0].capacity, 2);
  ASSERT_EQ(
      pool->prepare_private(rows, live, 1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(pool->bind(&handle), Error::Ok);
  pool->forget_handle(&handle);
}

TEST_F(CudaPrivatePoolTest, FailedStageGrowthFencesUncommittedArenaCopies) {
  auto pool = make_private_pool(/*huge_stage=*/true);
  auto fake = private_container(/*huge_stage=*/true);
  auto handle = make_handle(fake);
  ASSERT_TRUE(pool->note_handle(&handle).get());
  std::vector<std::vector<int64_t>> rows(2, std::vector<int64_t>(kOwners, 0));
  std::vector<int64_t> live(kOwners, 0);
  for (int owner = 0; owner < 3; ++owner) {
    rows[0][owner] = rows[1][owner] = 1;
  }
  ASSERT_EQ(
      pool->prepare_private(rows, live, 1, cudaStreamPerThread), Error::Ok);
  for (int owner = 0; owner < 3; ++owner) {
    write_row(*pool, 0, owner, 0, owner * 100);
    write_row(*pool, 1, owner, 0, owner * 100 + 50);
    live[owner] = 1;
  }
  ASSERT_EQ(pool->mark_step_done(cudaStreamPerThread), Error::Ok);
  rows[0][3] = rows[1][3] = 1;
  ASSERT_EQ(
      pool->prepare_private(rows, live, 1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(pool->bind(&handle), Error::Ok);
  write_row(*pool, 0, 3, 0, 300);
  write_row(*pool, 1, 3, 0, 350);
  live[3] = 1;
  ASSERT_EQ(pool->mark_step_done(cudaStreamPerThread), Error::Ok);
  const auto previous = pool->private_layout(0).spans;
  const auto bytes = pool->allocated_bytes();
  void* key = pool->layer_data(0, false);
  const int updates = fake.updates;
  handle.cuda_graph_state.phase = cu::CudaGraphPhase::Replay;
  rows[0][0] = rows[1][0] = 3;
  cu::CudaKVPoolTestPeer fences;
  // The owner copy uses spare rows of the existing arena. Only the later
  // 2-TiB staging allocation fails, after that copy has been enqueued.
  EXPECT_EQ(
      pool->prepare_private(rows, live, 1 << 20, cudaStreamPerThread),
      Error::MemoryAllocationFailed);
  EXPECT_EQ(fences.device_fences, 2);
  EXPECT_FALSE(pool->requires_clear());
  EXPECT_EQ(pool->allocated_bytes(), bytes);
  EXPECT_EQ(pool->layer_data(0, false), key);
  EXPECT_EQ(pool->private_layout(0).spans[0].base, previous[0].base);
  EXPECT_EQ(pool->private_layout(0).spans[0].capacity, previous[0].capacity);
  EXPECT_EQ(handle.cuda_graph_state.phase, cu::CudaGraphPhase::Replay);
  cudaStream_t retry_stream;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&retry_stream, cudaStreamNonBlocking),
      cudaSuccess);
  EXPECT_EQ(pool->prepare_private(rows, live, 1, retry_stream), Error::Ok);
  EXPECT_EQ(pool->bind(&handle), Error::Ok);
  EXPECT_EQ(fake.updates, updates);
  EXPECT_NE(pool->private_layout(0).spans[0].base, previous[0].base);
  for (int owner = 0; owner < 4; ++owner) {
    expect_row(*pool, 0, owner, 0, owner * 100);
    expect_row(*pool, 1, owner, 0, owner * 100 + 50);
  }
  EXPECT_EQ(cudaStreamSynchronize(retry_stream), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(retry_stream), cudaSuccess);
  pool->forget_handle(&handle);
}

TEST_F(CudaKVPoolTest, SideBuffersBindAtFixedAddressesAcrossGrowth) {
  auto pool = make_pool(4);
  auto container = full_container();
  auto handle = make_handle(container);
  ASSERT_TRUE(pool->note_handle(&handle).get());
  ASSERT_EQ(pool->validate(), Error::Ok);

  ASSERT_EQ(pool->prepare(3, 0, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(pool->bind(&handle), Error::Ok);
  void* grow_k = container.bound["grow_k"].data;
  void* fixed_k = container.bound["fixed_k"].data;
  void* cells = container.bound["cells"].data;
  void* mask = container.bound["mask"].data;
  EXPECT_EQ(cells, pool->side_buffer(0));
  EXPECT_EQ(mask, pool->side_buffer(1));
  EXPECT_EQ(pool->side_buffer_bytes(0), 16 * sizeof(int64_t));
  EXPECT_EQ(container.bound["cells"].sizes, std::vector<int64_t>({16}));
  const std::vector<int64_t> mask_sizes{1, 1, 16, kDeclaredRows};
  EXPECT_EQ(container.bound["mask"].sizes, mask_sizes);
  EXPECT_EQ(container.bound["mask"].dtype, slimc10::ScalarType::Bool);
  const std::vector<int64_t> fixed_sizes{1, kFixedRows, kHeads, kDim};
  EXPECT_EQ(container.bound["fixed_k"].sizes, fixed_sizes);

  // Side buffers start zeroed.
  std::vector<int64_t> cells_host(16, -1);
  ASSERT_EQ(
      cudaMemcpy(
          cells_host.data(),
          cells,
          cells_host.size() * sizeof(int64_t),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(cells_host, std::vector<int64_t>(16, 0));
  ASSERT_EQ(pool->mark_step_done(pool->stream()), Error::Ok);

  // Growth moves only the growable layer.
  ASSERT_EQ(pool->prepare(5, 3, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(pool->rows(), 8);
  EXPECT_EQ(pool->growth_count(), 1);
  ASSERT_EQ(pool->bind(&handle), Error::Ok);
  EXPECT_NE(container.bound["grow_k"].data, grow_k);
  EXPECT_EQ(container.bound["fixed_k"].data, fixed_k);
  EXPECT_EQ(container.bound["cells"].data, cells);
  EXPECT_EQ(container.bound["mask"].data, mask);
  pool->forget_handle(&handle);
}

TEST_F(CudaKVPoolTest, GrowthCarriesLiveRowsAndCapsAtDeclaredRows) {
  auto pool = make_pool(4);
  auto container = full_container();
  auto handle = make_handle(container);
  ASSERT_TRUE(pool->note_handle(&handle).get());
  ASSERT_EQ(pool->prepare(4, 0, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(pool->bind(&handle), Error::Ok);
  const auto history = iota_rows(3);
  ASSERT_EQ(
      cudaMemcpy(
          container.bound["grow_k"].data,
          history.data(),
          history.size() * sizeof(uint16_t),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(pool->mark_step_done(pool->stream()), Error::Ok);

  // Asking past twice the rows grows straight to what is asked; past the
  // declared rows it is capped there.
  ASSERT_EQ(pool->prepare(20, 3, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(pool->rows(), 20);
  ASSERT_EQ(pool->bind(&handle), Error::Ok);
  EXPECT_EQ(read_rows(container.bound["grow_k"].data, 3), history);
  ASSERT_EQ(pool->prepare(kDeclaredRows, 3, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(pool->rows(), kDeclaredRows);
  EXPECT_EQ(
      pool->allocated_bytes(),
      2 * (kDeclaredRows + kFixedRows) * kRow *
          static_cast<int64_t>(sizeof(uint16_t)));
  pool->forget_handle(&handle);
}

TEST_F(CudaKVPoolTest, CompiledSizeRoundedUpBy64IsAccepted) {
  // AOTI reports constant bytes rounded up to 64 when the program also holds
  // CPU constants: a 16-byte read_len buffer reads as 64. Its serialized
  // shape still matches what the pool declares.
  cu::CudaKVPool pool(
      {{kHeads, kDim, kDeclaredRows, /*growable=*/true}},
      {{"__et_offgraph_kv_read_len", slimc10::ScalarType::Long, {2}}},
      slimc10::ScalarType::BFloat16,
      4);
  FakeContainer container{
      {"k", "v", "read_len"},
      {"__et_offgraph_kv_layer_0_k",
       "__et_offgraph_kv_layer_0_v",
       "__et_offgraph_kv_read_len"},
      {},
      0};
  for (const char* fqn :
       {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"}) {
    declare(
        container,
        fqn,
        slimc10::ScalarType::BFloat16,
        {1, kDeclaredRows, kHeads, kDim});
  }
  declare(
      container, "__et_offgraph_kv_read_len", slimc10::ScalarType::Long, {2});
  container.declared["__et_offgraph_kv_read_len"].second = 64;
  auto handle = make_handle(container);
  EXPECT_TRUE(pool.note_handle(&handle).get());

  // Rounded past the next multiple of 64: the library and its metadata
  // disagree.
  container.declared["__et_offgraph_kv_read_len"].second = 128;
  EXPECT_EQ(pool.note_handle(&handle).error(), Error::InvalidProgram);
  pool.forget_handle(&handle);
}

TEST_F(CudaKVPoolTest, ShapeHiddenByPaddingIsRejected) {
  // A read_len compiled at {4} spans 32 bytes, and the pool's {2} spans 16;
  // padded, both read as 64. The serialized shape tells them apart.
  cu::CudaKVPool pool(
      {{kHeads, kDim, kDeclaredRows, /*growable=*/true}},
      {{"__et_offgraph_kv_read_len", slimc10::ScalarType::Long, {2}}},
      slimc10::ScalarType::BFloat16,
      4);
  FakeContainer container{
      {"k", "v", "read_len"},
      {"__et_offgraph_kv_layer_0_k",
       "__et_offgraph_kv_layer_0_v",
       "__et_offgraph_kv_read_len"},
      {},
      0};
  for (const char* fqn :
       {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"}) {
    declare(
        container,
        fqn,
        slimc10::ScalarType::BFloat16,
        {1, kDeclaredRows, kHeads, kDim});
  }
  declare(
      container, "__et_offgraph_kv_read_len", slimc10::ScalarType::Long, {4});
  container.declared["__et_offgraph_kv_read_len"].second = 64;
  auto handle = make_handle(container);
  EXPECT_EQ(pool.note_handle(&handle).error(), Error::InvalidProgram);

  // Nor is a constant accepted without a serialized shape to check.
  handle.offgraph_kv_sizes.erase("__et_offgraph_kv_read_len");
  EXPECT_EQ(pool.note_handle(&handle).error(), Error::InvalidProgram);
}

TEST_F(CudaKVPoolTest, ConstantNotMatchingItsCompiledSizeIsRejected) {
  // A side buffer compiled at another shape than the pool allocates would
  // be addressed past its allocation.
  auto pool = make_pool(4);
  auto container = full_container();
  declare(
      container,
      "__et_offgraph_kv_mask_w0",
      slimc10::ScalarType::Bool,
      {1, 1, 16, 2 * kDeclaredRows});
  auto handle = make_handle(container);
  EXPECT_EQ(pool->note_handle(&handle).error(), Error::InvalidProgram);

  // And one compiled with another dtype.
  auto wrong_dtype = full_container();
  declare(
      wrong_dtype, "__et_offgraph_kv_cells", slimc10::ScalarType::Int, {16});
  auto wrong_dtype_handle = make_handle(wrong_dtype);
  EXPECT_EQ(
      pool->note_handle(&wrong_dtype_handle).error(), Error::InvalidProgram);
}

TEST_F(CudaKVPoolTest, PrivateMetadataCannotBeBoundToLegacyStorage) {
  auto pool = make_pool(4);
  auto container = full_container();
  container.internal_names.push_back("owner_step");
  container.fqns.push_back("__et_offgraph_kv_owner_step");
  declare(
      container,
      "__et_offgraph_kv_owner_step",
      slimc10::ScalarType::Int,
      {5, 3});
  auto handle = make_handle(container);
  EXPECT_EQ(pool->note_handle(&handle).error(), Error::InvalidProgram);
  EXPECT_FALSE(pool->serves(&handle));
}

TEST_F(CudaKVPoolTest, NonCacheConstantsDoNotNeedRuntimeBinding) {
  auto pool = make_pool(4);
  auto container = full_container();
  container.internal_names.push_back("weight");
  container.fqns.push_back("attention.weight");
  declare(container, "attention.weight", slimc10::ScalarType::BFloat16, {16});
  auto handle = make_handle(container);
  ASSERT_TRUE(pool->note_handle(&handle).get());
  EXPECT_EQ(pool->validate(), Error::Ok);
  pool->forget_handle(&handle);
}

TEST_F(CudaKVPoolTest, ProgramMissingSideBuffersIsRejected) {
  auto pool = make_pool(4);
  // Every layer, but only one of the two side buffers: the lowering and the
  // runtime disagree about the layout.
  FakeContainer partial{
      {"grow_k", "grow_v", "fixed_k", "fixed_v", "cells"},
      {"__et_offgraph_kv_layer_0_k",
       "__et_offgraph_kv_layer_0_v",
       "__et_offgraph_kv_layer_1_k",
       "__et_offgraph_kv_layer_1_v",
       "__et_offgraph_kv_cells"},
      {},
      0};
  declare_pool(partial);
  auto handle = make_handle(partial);
  EXPECT_EQ(pool->note_handle(&handle).error(), Error::InvalidProgram);

  // A program with no storage at all is simply not served.
  FakeContainer embedding{{"weight"}, {"tok_embeddings.weight"}, {}, 0};
  auto embedding_handle = make_handle(embedding);
  const auto serves = pool->note_handle(&embedding_handle);
  ASSERT_EQ(serves.error(), Error::Ok);
  EXPECT_FALSE(serves.get());
  // Neither program contributed storage, the rejected one included.
  EXPECT_EQ(pool->validate(), Error::InvalidProgram);
}

TEST_F(CudaKVPoolTest, FailedSideBufferAllocationLeavesThePoolRetryable) {
  // A side buffer too large for any device fails the first allocation, which
  // must release the layers it already allocated.
  cu::CudaKVPool pool(
      {{kHeads, kDim, kDeclaredRows, /*growable=*/true}},
      {{"__et_offgraph_kv_huge",
        slimc10::ScalarType::Byte,
        {int64_t{1} << 50}}},
      slimc10::ScalarType::BFloat16,
      4);
  FakeContainer container{
      {"k", "v", "huge"},
      {"__et_offgraph_kv_layer_0_k",
       "__et_offgraph_kv_layer_0_v",
       "__et_offgraph_kv_huge"},
      {},
      0};
  for (const char* fqn :
       {"__et_offgraph_kv_layer_0_k", "__et_offgraph_kv_layer_0_v"}) {
    declare(
        container,
        fqn,
        slimc10::ScalarType::BFloat16,
        {1, kDeclaredRows, kHeads, kDim});
  }
  declare(
      container,
      "__et_offgraph_kv_huge",
      slimc10::ScalarType::Byte,
      {int64_t{1} << 50});
  auto handle = make_handle(container);
  ASSERT_TRUE(pool.note_handle(&handle).get());
  EXPECT_NE(pool.prepare(1, 0, cudaStreamPerThread), Error::Ok);
  EXPECT_FALSE(pool.allocated());
  EXPECT_EQ(pool.allocated_bytes(), 0);
  pool.forget_handle(&handle);
}

namespace {

// A flat layer and a window-2 layer over 16 cells, steps of up to 8 tokens.
class CudaCellCacheTest : public CudaKVCacheTest {
 protected:
  static constexpr int kCells = 16;
  static constexpr int kMaxWrite = 8;
  static constexpr int kWindow = 2;

  static std::shared_ptr<cache::Cache> make(
      int initial = 4,
      std::optional<int> block_size = std::nullopt) {
    auto geometry = flat_and_ring(kWindow);
    geometry.dflash_block_size = block_size;
    return cu::make_cuda_cell_kv_cache(
        geometry, config(kCells, initial, kMaxWrite));
  }

  // What a program lowered in the cell layout for make() declares: every
  // layer at kCells rows, and the step buffers at the widest step.
  static FakeContainer container() {
    FakeContainer container{
        {"f_k", "f_v", "r_k", "r_v", "cells", "read_len", "mask0", "mask2"},
        {"__et_offgraph_kv_layer_0_k",
         "__et_offgraph_kv_layer_0_v",
         "__et_offgraph_kv_layer_1_k",
         "__et_offgraph_kv_layer_1_v",
         "__et_offgraph_kv_cells",
         "__et_offgraph_kv_read_len",
         "__et_offgraph_kv_mask_w0",
         "__et_offgraph_kv_mask_w2"},
        {},
        0};
    for (size_t index = 0; index < 4; ++index) {
      declare(
          container,
          container.fqns[index],
          slimc10::ScalarType::BFloat16,
          {1, kCells, kHeads, kDim});
    }
    declare(
        container,
        "__et_offgraph_kv_cells",
        slimc10::ScalarType::Long,
        {kMaxWrite});
    declare(
        container, "__et_offgraph_kv_read_len", slimc10::ScalarType::Long, {1});
    for (const char* mask :
         {"__et_offgraph_kv_mask_w0", "__et_offgraph_kv_mask_w2"}) {
      declare(
          container,
          mask,
          slimc10::ScalarType::Bool,
          {1, 1, kMaxWrite, kCells});
    }
    return container;
  }

  static std::vector<int64_t> read_longs(void* device, int count) {
    std::vector<int64_t> values(count);
    EXPECT_EQ(
        cudaMemcpy(
            values.data(),
            device,
            count * sizeof(int64_t),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    return values;
  }

  // Rows [0, rows) over columns [0, cols) of a [kMaxWrite, kCells] mask.
  static std::vector<std::vector<int>>
  read_mask(void* device, int rows, int cols) {
    std::vector<uint8_t> flat(static_cast<size_t>(kMaxWrite) * kCells);
    EXPECT_EQ(
        cudaMemcpy(flat.data(), device, flat.size(), cudaMemcpyDeviceToHost),
        cudaSuccess);
    std::vector<std::vector<int>> out(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        out[i][j] = flat[static_cast<size_t>(i) * kCells + j];
      }
    }
    return out;
  }

  // declare + prepare + bind, as the executor and the delegate do per forward.
  // `width` is the program's step width, the declared tokens by default; a
  // wider program pads.
  static Error step(
      cache::Cache& cache,
      cu::CudaDelegateHandle& handle,
      const std::vector<int32_t>& seq_ids,
      int64_t width = 0) {
    auto* control = cache.as<cache::BatchControl>();
    auto* kv = cache.as<cu::CudaKVCache>();
    if (!control->declare_step(seq_ids)) {
      return Error::InvalidArgument;
    }
    if (width == 0) {
      width = static_cast<int64_t>(seq_ids.size());
    }
    ET_CHECK_OK_OR_RETURN_ERROR(kv->prepare_step(width, cudaStreamPerThread));
    ET_CHECK_OK_OR_RETURN_ERROR(kv->rebind_for_execute(&handle));
    return kv->commit_step(width);
  }
};

} // namespace

TEST_F(CudaCellCacheTest, InterleavedSequencesWritePlacementAndMasks) {
  auto cache_ptr = make();
  ASSERT_NE(cache_ptr, nullptr);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  ASSERT_NE(control, nullptr);
  ASSERT_NE(kv, nullptr);
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();
  const int32_t b = *control->seq_new();

  // Both prefill in one forward.
  ASSERT_EQ(step(*cache_ptr, handle, {a, a, a, b, b}), Error::Ok);
  EXPECT_EQ(
      read_longs(fake.bound["cells"].data, 5),
      std::vector<int64_t>({0, 1, 2, 3, 4}));
  EXPECT_EQ(
      read_longs(fake.bound["read_len"].data, 1), std::vector<int64_t>({5}));
  using Mask = std::vector<std::vector<int>>;
  EXPECT_EQ(
      read_mask(fake.bound["mask0"].data, 5, 5),
      Mask(
          {{1, 0, 0, 0, 0},
           {1, 1, 0, 0, 0},
           {1, 1, 1, 0, 0},
           {0, 0, 0, 1, 0},
           {0, 0, 0, 1, 1}}));
  // The window-2 layer drops a's position 0 for its position-2 query.
  EXPECT_EQ(
      read_mask(fake.bound["mask2"].data, 5, 5),
      Mask(
          {{1, 0, 0, 0, 0},
           {1, 1, 0, 0, 0},
           {0, 1, 1, 0, 0},
           {0, 0, 0, 1, 0},
           {0, 0, 0, 1, 1}}));
  EXPECT_EQ(
      fake.bound["mask0"].sizes,
      std::vector<int64_t>({1, 1, kMaxWrite, kCells}));
  EXPECT_EQ(fake.bound["cells"].dtype, slimc10::ScalarType::Long);
  EXPECT_EQ(fake.bound["mask0"].dtype, slimc10::ScalarType::Bool);
  // Pools are declared at every cell; allocated past the first step's reach,
  // plus the scratch row.
  EXPECT_EQ(
      fake.bound["r_k"].sizes, std::vector<int64_t>({1, kCells, kHeads, kDim}));
  EXPECT_EQ(kv->metrics().flat_capacity, 6);

  // Then they decode together, in the other order.
  ASSERT_EQ(step(*cache_ptr, handle, {b, a}), Error::Ok);
  EXPECT_EQ(
      read_longs(fake.bound["cells"].data, 2), std::vector<int64_t>({5, 6}));
  EXPECT_EQ(
      read_longs(fake.bound["read_len"].data, 1), std::vector<int64_t>({7}));
  EXPECT_EQ(
      read_mask(fake.bound["mask0"].data, 2, 7),
      Mask({{0, 0, 0, 1, 1, 1, 0}, {1, 1, 1, 0, 0, 0, 1}}));
  EXPECT_EQ(control->pos(a), 4);
  EXPECT_EQ(control->pos(b), 3);
  const auto metrics = kv->metrics();
  EXPECT_EQ(metrics.logical_length, 7);
  EXPECT_EQ(metrics.flat_capacity, 12);
  EXPECT_EQ(metrics.growth_count, 1);
  kv->forget_handle(&handle);
}

TEST_F(CudaCellCacheTest, SwitchingSequencesKeepsACapturedGraph) {
  // Room for every step below but the last: the pool grows ahead of placement
  // to where the step could reach, occupied cells + width (no holes here).
  auto cache_ptr = make(/*initial=*/12);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto prefill_fake = container();
  auto decode_fake = container();
  auto prefill = make_handle(prefill_fake);
  auto decode = make_handle(decode_fake);
  ASSERT_TRUE(kv->note_handle(&prefill).get());
  ASSERT_TRUE(kv->note_handle(&decode).get());
  const int32_t a = *control->seq_new();
  const int32_t b = *control->seq_new();
  ASSERT_EQ(step(*cache_ptr, prefill, {a, a, b, b}), Error::Ok);

  ASSERT_EQ(step(*cache_ptr, decode, {a}), Error::Ok);
  auto& graph = decode.cuda_graph_state;
  graph.phase = cu::CudaGraphPhase::Replay;
  const auto bound = decode_fake.bound;

  // Another sequence, then both: the graph reads new placements from the same
  // addresses, so it stays captured.
  ASSERT_EQ(step(*cache_ptr, decode, {b}), Error::Ok);
  EXPECT_EQ(graph.phase, cu::CudaGraphPhase::Replay);
  ASSERT_EQ(step(*cache_ptr, prefill, {b, a}), Error::Ok);
  ASSERT_EQ(step(*cache_ptr, decode, {a}), Error::Ok);
  EXPECT_EQ(graph.phase, cu::CudaGraphPhase::Replay);
  for (const auto& [name, binding] : bound) {
    EXPECT_EQ(decode_fake.bound[name].data, binding.data) << name;
  }
  // The two methods share every buffer.
  for (const auto& [name, binding] : decode_fake.bound) {
    EXPECT_EQ(prefill_fake.bound[name].data, binding.data) << name;
  }

  // Growth moves the pools: the graph is captured again, and only the pools'
  // bindings change.
  ASSERT_EQ(step(*cache_ptr, prefill, {a, a, a, b, b}), Error::Ok);
  EXPECT_NE(graph.phase, cu::CudaGraphPhase::Replay);
  ASSERT_EQ(step(*cache_ptr, decode, {a}), Error::Ok);
  EXPECT_NE(decode_fake.bound["f_k"].data, bound.at("f_k").data);
  EXPECT_EQ(decode_fake.bound["cells"].data, bound.at("cells").data);
  EXPECT_EQ(decode_fake.bound["mask2"].data, bound.at("mask2").data);
  kv->forget_handle(&prefill);
  kv->forget_handle(&decode);
}

TEST_F(CudaCellCacheTest, StepIntoAFreedCellDoesNotGrow) {
  // Four cells holding a, a, b, b, plus the scratch row. Once a goes, b's
  // next token takes cell 0 and read_len stays 4, so the pool must not grow,
  // copy, or drop the graph.
  auto cache_ptr = make(/*initial=*/5);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();
  const int32_t b = *control->seq_new();
  ASSERT_EQ(step(*cache_ptr, handle, {a, a, b, b}), Error::Ok);
  ASSERT_EQ(kv->metrics().flat_capacity, 5);
  auto& graph = handle.cuda_graph_state;
  graph.phase = cu::CudaGraphPhase::Replay;
  const void* pool = fake.bound["f_k"].data;

  ASSERT_TRUE(control->seq_rm(a));
  ASSERT_EQ(step(*cache_ptr, handle, {b}), Error::Ok);

  EXPECT_EQ(read_longs(fake.bound["cells"].data, 1), std::vector<int64_t>({0}));
  EXPECT_EQ(
      read_longs(fake.bound["read_len"].data, 1), std::vector<int64_t>({4}));
  const auto metrics = kv->metrics();
  EXPECT_EQ(metrics.flat_capacity, 5);
  EXPECT_EQ(metrics.growth_count, 0);
  EXPECT_EQ(graph.phase, cu::CudaGraphPhase::Replay);
  EXPECT_EQ(fake.bound["f_k"].data, pool);
  kv->forget_handle(&handle);
}

TEST_F(CudaCellCacheTest, UploadsOutliveTheStepTheyWereBuiltFrom) {
  // The base's step is invalidated by the next verb, which may run before
  // the stream reaches the uploads. They must still deliver the step that
  // was prepared.
  auto cache_ptr = make(/*initial=*/8);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();
  const int32_t b = *control->seq_new();
  ASSERT_TRUE(control->declare_step({a, a, b}));
  ASSERT_EQ(kv->prepare_step(3, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  ASSERT_TRUE(control->seq_rm(b));
  ASSERT_TRUE(control->declare_step({a}));

  ASSERT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
  EXPECT_EQ(
      read_longs(fake.bound["cells"].data, 3), std::vector<int64_t>({0, 1, 2}));
  EXPECT_EQ(
      read_longs(fake.bound["read_len"].data, 1), std::vector<int64_t>({3}));
  EXPECT_EQ(
      read_mask(fake.bound["mask0"].data, 3, 3),
      std::vector<std::vector<int>>({{1, 0, 0}, {1, 1, 0}, {0, 0, 1}}));
  kv->forget_handle(&handle);
}

TEST_F(CudaCellCacheTest, ClonedSequenceSharesCellsUntilItsLastOwnerGoes) {
  auto cache_ptr = make(/*initial=*/8);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();
  ASSERT_EQ(step(*cache_ptr, handle, {a, a, a}), Error::Ok);

  const auto c = control->seq_clone(a, std::nullopt);
  ASSERT_TRUE(c.has_value());
  EXPECT_EQ(control->pos(*c), 3);
  ASSERT_EQ(step(*cache_ptr, handle, {*c}), Error::Ok);
  EXPECT_EQ(read_longs(fake.bound["cells"].data, 1), std::vector<int64_t>({3}));
  EXPECT_EQ(
      read_mask(fake.bound["mask0"].data, 1, 4),
      std::vector<std::vector<int>>({{1, 1, 1, 1}}));

  // The table holds every declared cell but the scratch row.
  auto* cells = static_cast<cache::CellCache*>(cache_ptr.get());
  EXPECT_EQ(cells->capacity(), kCells - 1);
  ASSERT_TRUE(control->seq_rm(a));
  EXPECT_EQ(cells->free_cells(), kCells - 1 - 4);
  ASSERT_TRUE(control->seq_rm(*c));
  EXPECT_EQ(cells->free_cells(), kCells - 1);
  EXPECT_EQ(cells->used_end(), 0);
  kv->forget_handle(&handle);
}

TEST_F(CudaCellCacheTest, RejectsStepsThatDisagreeWithTheDeclaration) {
  auto cache_ptr = make();
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();

  // Nothing declared.
  EXPECT_EQ(kv->prepare_step(1, cudaStreamPerThread), Error::InvalidState);
  // Declared three, the program takes two: a narrower program cannot run it.
  ASSERT_TRUE(control->declare_step({a, a, a}));
  EXPECT_EQ(kv->prepare_step(2, cudaStreamPerThread), Error::InvalidState);
  // Wider than the step buffers the program declared.
  EXPECT_EQ(
      kv->prepare_step(kMaxWrite + 1, cudaStreamPerThread),
      Error::InvalidArgument);
  // More tokens than free cells, or an id that was never handed out, never
  // declares; the scratch row is not free.
  EXPECT_FALSE(control->declare_step(std::vector<int32_t>(kCells, a)));
  EXPECT_FALSE(control->declare_step({-1}));
  // A rejected step leaves the table untouched.
  EXPECT_EQ(control->pos(a), 0);
  // And the earlier accepted declaration still stands.
  ASSERT_EQ(kv->prepare_step(3, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  ASSERT_EQ(kv->commit_step(3), Error::Ok);
  EXPECT_EQ(control->pos(a), 3);
  kv->forget_handle(&handle);
}

TEST_F(CudaCellCacheTest, PaddingRowsWriteTheScratchRowAndCopyTheLastMask) {
  auto cache_ptr = make(/*initial=*/12);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();
  const int32_t b = *control->seq_new();

  // Three tokens in a program of width 8: rows 3..7 are padding.
  ASSERT_EQ(step(*cache_ptr, handle, {a, a, b}, kMaxWrite), Error::Ok);
  const int64_t rows = kv->metrics().flat_capacity;
  const int64_t scratch = rows - 1;
  const auto cells = read_longs(fake.bound["cells"].data, kMaxWrite);
  EXPECT_EQ(
      cells,
      std::vector<int64_t>(
          {0, 1, 2, scratch, scratch, scratch, scratch, scratch}));
  const int64_t read_len = read_longs(fake.bound["read_len"].data, 1)[0];
  EXPECT_EQ(read_len, 3);
  // The scratch row lies past every cell the step reads.
  EXPECT_GE(scratch, read_len);
  using Mask = std::vector<std::vector<int>>;
  const Mask mask = read_mask(fake.bound["mask0"].data, kMaxWrite, 3);
  for (int row = 3; row < kMaxWrite; ++row) {
    EXPECT_EQ(mask[row], mask[2]) << row;
  }
  EXPECT_EQ(mask[2], std::vector<int>({0, 0, 1}));
  // Padding was never declared: the table saw three tokens.
  EXPECT_EQ(control->pos(a), 2);
  EXPECT_EQ(control->pos(b), 1);
  EXPECT_EQ(kv->metrics().logical_length, 3);
  kv->forget_handle(&handle);
}

TEST_F(CudaCellCacheTest, NarrowStepsNeverLeaveAWiderStepsCells) {
  auto cache_ptr = make(/*initial=*/16);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto wide_fake = container();
  auto narrow_fake = container();
  auto wide = make_handle(wide_fake);
  auto narrow = make_handle(narrow_fake);
  ASSERT_TRUE(kv->note_handle(&wide).get());
  ASSERT_TRUE(kv->note_handle(&narrow).get());
  const int32_t a = *control->seq_new();
  const int32_t b = *control->seq_new();

  // A full-width step fills cells 0..7 of the shared placement buffer.
  ASSERT_EQ(
      step(*cache_ptr, wide, {a, a, a, a, b, b, b, b}, kMaxWrite), Error::Ok);
  // Then one token through a width-4 program: rows 1..3 must not still name
  // cells 1..3 (a's history), or the forward would overwrite it.
  ASSERT_EQ(step(*cache_ptr, narrow, {b}, /*width=*/4), Error::Ok);
  const int64_t scratch = kv->metrics().flat_capacity - 1;
  EXPECT_EQ(
      read_longs(narrow_fake.bound["cells"].data, 4),
      std::vector<int64_t>({8, scratch, scratch, scratch}));
  const int64_t read_len = read_longs(narrow_fake.bound["read_len"].data, 1)[0];
  EXPECT_EQ(read_len, 9);
  EXPECT_GE(scratch, read_len);
  kv->forget_handle(&wide);
  kv->forget_handle(&narrow);
}

TEST_F(CudaCellCacheTest, ScratchRowStaysPastEveryReadAsThePoolGrows) {
  auto cache_ptr = make(/*initial=*/2);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();
  // Decode one token at a time through a width-2 program until the table is
  // full: every step pads one row, and the pool grows several times.
  for (int token = 0; token < kCells - 1; ++token) {
    ASSERT_EQ(step(*cache_ptr, handle, {a}, /*width=*/2), Error::Ok) << token;
    const int64_t rows = kv->metrics().flat_capacity;
    const auto cells = read_longs(fake.bound["cells"].data, 2);
    const int64_t read_len = read_longs(fake.bound["read_len"].data, 1)[0];
    EXPECT_EQ(cells[0], token) << token;
    EXPECT_EQ(cells[1], rows - 1) << token;
    EXPECT_GE(rows - 1, read_len) << token;
    EXPECT_LE(rows, kCells) << token;
  }
  EXPECT_GE(kv->metrics().growth_count, 2);
  // The table is full; the scratch row was never handed out.
  EXPECT_FALSE(control->declare_step({a}));
  kv->forget_handle(&handle);
}

TEST_F(CudaCellCacheTest, GrowthGivesAHandleAboutToCaptureOneEagerRunFirst) {
  // Growth rebinds every handle, and a rebound handle's next run refolds
  // AOTI's constants, which cannot happen inside a stream capture. A handle
  // whose warmup is about to capture on that very run must run eagerly once
  // more; one with warmup left keeps it.
  auto cache_ptr = make(/*initial=*/4);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto due_fake = container();
  auto early_fake = container();
  auto due = make_handle(due_fake);
  auto early = make_handle(early_fake);
  ASSERT_TRUE(kv->note_handle(&due).get());
  ASSERT_TRUE(kv->note_handle(&early).get());
  const int32_t a = *control->seq_new();
  ASSERT_EQ(step(*cache_ptr, due, {a}), Error::Ok);
  due.cuda_graph_state.phase = cu::CudaGraphPhase::Warmup;
  due.cuda_graph_state.warmup_remaining = 0;
  early.cuda_graph_state.phase = cu::CudaGraphPhase::Warmup;
  early.cuda_graph_state.warmup_remaining = 3;

  // Past the four allocated rows: the pool grows.
  ASSERT_EQ(step(*cache_ptr, early, {a, a, a, a}), Error::Ok);
  ASSERT_EQ(kv->metrics().growth_count, 1);
  EXPECT_EQ(due.cuda_graph_state.phase, cu::CudaGraphPhase::Warmup);
  EXPECT_EQ(due.cuda_graph_state.warmup_remaining, 1);
  EXPECT_EQ(early.cuda_graph_state.warmup_remaining, 3);
  kv->forget_handle(&due);
  kv->forget_handle(&early);
}

TEST_F(CudaCellCacheTest, DFlashUploadsKVWideCellsAndQueryWideOwnerMasks) {
  auto cache_ptr = make(4, 2);
  ASSERT_NE(cache_ptr, nullptr);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new(), b = *control->seq_new();
  ASSERT_EQ(step(*cache_ptr, handle, {b, b, b, b, b, a, a, a}), Error::Ok);
  EXPECT_EQ(
      read_longs(fake.bound["cells"].data, 8),
      std::vector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(read_longs(fake.bound["read_len"].data, 1)[0], 8);
  using Mask = std::vector<std::vector<int>>;
  EXPECT_EQ(
      read_mask(fake.bound["mask0"].data, 4, 8),
      Mask(
          {{1, 1, 1, 1, 1, 0, 0, 0},
           {1, 1, 1, 1, 1, 0, 0, 0},
           {0, 0, 0, 0, 0, 1, 1, 1},
           {0, 0, 0, 0, 0, 1, 1, 1}}));
  EXPECT_EQ(
      read_mask(fake.bound["mask2"].data, 4, 8),
      Mask(
          {{0, 1, 1, 1, 1, 0, 0, 0},
           {0, 1, 1, 1, 1, 0, 0, 0},
           {0, 0, 0, 0, 0, 1, 1, 1},
           {0, 0, 0, 0, 0, 1, 1, 1}}));
  EXPECT_EQ(control->pos(b), 3);
  EXPECT_EQ(control->pos(a), 1);
  // Block removal leaves holes; the next owner order is reversed.
  handle.cuda_graph_state.phase = cu::CudaGraphPhase::Replay;
  void* cells = fake.bound["cells"].data;
  ASSERT_EQ(step(*cache_ptr, handle, {a, a, b, b}), Error::Ok);
  EXPECT_EQ(
      read_longs(fake.bound["cells"].data, 4),
      std::vector<int64_t>({3, 4, 6, 7}));
  EXPECT_EQ(
      read_mask(fake.bound["mask0"].data, 4, 8),
      Mask(
          {{0, 0, 0, 1, 1, 1, 0, 0},
           {0, 0, 0, 1, 1, 1, 0, 0},
           {1, 1, 1, 0, 0, 0, 1, 1},
           {1, 1, 1, 0, 0, 0, 1, 1}}));
  EXPECT_EQ(handle.cuda_graph_state.phase, cu::CudaGraphPhase::Replay);
  EXPECT_EQ(fake.bound["cells"].data, cells);
  EXPECT_EQ(control->pos(b), 3);
  EXPECT_EQ(control->pos(a), 1);
  kv->forget_handle(&handle);
}

TEST_F(
    CudaCellCacheTest,
    DFlashRejectsGroupingAndNarrowProgramsBeforeAllocation) {
  auto cache_ptr = make(4, 2);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new(), b = *control->seq_new();
  EXPECT_FALSE(control->declare_step({a, a, b, b, a, a}));
  EXPECT_FALSE(control->declare_step({a, a, b}));
  ASSERT_TRUE(control->declare_step({b, b, a, a}));
  EXPECT_EQ(kv->prepare_step(3, cudaStreamPerThread), Error::InvalidState);
  EXPECT_EQ(kv->metrics().allocated_bytes, 0);
  EXPECT_EQ(control->pos(a), 0);
  EXPECT_EQ(control->pos(b), 0);
  ASSERT_EQ(kv->prepare_step(4, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv->commit_step(4), Error::Ok);
  EXPECT_EQ(control->pos(a), 0);
  EXPECT_EQ(control->pos(b), 0);
  kv->forget_handle(&handle);
}

TEST_F(
    CudaCellCacheTest,
    DFlashAbandonedCoBatchCannotContinueUntilWholeCacheCleared) {
  auto cache_ptr = make(4, 2);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new(), b = *control->seq_new();
  ASSERT_TRUE(control->declare_step({a, a, a, b, b, b}));
  ASSERT_EQ(kv->prepare_step(6, cudaStreamPerThread), Error::Ok);
  // A failed forward does not commit: neither owner may continue or fork.
  EXPECT_FALSE(control->declare_step({a, a}));
  EXPECT_FALSE(control->rewind(a, 0));
  EXPECT_FALSE(control->seq_clone(b, std::nullopt));
  EXPECT_EQ(kv->prepare_step(6, cudaStreamPerThread), Error::InvalidState);
  EXPECT_EQ(kv->commit_step(5), Error::InvalidState);
  EXPECT_EQ(kv->commit_step(6), Error::InvalidState);
  control->clear();
  EXPECT_EQ(kv->metrics().logical_length, 0);
  const int32_t reused = *control->seq_new();
  ASSERT_EQ(step(*cache_ptr, handle, {reused, reused, reused}), Error::Ok);
  EXPECT_EQ(control->pos(reused), 1);
  kv->forget_handle(&handle);
}

TEST_F(CudaCellCacheTest, DFlashStaticProgramPadsTailIntoScratch) {
  auto cache_ptr = make(4, 2);
  ASSERT_NE(cache_ptr, nullptr);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new(), b = *control->seq_new();
  // One context row for a, none for b, padded to an eight-row program.
  ASSERT_TRUE(control->declare_step({a, a, a, b, b}));
  ASSERT_EQ(kv->prepare_step(8, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  const auto cells = read_longs(fake.bound["cells"].data, 8);
  EXPECT_EQ(
      std::vector<int64_t>(cells.begin(), cells.begin() + 5),
      std::vector<int64_t>({0, 1, 2, 3, 4}));
  for (int row = 5; row < 8; ++row) {
    EXPECT_EQ(cells[row], cells[7]) << row;
    EXPECT_GE(cells[row], 5) << row;
  }
  using Mask = std::vector<std::vector<int>>;
  EXPECT_EQ(
      read_mask(fake.bound["mask0"].data, 4, 5),
      Mask(
          {{1, 1, 1, 0, 0},
           {1, 1, 1, 0, 0},
           {0, 0, 0, 1, 1},
           {0, 0, 0, 1, 1}}));
  ASSERT_EQ(kv->commit_step(8), Error::Ok);
  EXPECT_EQ(control->pos(a), 1);
  EXPECT_EQ(control->pos(b), 0);
  kv->forget_handle(&handle);
}

TEST_F(CudaCellCacheTest, DFlashWarmupCommitsAllContextWithOneQueryPerOwner) {
  auto cache_ptr = make(4, 0);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new(), b = *control->seq_new();
  ASSERT_EQ(step(*cache_ptr, handle, {b, b, b, a}), Error::Ok);
  EXPECT_EQ(
      read_mask(fake.bound["mask2"].data, 2, 4),
      std::vector<std::vector<int>>({{0, 1, 1, 0}, {0, 0, 0, 1}}));
  EXPECT_EQ(control->pos(b), 3);
  EXPECT_EQ(control->pos(a), 1);
  kv->forget_handle(&handle);
}

TEST_F(
    CudaCellCacheTest,
    DFlashCommitFencesGPUWorkBeforeDroppingTemporaryCells) {
  auto cache_ptr = make(4, 2);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();
  ASSERT_TRUE(control->declare_step({a, a, a}));
  ASSERT_EQ(kv->prepare_step(3, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  std::atomic<bool> finished{false};
  ASSERT_EQ(
      cudaLaunchHostFunc(
          cudaStreamPerThread,
          [](void* state) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            static_cast<std::atomic<bool>*>(state)->store(true);
          },
          &finished),
      cudaSuccess);
  ASSERT_EQ(kv->commit_step(3), Error::Ok);
  EXPECT_TRUE(finished.load());
  EXPECT_EQ(control->pos(a), 1);
  kv->forget_handle(&handle);
}

TEST_F(CudaCellCacheTest, DFlashClearFromAnotherThreadFencesAbandonedForward) {
  auto cache_ptr = make(4, 2);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();
  ASSERT_TRUE(control->declare_step({a, a, a}));
  ASSERT_EQ(kv->prepare_step(3, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  std::atomic<bool> finished{false};
  ASSERT_EQ(
      cudaLaunchHostFunc(
          cudaStreamPerThread,
          [](void* state) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            static_cast<std::atomic<bool>*>(state)->store(true);
          },
          &finished),
      cudaSuccess);
  bool finished_when_cleared = false;
  std::thread cleanup([&] {
    control->clear();
    finished_when_cleared = finished.load();
  });
  cleanup.join();
  EXPECT_TRUE(finished_when_cleared);
  EXPECT_EQ(kv->metrics().logical_length, 0);
  const int32_t reused = *control->seq_new();
  ASSERT_EQ(step(*cache_ptr, handle, {reused, reused, reused}), Error::Ok);
  EXPECT_EQ(control->pos(reused), 1);
  kv->forget_handle(&handle);
}

TEST_F(CudaCellCacheTest, DFlashBuilderRejectsUnsupportedSequenceLayout) {
  auto geometry = flat_and_ring(kWindow);
  for (const int block : {0, 2}) {
    geometry.dflash_block_size = block;
    EXPECT_EQ(make_cache(geometry, config(kCells, 4, kMaxWrite)), nullptr);
  }
  geometry.dflash_block_size = -1;
  EXPECT_EQ(
      cu::make_cuda_cell_kv_cache(geometry, config(kCells, 4, kMaxWrite)),
      nullptr);
  geometry.dflash_block_size = kMaxWrite + 1;
  EXPECT_EQ(
      cu::make_cuda_cell_kv_cache(geometry, config(kCells, 4, kMaxWrite)),
      nullptr);
}

TEST_F(CudaCellCacheTest, BuilderValidatesAndIsRegisteredForBatchedKinds) {
  auto no_max_write = config(kCells, 4, kMaxWrite);
  no_max_write.max_write.reset();
  EXPECT_EQ(
      cu::make_cuda_cell_kv_cache(flat_and_ring(kWindow), no_max_write),
      nullptr);
  EXPECT_EQ(
      cu::make_cuda_cell_kv_cache(
          flat_and_ring(kWindow), config(kCells, 4, kCells + 1)),
      nullptr);
  // One cell would all go to the padding scratch row.
  EXPECT_EQ(
      cu::make_cuda_cell_kv_cache(flat_and_ring(kWindow), config(1, 1, 1)),
      nullptr);
  for (const char* kind : {cache::kind::kBatchedCell, cache::kind::kBatched}) {
    auto built = cache::CacheFactory::global().build(
        cu::kCudaBackendId,
        kind,
        flat_and_ring(kWindow),
        config(kCells, 4, kMaxWrite));
    ASSERT_TRUE(built.ok()) << kind;
    EXPECT_NE(built.get()->as<cache::BatchControl>(), nullptr) << kind;
    EXPECT_NE(built.get()->as<cu::CudaKVCache>(), nullptr) << kind;
  }
}

namespace {

class CudaBatchedSequenceCacheTest : public CudaKVCacheTest {
 protected:
  static constexpr int kCapacity = 16;
  static constexpr int kMaxWrite = 8;
  static constexpr int kMaxSeqs = 3;
  static constexpr int kColumns = kMaxSeqs + 1;
  static constexpr int kWindow = 2;
  // min(capacity, window + max_write - 1)
  static constexpr int kRingRows = kWindow + kMaxWrite - 1;
  static constexpr int kSegments = 2 * kMaxSeqs;

  static cache::CacheConfig sequence_config(int initial = 4) {
    auto cfg = config(kCapacity, initial, kMaxWrite);
    cfg.max_seqs = kMaxSeqs;
    return cfg;
  }

  static std::shared_ptr<cache::Cache> make(
      int initial = 4,
      std::optional<int> block_size = std::nullopt) {
    auto geometry = flat_and_ring(kWindow);
    geometry.dflash_block_size = block_size;
    return cu::make_cuda_batched_sequence_kv_cache(
        geometry, sequence_config(initial));
  }

  // What a program lowered in the batched-sequence layout declares for make().
  static FakeContainer container() {
    FakeContainer container{
        {"f_k", "f_v", "r_k", "r_v", "seq_cols", "seq_info"},
        {"__et_offgraph_kv_layer_0_k",
         "__et_offgraph_kv_layer_0_v",
         "__et_offgraph_kv_layer_1_k",
         "__et_offgraph_kv_layer_1_v",
         "__et_offgraph_kv_seq_cols",
         "__et_offgraph_kv_seq_info"},
        {},
        0};
    for (size_t index = 0; index < 4; ++index) {
      declare(
          container,
          container.fqns[index],
          slimc10::ScalarType::BFloat16,
          {1, (index < 2 ? kCapacity : kRingRows) * kColumns, kHeads, kDim});
    }
    declare(
        container,
        "__et_offgraph_kv_seq_cols",
        slimc10::ScalarType::Long,
        {kMaxWrite});
    declare(
        container,
        "__et_offgraph_kv_seq_info",
        slimc10::ScalarType::Int,
        {kSegments, 4});
    return container;
  }

  static std::vector<int64_t> read_cols(void* device, int count) {
    std::vector<int64_t> values(count);
    EXPECT_EQ(
        cudaMemcpy(
            values.data(),
            device,
            count * sizeof(int64_t),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    return values;
  }

  using Info = std::vector<std::array<int32_t, 4>>;
  static Info read_info(void* device, int rows) {
    std::vector<int32_t> flat(kSegments * 4);
    EXPECT_EQ(
        cudaMemcpy(
            flat.data(),
            device,
            flat.size() * sizeof(int32_t),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    Info out(rows);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < 4; ++j) {
        out[i][j] = flat[i * 4 + j];
      }
    }
    return out;
  }

  static Error step(
      cache::Cache& cache,
      cu::CudaDelegateHandle& handle,
      const std::vector<int32_t>& seq_ids,
      int64_t width = 0) {
    auto* control = cache.as<cache::BatchControl>();
    auto* kv = cache.as<cu::CudaKVCache>();
    if (!control->declare_step(seq_ids)) {
      return Error::InvalidArgument;
    }
    if (width == 0) {
      width = static_cast<int64_t>(seq_ids.size());
    }
    ET_CHECK_OK_OR_RETURN_ERROR(kv->prepare_step(width, cudaStreamPerThread));
    ET_CHECK_OK_OR_RETURN_ERROR(kv->rebind_for_execute(&handle));
    return kv->commit_step(width);
  }
};

} // namespace

TEST_F(CudaBatchedSequenceCacheTest, EachSequenceWritesItsOwnColumn) {
  auto cache_ptr = make();
  ASSERT_NE(cache_ptr, nullptr);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  ASSERT_NE(control, nullptr);
  ASSERT_NE(kv, nullptr);
  EXPECT_EQ(control->max_seqs(), kMaxSeqs);
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();
  const int32_t b = *control->seq_new();

  // Prefilled together in a program two wider: padding names the scratch.
  ASSERT_EQ(step(*cache_ptr, handle, {a, a, a, b, b}, 7), Error::Ok);
  EXPECT_EQ(
      read_cols(fake.bound["seq_cols"].data, 7),
      std::vector<int64_t>({0, 0, 0, 1, 1, kMaxSeqs, kMaxSeqs}));
  EXPECT_EQ(
      read_info(fake.bound["seq_info"].data, 3),
      Info({{0, 3, 0, 3}, {3, 2, 1, 2}, {0, 0, 0, 0}}));
  EXPECT_EQ(
      fake.bound["f_k"].sizes,
      std::vector<int64_t>({1, kCapacity * kColumns, kHeads, kDim}));
  EXPECT_EQ(fake.bound["seq_info"].dtype, slimc10::ScalarType::Int);
  EXPECT_EQ(kv->metrics().flat_capacity, 4);

  // Decoding in the other order keeps each one's column and own length.
  ASSERT_EQ(step(*cache_ptr, handle, {b, a}), Error::Ok);
  EXPECT_EQ(
      read_cols(fake.bound["seq_cols"].data, 2), std::vector<int64_t>({1, 0}));
  EXPECT_EQ(
      read_info(fake.bound["seq_info"].data, 2),
      Info({{0, 1, 1, 3}, {1, 1, 0, 4}}));
  EXPECT_EQ(control->pos(a), 4);
  EXPECT_EQ(control->pos(b), 3);
  EXPECT_EQ(kv->metrics().logical_length, 4);

  // The longest sequence decides the rows; growth doubles them.
  ASSERT_EQ(step(*cache_ptr, handle, {a}), Error::Ok);
  const auto metrics = kv->metrics();
  EXPECT_EQ(metrics.flat_capacity, 8);
  EXPECT_EQ(metrics.growth_count, 1);
  kv->forget_handle(&handle);
}

TEST_F(CudaBatchedSequenceCacheTest, AdmissionIsPerSequenceAndBounded) {
  auto cache_ptr = make(/*initial=*/kCapacity);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  std::vector<int32_t> ids;
  for (int i = 0; i < kMaxSeqs; ++i) {
    ids.push_back(*control->seq_new());
  }
  EXPECT_FALSE(control->seq_new().has_value());
  // Unknown ids, and steps past max_write, are refused.
  EXPECT_FALSE(control->declare_step({ids[0], 99}));
  EXPECT_FALSE(
      control->declare_step(std::vector<int32_t>(kMaxWrite + 1, ids[0])));
  // A sequence fills its own capacity without touching another's room.
  for (int i = 0; i < kCapacity / kMaxWrite; ++i) {
    ASSERT_EQ(
        step(*cache_ptr, handle, std::vector<int32_t>(kMaxWrite, ids[0])),
        Error::Ok);
  }
  EXPECT_EQ(control->pos(ids[0]), kCapacity);
  EXPECT_FALSE(control->declare_step({ids[0]}));
  EXPECT_TRUE(control->declare_step({ids[1]}));
  ASSERT_TRUE(control->seq_rm(ids[0]));
  EXPECT_EQ(control->seq_new(), ids[0]);
  EXPECT_EQ(control->pos(ids[0]), 0);
  kv->forget_handle(&handle);
}

TEST_F(CudaBatchedSequenceCacheTest, CloneCopiesThePrefixIntoAFreeColumn) {
  auto cache_ptr = make(/*initial=*/8);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();
  ASSERT_EQ(step(*cache_ptr, handle, {a, a, a, a}), Error::Ok);

  // Fill a's column with a recognizable pattern, then clone its first 3 rows.
  const size_t slot = kHeads * kDim * sizeof(uint16_t);
  const size_t pitch = slot * kColumns;
  std::vector<uint8_t> rows(pitch * 8, 0);
  for (int r = 0; r < 4; ++r) {
    std::fill_n(rows.begin() + r * pitch, slot, static_cast<uint8_t>(r + 1));
  }
  ASSERT_EQ(
      cudaMemcpy(
          fake.bound["f_k"].data,
          rows.data(),
          rows.size(),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  const auto c = control->seq_clone(a, 3);
  ASSERT_TRUE(c.has_value());
  EXPECT_NE(*c, a);
  EXPECT_EQ(control->pos(*c), 3);
  EXPECT_EQ(control->pos(a), 4);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint8_t> after(rows.size());
  ASSERT_EQ(
      cudaMemcpy(
          after.data(),
          fake.bound["f_k"].data,
          after.size(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (int r = 0; r < 4; ++r) {
    EXPECT_EQ(after[r * pitch + *c * slot], r < 3 ? r + 1 : 0) << r;
  }
  // The clone continues at its own length.
  ASSERT_EQ(step(*cache_ptr, handle, {*c}), Error::Ok);
  EXPECT_EQ(read_info(fake.bound["seq_info"].data, 1), Info({{0, 1, *c, 4}}));
  EXPECT_FALSE(control->seq_clone(99, std::nullopt).has_value());
  kv->forget_handle(&handle);
}

TEST_F(CudaBatchedSequenceCacheTest, ClearRecoversAbandonedDFlashStep) {
  auto cache_ptr = make(kCapacity, 2);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const auto a = *control->seq_new();
  ASSERT_TRUE(control->declare_step({a, a, a}));
  ASSERT_EQ(kv->prepare_step(3, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(control->pos(a), 3);
  control->clear();
  const auto b = *control->seq_new();
  ASSERT_TRUE(control->declare_step({b, b, b}));
  EXPECT_EQ(kv->prepare_step(3, cudaStreamPerThread), Error::Ok);
  kv->forget_handle(&handle);
}

TEST_F(CudaBatchedSequenceCacheTest, PendingDFlashOwnerCanClose) {
  auto cache_ptr = make(kCapacity, 2);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const auto id = control->seq_new();
  ASSERT_TRUE(id.has_value());
  ASSERT_TRUE(control->declare_step({*id, *id, *id}));
  ASSERT_EQ(kv->prepare_step(3, cudaStreamPerThread), Error::Ok);
  ASSERT_TRUE(control->seq_rm(*id));
  EXPECT_EQ(kv->commit_step(3), Error::Ok);
  EXPECT_EQ(control->seq_new(), id);
  EXPECT_EQ(control->pos(*id), 0);
  ASSERT_EQ(step(*cache_ptr, handle, {*id, *id, *id}), Error::Ok);
  kv->forget_handle(&handle);
}

TEST_F(CudaBatchedSequenceCacheTest, CloneOnAnotherStreamWaitsForTheCopy) {
  auto cache_ptr = make(kCapacity);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  cudaStream_t first = nullptr;
  cudaStream_t second = nullptr;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking), cudaSuccess);
  const auto a = *control->seq_new();
  ASSERT_TRUE(control->declare_step({a, a, a, a}));
  ASSERT_EQ(kv->prepare_step(4, first), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  ASSERT_EQ(kv->commit_step(4), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(first), cudaSuccess);
  const size_t slot = kHeads * kDim * sizeof(uint16_t);
  const size_t pitch = slot * kColumns;
  std::vector<uint8_t> before(pitch * kCapacity, 0);
  for (int row = 0; row < 4; ++row) {
    std::fill_n(before.begin() + row * pitch + a * slot, slot, 0x3c);
  }
  ASSERT_EQ(
      cudaMemcpy(
          fake.bound["f_k"].data,
          before.data(),
          before.size(),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  uint8_t* after = nullptr;
  ASSERT_EQ(cudaMallocHost(&after, before.size()), cudaSuccess);
  ASSERT_EQ(
      cudaLaunchHostFunc(
          first,
          [](void*) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
          },
          nullptr),
      cudaSuccess);
  const auto b = control->seq_clone(a, 4);
  ASSERT_TRUE(b.has_value());
  ASSERT_TRUE(control->declare_step({*b}));
  ASSERT_EQ(kv->prepare_step(1, second), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  ASSERT_EQ(
      cudaMemcpyAsync(
          after,
          fake.bound["f_k"].data,
          before.size(),
          cudaMemcpyDeviceToHost,
          second),
      cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(second), cudaSuccess);
  for (int row = 0; row < 4; ++row) {
    EXPECT_EQ(after[row * pitch + *b * slot], 0x3c) << row;
  }
  EXPECT_EQ(cudaStreamSynchronize(first), cudaSuccess);
  EXPECT_EQ(cudaFreeHost(after), cudaSuccess);
  kv->forget_handle(&handle);
  cache_ptr.reset();
  EXPECT_EQ(cudaStreamDestroy(second), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(first), cudaSuccess);
}

TEST_F(
    CudaBatchedSequenceCacheTest,
    CloneAcrossPerThreadStreamsWaitsForWrites) {
  auto cache_ptr = make(kCapacity);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const auto a = *control->seq_new();
  ASSERT_TRUE(control->declare_step({a, a, a, a}));
  ASSERT_EQ(kv->prepare_step(4, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  ASSERT_EQ(kv->commit_step(4), Error::Ok);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const size_t slot = kHeads * kDim * sizeof(uint16_t);
  const size_t pitch = slot * kColumns;
  auto* data = fake.bound["f_k"].data;
  ASSERT_EQ(cudaMemset(data, 0, pitch * kCapacity), cudaSuccess);
  uint8_t* after = nullptr;
  cudaStream_t reader = nullptr;
  ASSERT_EQ(cudaMallocHost(&after, pitch * kCapacity), cudaSuccess);
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&reader, cudaStreamNonBlocking), cudaSuccess);
  // The writer runs a second step wholly on its own per-thread stream, as a
  // delegate thread would, so the step-done event lands behind its delayed
  // writes. The clone below runs on another thread's stream and must wait for
  // them: without the fence it reads zeros, the writes still being behind
  // the sleep.
  std::atomic<bool> ready{false}, release{false};
  std::thread writer([&]() {
    EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_TRUE(control->declare_step({a, a, a, a}));
    EXPECT_EQ(kv->prepare_step(4, cudaStreamPerThread), Error::Ok);
    EXPECT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
    EXPECT_EQ(
        cudaLaunchHostFunc(
            cudaStreamPerThread,
            [](void*) {
              std::this_thread::sleep_for(std::chrono::milliseconds(200));
            },
            nullptr),
        cudaSuccess);
    EXPECT_EQ(
        cudaMemset2DAsync(
            static_cast<uint8_t*>(data) + a * slot,
            pitch,
            0x3c,
            slot,
            4,
            cudaStreamPerThread),
        cudaSuccess);
    EXPECT_EQ(kv->commit_step(4), Error::Ok);
    ready.store(true);
    while (!release.load())
      std::this_thread::yield();
    EXPECT_EQ(cudaStreamSynchronize(cudaStreamPerThread), cudaSuccess);
  });
  while (!ready.load())
    std::this_thread::yield();
  const auto b = control->seq_clone(a, 4);
  EXPECT_TRUE(b.has_value());
  if (b) {
    EXPECT_TRUE(control->declare_step({*b}));
    EXPECT_EQ(kv->prepare_step(1, reader), Error::Ok);
    EXPECT_EQ(
        cudaMemcpyAsync(
            after, data, pitch * kCapacity, cudaMemcpyDeviceToHost, reader),
        cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(reader), cudaSuccess);
    for (int row = 0; row < 4; ++row) {
      EXPECT_EQ(after[row * pitch + *b * slot], 0x3c) << row;
    }
  }
  release.store(true);
  writer.join();
  EXPECT_EQ(cudaFreeHost(after), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(reader), cudaSuccess);
  kv->forget_handle(&handle);
}

TEST_F(CudaBatchedSequenceCacheTest, CloneOfAnUnwrappedRingCopiesThePrefix) {
  auto cache_ptr = make(/*initial=*/4);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();
  ASSERT_EQ(step(*cache_ptr, handle, {a, a, a, a}), Error::Ok);

  // The ring holds kRingRows rows but the pool only 4 per column: the clone
  // copies the 4-row prefix, not the whole ring.
  const size_t slot = kHeads * kDim * sizeof(uint16_t);
  const size_t pitch = slot * kColumns;
  std::vector<uint8_t> rows(pitch * 4, 0);
  for (int r = 0; r < 4; ++r) {
    std::fill_n(rows.begin() + r * pitch, slot, static_cast<uint8_t>(r + 1));
  }
  ASSERT_EQ(
      cudaMemcpy(
          fake.bound["r_k"].data,
          rows.data(),
          rows.size(),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  const auto c = control->seq_clone(a, std::nullopt);
  ASSERT_TRUE(c.has_value());
  EXPECT_EQ(control->pos(*c), 4);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint8_t> after(rows.size());
  ASSERT_EQ(
      cudaMemcpy(
          after.data(),
          fake.bound["r_k"].data,
          after.size(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (int r = 0; r < 4; ++r) {
    EXPECT_EQ(after[r * pitch + *c * slot], r + 1) << r;
  }
  kv->forget_handle(&handle);
}

TEST_F(CudaBatchedSequenceCacheTest, RingLayersGrowUntilTheyHoldTheirSlots) {
  auto cache_ptr = make(/*initial=*/2);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  constexpr int64_t kRowBytes =
      kHeads * kDim * static_cast<int64_t>(sizeof(uint16_t)) * kColumns;
  const int32_t a = *control->seq_new();

  // Two rows per column for both layers, not the ring's kRingRows up front.
  ASSERT_EQ(step(*cache_ptr, handle, {a, a}), Error::Ok);
  EXPECT_EQ(kv->metrics().allocated_bytes, 2 * (2 + 2) * kRowBytes);

  // Past kRingRows the ring stops growing while the flat layer keeps going.
  ASSERT_EQ(
      step(*cache_ptr, handle, std::vector<int32_t>(kMaxWrite, a)), Error::Ok);
  constexpr int kLength = 2 + kMaxWrite;
  static_assert(kLength > kRingRows);
  EXPECT_EQ(control->pos(a), kLength);
  EXPECT_EQ(kv->metrics().flat_capacity, kLength);
  EXPECT_EQ(
      kv->metrics().allocated_bytes, 2 * (kLength + kRingRows) * kRowBytes);

  // A wrapped ring is cloned whole.
  const size_t slot = kHeads * kDim * sizeof(uint16_t);
  std::vector<uint8_t> ring(slot * kColumns * kRingRows, 0);
  for (int r = 0; r < kRingRows; ++r) {
    std::fill_n(
        ring.begin() + (r * kColumns + a) * slot,
        slot,
        static_cast<uint8_t>(r + 1));
  }
  ASSERT_EQ(
      cudaMemcpy(
          fake.bound["r_k"].data,
          ring.data(),
          ring.size(),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  const auto c = control->seq_clone(a, std::nullopt);
  ASSERT_TRUE(c.has_value());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint8_t> after(ring.size());
  ASSERT_EQ(
      cudaMemcpy(
          after.data(),
          fake.bound["r_k"].data,
          after.size(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  for (int r = 0; r < kRingRows; ++r) {
    EXPECT_EQ(after[(r * kColumns + *c) * slot], r + 1) << r;
  }
  kv->forget_handle(&handle);
}

TEST_F(CudaBatchedSequenceCacheTest, RewindRespectsTheRingFloor) {
  auto cache_ptr = make(/*initial=*/kCapacity);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();
  ASSERT_EQ(
      step(*cache_ptr, handle, std::vector<int32_t>(kMaxWrite, a)), Error::Ok);
  ASSERT_EQ(
      step(*cache_ptr, handle, std::vector<int32_t>(kMaxWrite, a)), Error::Ok);
  // A ring of kRingRows rows retains back to 16 - window = 14 only.
  EXPECT_FALSE(control->rewind(a, 2));
  EXPECT_TRUE(control->rewind(a, 15));
  EXPECT_EQ(control->pos(a), 15);
  EXPECT_FALSE(control->rewind(a, 16));
  kv->forget_handle(&handle);
}

TEST_F(CudaBatchedSequenceCacheTest, DFlashQueriesOnlyEachOwnersBlock) {
  constexpr int kBlock = 2;
  auto cache_ptr = make(/*initial=*/kCapacity, kBlock);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const int32_t a = *control->seq_new();
  const int32_t b = *control->seq_new();
  // a: 3 new context + block; b: 1 new context + block; padded to 8.
  ASSERT_EQ(step(*cache_ptr, handle, {a, a, a, a, a, b, b, b}), Error::Ok);
  EXPECT_EQ(
      read_cols(fake.bound["seq_cols"].data, 8),
      std::vector<int64_t>({0, 0, 0, 0, 0, 1, 1, 1}));
  // Queries pack block by block; each sees its context and its block.
  EXPECT_EQ(
      read_info(fake.bound["seq_info"].data, 2),
      Info({{0, kBlock, 0, 5}, {kBlock, kBlock, 1, 3}}));
  // The blocks are dropped after the forward.
  EXPECT_EQ(control->pos(a), 3);
  EXPECT_EQ(control->pos(b), 1);
  // An owner split into two runs, or one short of its block, is refused.
  EXPECT_FALSE(control->declare_step({a, a, b, b, a}));
  EXPECT_FALSE(control->declare_step({a, b, b}));
  kv->forget_handle(&handle);
}

namespace {

class CudaRuntimeColumnCacheTest : public CudaKVCacheTest {
 protected:
  static constexpr int kCapacity = 64;
  static constexpr int kMaxWrite = 32;
  static constexpr int kMaxSeqs = 21;
  static constexpr int kDeclaredColumns = kMaxSeqs + 1;
  static constexpr int kSegments = 2 * kMaxSeqs;

  static std::shared_ptr<cache::Cache> make() {
    auto cfg = config(kCapacity, 4, kMaxWrite);
    cfg.max_seqs = kMaxSeqs;
    return cu::make_cuda_batched_sequence_kv_cache(
        flat_and_ring(2), cfg, /*runtime_columns=*/true);
  }

  static FakeContainer container(int info_width = 5) {
    auto fake = flat_and_ring_container();
    fake.internal_names.insert(
        fake.internal_names.end(), {"seq_cols", "seq_info"});
    fake.fqns.insert(
        fake.fqns.end(),
        {"__et_offgraph_kv_seq_cols", "__et_offgraph_kv_seq_info"});
    for (size_t index = 0; index < 4; ++index) {
      declare(
          fake,
          fake.fqns[index],
          slimc10::ScalarType::BFloat16,
          {1,
           (index < 2 ? kCapacity : kMaxWrite + 1) * kDeclaredColumns,
           kHeads,
           kDim});
    }
    declare(fake, fake.fqns[4], slimc10::ScalarType::Long, {kMaxWrite});
    declare(
        fake, fake.fqns[5], slimc10::ScalarType::Int, {kSegments, info_width});
    return fake;
  }

  static Error step(
      cache::Cache& cache,
      cu::CudaDelegateHandle& handle,
      const std::vector<int32_t>& ids,
      int width = 0) {
    auto* control = cache.as<cache::BatchControl>();
    auto* kv = cache.as<cu::CudaKVCache>();
    if (!control->declare_step(ids)) {
      return Error::InvalidArgument;
    }
    width = width == 0 ? static_cast<int>(ids.size()) : width;
    ET_CHECK_OK_OR_RETURN_ERROR(kv->prepare_step(width, cudaStreamPerThread));
    ET_CHECK_OK_OR_RETURN_ERROR(kv->rebind_for_execute(&handle));
    return kv->commit_step(width);
  }

  static std::vector<uint16_t>
  read_column(void* data, int rows, int columns, int column) {
    std::vector<uint16_t> out(rows * kRow);
    EXPECT_EQ(
        cudaMemcpy2D(
            out.data(),
            kRow * sizeof(uint16_t),
            static_cast<uint16_t*>(data) + column * kRow,
            columns * kRow * sizeof(uint16_t),
            kRow * sizeof(uint16_t),
            rows,
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    return out;
  }

  static void seed(FakeContainer& fake, int rows, int columns) {
    for (size_t layer = 0; layer < 4; ++layer) {
      std::vector<uint16_t> values(rows * columns * kRow);
      for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
          for (int element = 0; element < kRow; ++element) {
            values[(row * columns + column) * kRow + element] =
                static_cast<uint16_t>(
                    1000 * layer + 100 * row + 20 * column + element);
          }
        }
      }
      EXPECT_EQ(
          cudaMemcpy(
              fake.bound[fake.internal_names[layer]].data,
              values.data(),
              values.size() * sizeof(uint16_t),
              cudaMemcpyHostToDevice),
          cudaSuccess);
    }
  }
};

} // namespace

TEST_F(CudaRuntimeColumnCacheTest, GrowthAndSparseIdCompactionPreserveAllKV) {
  auto cache_ptr = make();
  ASSERT_NE(cache_ptr, nullptr);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  std::vector<int32_t> ids;
  int old_columns = 0;
  for (const int sessions : {1, 2, 4, 8, 16}) {
    const int live_rows = ids.empty() ? 0 : control->pos(ids.front());
    std::vector<std::array<std::vector<uint16_t>, 4>> retained(ids.size());
    if (old_columns != 0) {
      seed(fake, live_rows, old_columns);
      for (size_t owner = 0; owner < retained.size(); ++owner) {
        for (size_t layer = 0; layer < 4; ++layer) {
          retained[owner][layer] = read_column(
              fake.bound[fake.internal_names[layer]].data,
              live_rows,
              old_columns,
              owner);
        }
      }
    }
    while (ids.size() < static_cast<size_t>(sessions)) {
      const auto id = control->seq_new();
      ASSERT_TRUE(id.has_value());
      ids.push_back(*id);
    }
    ASSERT_EQ(step(*cache_ptr, handle, ids), Error::Ok);
    const auto metrics = kv->metrics();
    EXPECT_EQ(metrics.physical_columns, sessions + 1);
    EXPECT_EQ(metrics.declared_columns, kDeclaredColumns);
    EXPECT_EQ(metrics.live_sequences, sessions);
    EXPECT_EQ(control->max_seqs(), kMaxSeqs);
    EXPECT_EQ(
        fake.bound["flat_k"].sizes,
        std::vector<int64_t>({1, kCapacity * kDeclaredColumns, kHeads, kDim}));
    if (old_columns != 0) {
      for (size_t owner = 0; owner < retained.size(); ++owner) {
        for (size_t layer = 0; layer < 4; ++layer) {
          EXPECT_EQ(
              read_column(
                  fake.bound[fake.internal_names[layer]].data,
                  live_rows,
                  sessions + 1,
                  owner),
              retained[owner][layer]);
        }
      }
    }
    old_columns = sessions + 1;
  }

  // Preserve a nonzero logical ID while compacting it into physical column 0.
  const int32_t survivor = ids[5];
  const int rows = control->pos(survivor);
  seed(fake, rows, old_columns);
  std::array<std::vector<uint16_t>, 4> retained;
  for (size_t layer = 0; layer < 4; ++layer) {
    retained[layer] = read_column(
        fake.bound[fake.internal_names[layer]].data, rows, old_columns, 5);
  }
  const int position = control->pos(survivor);
  const auto before = kv->metrics();
  for (const int32_t id : ids) {
    if (id != survivor) {
      ASSERT_TRUE(control->seq_rm(id));
    }
  }
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  const auto after = kv->metrics();
  EXPECT_EQ(after.physical_columns, 2);
  EXPECT_EQ(after.live_sequences, 1);
  EXPECT_EQ(after.flat_capacity, 4);
  EXPECT_EQ(control->pos(survivor), position);
  EXPECT_LT(after.allocated_bytes, before.allocated_bytes);
  EXPECT_LT(after.allocated_bytes * 17, before.allocated_bytes * 2);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  for (size_t layer = 0; layer < 4; ++layer) {
    EXPECT_EQ(
        read_column(fake.bound[fake.internal_names[layer]].data, rows, 2, 0),
        retained[layer]);
  }
  ASSERT_EQ(step(*cache_ptr, handle, {survivor}, 4), Error::Ok);
  std::array<int64_t, 4> cols{};
  ASSERT_EQ(
      cudaMemcpy(
          cols.data(),
          fake.bound["seq_cols"].data,
          sizeof(cols),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(cols, (std::array<int64_t, 4>{0, 1, 1, 1}));
  std::vector<int32_t> info(kSegments * 5);
  ASSERT_EQ(
      cudaMemcpy(
          info.data(),
          fake.bound["seq_info"].data,
          info.size() * sizeof(int32_t),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(info[2], 0);
  EXPECT_EQ(info[3], position + 1);
  for (int row = 0; row < kSegments; ++row) {
    EXPECT_EQ(info[row * 5 + 4], 2);
  }
  kv->forget_handle(&handle);
}

TEST_F(
    CudaRuntimeColumnCacheTest,
    ContractionDropsClosedOwnersUnusedRowCapacity) {
  auto cache_ptr = make();
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const auto long_owner = control->seq_new();
  const auto short_owner = control->seq_new();
  ASSERT_TRUE(long_owner.has_value());
  ASSERT_TRUE(short_owner.has_value());
  for (int chunk = 0; chunk < 2; ++chunk) {
    ASSERT_EQ(
        step(*cache_ptr, handle, std::vector<int32_t>(32, *long_owner)),
        Error::Ok);
  }
  ASSERT_EQ(
      step(*cache_ptr, handle, std::vector<int32_t>(3, *short_owner)),
      Error::Ok);
  seed(fake, 3, 3);
  std::array<std::vector<uint16_t>, 4> expected;
  for (size_t layer = 0; layer < expected.size(); ++layer) {
    expected[layer] =
        read_column(fake.bound[fake.internal_names[layer]].data, 3, 3, 1);
  }
  const auto before = kv->metrics();
  ASSERT_GE(before.flat_capacity, 64);
  ASSERT_TRUE(control->seq_rm(*long_owner));
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  const auto after = kv->metrics();
  EXPECT_EQ(after.physical_columns, 2);
  EXPECT_EQ(after.flat_capacity, 4);
  EXPECT_EQ(control->pos(*short_owner), 3);
  EXPECT_EQ(control->max_seqs(), kMaxSeqs);
  EXPECT_LT(after.allocated_bytes * 8, before.allocated_bytes);
  EXPECT_EQ(
      fake.bound["flat_k"].sizes,
      std::vector<int64_t>({1, kCapacity * kDeclaredColumns, kHeads, kDim}));
  for (size_t layer = 0; layer < expected.size(); ++layer) {
    EXPECT_EQ(
        read_column(fake.bound[fake.internal_names[layer]].data, 3, 2, 0),
        expected[layer]);
  }
  const auto growth = after.growth_count;
  ASSERT_EQ(step(*cache_ptr, handle, {*short_owner}), Error::Ok);
  EXPECT_EQ(kv->metrics().growth_count, growth);
  ASSERT_EQ(step(*cache_ptr, handle, {*short_owner}), Error::Ok);
  EXPECT_GT(kv->metrics().growth_count, growth);
  for (size_t layer = 0; layer < expected.size(); ++layer) {
    EXPECT_EQ(
        read_column(fake.bound[fake.internal_names[layer]].data, 3, 2, 0),
        expected[layer]);
  }
  kv->forget_handle(&handle);
}

TEST_F(CudaRuntimeColumnCacheTest, CloneAndContractionCopyWrappedRingHistory) {
  auto cache_ptr = make();
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const auto source = control->seq_new();
  ASSERT_TRUE(source.has_value());
  ASSERT_EQ(
      step(*cache_ptr, handle, std::vector<int32_t>(32, *source)), Error::Ok);
  ASSERT_EQ(
      step(*cache_ptr, handle, std::vector<int32_t>(8, *source)), Error::Ok);
  std::array<std::vector<uint16_t>, 4> expected;
  for (size_t layer = 0; layer < 4; ++layer) {
    const int rows = layer < 2 ? 40 : 33;
    std::vector<uint16_t> values(rows * 2 * kRow);
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < 2; ++column) {
        for (int element = 0; element < kRow; ++element) {
          const uint16_t value =
              1000 * layer + 100 * row + 20 * column + element;
          values[(row * 2 + column) * kRow + element] = value;
          if (column == 0) {
            expected[layer].push_back(value);
          }
        }
      }
    }
    ASSERT_EQ(
        cudaMemcpy(
            fake.bound[fake.internal_names[layer]].data,
            values.data(),
            values.size() * sizeof(uint16_t),
            cudaMemcpyHostToDevice),
        cudaSuccess);
  }
  const auto clone = control->seq_clone(*source, 40);
  ASSERT_TRUE(clone.has_value());
  EXPECT_EQ(kv->metrics().physical_columns, 3);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  for (size_t layer = 0; layer < 4; ++layer) {
    const int rows = layer < 2 ? 40 : 33;
    EXPECT_EQ(
        read_column(fake.bound[fake.internal_names[layer]].data, rows, 3, 0),
        read_column(fake.bound[fake.internal_names[layer]].data, rows, 3, 1));
  }
  ASSERT_TRUE(control->seq_rm(*source));
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  EXPECT_EQ(control->pos(*clone), 40);
  for (size_t layer = 0; layer < 4; ++layer) {
    const int rows = layer < 2 ? 40 : 33;
    EXPECT_EQ(
        read_column(fake.bound[fake.internal_names[layer]].data, rows, 2, 0),
        expected[layer]);
  }
  ASSERT_EQ(step(*cache_ptr, handle, {*clone}), Error::Ok);
  EXPECT_EQ(control->pos(*clone), 41);
  kv->forget_handle(&handle);
}

TEST_F(CudaRuntimeColumnCacheTest, DeclaredStepCanCompactOrClose) {
  auto cache_ptr = make();
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const auto id = control->seq_new();
  ASSERT_TRUE(id.has_value());
  ASSERT_TRUE(control->declare_step({*id}));
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  ASSERT_EQ(kv->prepare_step(1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv->commit_step(1), Error::Ok);
  ASSERT_TRUE(control->declare_step({*id}));
  ASSERT_TRUE(control->seq_rm(*id));
  EXPECT_EQ(kv->prepare_step(1, cudaStreamPerThread), Error::InvalidArgument);
  EXPECT_EQ(kv->metrics().live_sequences, 0);
  EXPECT_EQ(control->seq_new(), id);
  ASSERT_EQ(step(*cache_ptr, handle, {*id}), Error::Ok);
  kv->forget_handle(&handle);
}

TEST_F(CudaRuntimeColumnCacheTest, AbortFencesBeforeCloseAndReuse) {
  auto cache_ptr = make();
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const auto id = control->seq_new();
  ASSERT_TRUE(id.has_value());
  ASSERT_TRUE(control->declare_step({*id}));
  ASSERT_EQ(kv->prepare_step(1, cudaStreamPerThread), Error::Ok);
  EXPECT_FALSE(control->seq_clone(*id, std::nullopt).has_value());
  EXPECT_FALSE(control->seq_new().has_value());
  EXPECT_EQ(kv->metrics().live_sequences, 1);
  std::atomic<bool> finished{false};
  ASSERT_EQ(
      cudaLaunchHostFunc(
          cudaStreamPerThread,
          [](void* state) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            static_cast<std::atomic<bool>*>(state)->store(true);
          },
          &finished),
      cudaSuccess);
  ASSERT_EQ(kv->abort_step(), Error::Ok);
  EXPECT_TRUE(finished.load());
  ASSERT_TRUE(control->seq_rm(*id));
  EXPECT_EQ(control->seq_new(), id);
  ASSERT_EQ(step(*cache_ptr, handle, {*id}), Error::Ok);
  EXPECT_TRUE(finished.load());
  EXPECT_EQ(control->pos(*id), 1);
  EXPECT_EQ(kv->metrics().live_sequences, 1);
  kv->forget_handle(&handle);
}

TEST_F(CudaBatchedSequenceCacheTest, FailedDeclaredStepCanCloseAndReuseId) {
  auto cache_ptr = make(/*initial=*/4);
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const auto id = control->seq_new();
  ASSERT_TRUE(id.has_value());
  ASSERT_TRUE(control->declare_step({*id}));
  EXPECT_EQ(kv->prepare_step(0, cudaStreamPerThread), Error::InvalidArgument);
  ASSERT_TRUE(control->seq_rm(*id));
  EXPECT_EQ(kv->prepare_step(1, cudaStreamPerThread), Error::InvalidArgument);
  EXPECT_EQ(control->seq_new(), id);
  ASSERT_EQ(step(*cache_ptr, handle, {*id}), Error::Ok);
  kv->forget_handle(&handle);
}

TEST_F(
    CudaRuntimeColumnCacheTest,
    AdmissionsGrowGeometricallyAndStepsKeepWidth) {
  auto cache_ptr = make();
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  std::vector<int32_t> ids;
  int expected_columns = 2;
  for (int count = 1; count <= kMaxSeqs; ++count) {
    const auto id = control->seq_new();
    ASSERT_TRUE(id.has_value());
    ids.push_back(*id);
    if (count >= expected_columns) {
      expected_columns =
          std::min(kDeclaredColumns, 2 * (expected_columns - 1) + 1);
    }
    EXPECT_EQ(kv->metrics().physical_columns, expected_columns) << count;
    const int64_t resizes = kv->metrics().column_resize_count;
    for (int repeat = 0; repeat < 2; ++repeat) {
      ASSERT_EQ(step(*cache_ptr, handle, ids), Error::Ok);
      EXPECT_EQ(kv->metrics().physical_columns, expected_columns);
      EXPECT_EQ(kv->metrics().column_resize_count, resizes);
    }
  }
  EXPECT_FALSE(control->seq_new().has_value());
  kv->forget_handle(&handle);
}

TEST_F(CudaRuntimeColumnCacheTest, ActiveStepRejectsResizeUntilExplicitAbort) {
  auto cache_ptr = make();
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const auto id = control->seq_new();
  ASSERT_TRUE(id.has_value());
  // Simulate a failed forward: prepare runs, commit never does.
  ASSERT_TRUE(control->declare_step({*id}));
  ASSERT_EQ(kv->prepare_step(1, cudaStreamPerThread), Error::Ok);
  const auto before = kv->metrics();
  EXPECT_EQ(kv->compact_columns(0), Error::InvalidState);
  EXPECT_EQ(kv->prepare_step(1, cudaStreamPerThread), Error::InvalidState);
  EXPECT_EQ(kv->metrics().column_resize_count, before.column_resize_count);
  EXPECT_EQ(kv->metrics().physical_columns, before.physical_columns);
  ASSERT_EQ(kv->abort_step(), Error::Ok);
  EXPECT_EQ(kv->abort_step(), Error::Ok);
  ASSERT_TRUE(control->declare_step({*id}));
  ASSERT_EQ(kv->prepare_step(1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv->commit_step(1), Error::Ok);
  EXPECT_EQ(control->pos(*id), 2);
  ASSERT_TRUE(control->seq_rm(*id));
  kv->forget_handle(&handle);
}

TEST_F(CudaRuntimeColumnCacheTest, FixedInfoABIIsRejected) {
  auto cache_ptr = make();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container(4);
  auto handle = make_handle(fake);
  EXPECT_EQ(kv->note_handle(&handle).error(), Error::InvalidProgram);
  kv->forget_handle(&handle);
}

TEST_F(CudaRuntimeColumnCacheTest, SameWidthCompactionKeepsMappingAndBindings) {
  auto cache_ptr = make();
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const auto first = control->seq_new();
  const auto second = control->seq_new();
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  ASSERT_EQ(step(*cache_ptr, handle, {*first, *second}), Error::Ok);
  ASSERT_TRUE(control->seq_rm(*first));
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  ASSERT_EQ(kv->metrics().physical_columns, 2);
  EXPECT_EQ(control->seq_new(), first);
  ASSERT_EQ(step(*cache_ptr, handle, {*first, *second}), Error::Ok);
  ASSERT_EQ(kv->metrics().physical_columns, 3);
  seed(fake, 2, 3);
  const auto before = read_column(fake.bound["flat_k"].data, 2, 3, 0);
  void* const pointer = fake.bound["flat_k"].data;
  const auto count = kv->metrics().column_resize_count;
  handle.cuda_graph_state.phase = cu::CudaGraphPhase::Warmup;
  handle.cuda_graph_state.warmup_remaining = 0;
  ASSERT_EQ(kv->compact_columns(kMaxSeqs), Error::Ok);
  EXPECT_EQ(kv->metrics().physical_columns, 3);
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  EXPECT_EQ(kv->metrics().column_resize_count, count);
  EXPECT_EQ(fake.bound["flat_k"].data, pointer);
  EXPECT_EQ(handle.cuda_graph_state.phase, cu::CudaGraphPhase::Warmup);
  EXPECT_EQ(handle.cuda_graph_state.warmup_remaining, 0);
  EXPECT_EQ(read_column(pointer, 2, 3, 0), before);
  ASSERT_EQ(step(*cache_ptr, handle, {*first, *second}), Error::Ok);
  std::vector<std::array<int32_t, 5>> info(2);
  ASSERT_EQ(
      cudaMemcpy(
          info.data(),
          fake.bound["seq_info"].data,
          info.size() * sizeof(info[0]),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(
      info,
      (std::vector<std::array<int32_t, 5>>{{0, 1, 1, 2, 3}, {1, 1, 0, 3, 3}}));
  kv->forget_handle(&handle);
}

TEST_F(CudaRuntimeColumnCacheTest, RegistrySelectsTheRequestedColumnABI) {
  auto cfg = config(kCapacity, 4, kMaxWrite);
  cfg.max_seqs = kMaxSeqs;
  auto fixed = cache::CacheFactory::global().build(
      cu::kCudaBackendId, cache::kind::kBatchedSequence, flat_and_ring(2), cfg);
  auto runtime = cache::CacheFactory::global().build(
      cu::kCudaBackendId,
      cache::kind::kBatchedSequenceRuntime,
      flat_and_ring(2),
      cfg);
  ASSERT_TRUE(fixed.ok());
  ASSERT_TRUE(runtime.ok());
  EXPECT_EQ(
      fixed.get()->as<cu::CudaKVCache>()->metrics().physical_columns,
      kDeclaredColumns);
  EXPECT_EQ(
      runtime.get()->as<cu::CudaKVCache>()->metrics().physical_columns, 2);
}

TEST_F(CudaRuntimeColumnCacheTest, MaintenanceSurvivesDestroyedCallerStream) {
  auto cache_ptr = make();
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto fake = container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  const auto run = [&](const std::vector<int32_t>& ids) {
    cudaStream_t stream = nullptr;
    ASSERT_EQ(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_TRUE(control->declare_step(ids));
    ASSERT_EQ(kv->prepare_step(ids.size(), stream), Error::Ok);
    ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
    ASSERT_EQ(kv->commit_step(ids.size()), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  };
  const auto first = control->seq_new();
  ASSERT_TRUE(first.has_value());
  run({*first, *first, *first, *first});
  ASSERT_FALSE(HasFatalFailure());
  seed(fake, 4, 2);
  const auto before = read_column(fake.bound["flat_k"].data, 4, 2, 0);
  const auto clone = control->seq_clone(*first, 4);
  ASSERT_TRUE(clone.has_value());
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  EXPECT_EQ(read_column(fake.bound["flat_k"].data, 4, 3, 1), before);
  run({*clone});
  ASSERT_FALSE(HasFatalFailure());
  const auto third = control->seq_new();
  ASSERT_TRUE(third.has_value());
  EXPECT_EQ(kv->metrics().physical_columns, 5);
  run({*clone});
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_TRUE(control->seq_rm(*first));
  ASSERT_TRUE(control->seq_rm(*third));
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  EXPECT_EQ(kv->metrics().physical_columns, 2);
  EXPECT_EQ(read_column(fake.bound["flat_k"].data, 4, 2, 0), before);
  kv->forget_handle(&handle);
}

TEST_F(CudaRuntimeColumnCacheTest, ContractionInvalidatesEveryGraphAndBinding) {
  auto cache_ptr = make();
  auto* control = cache_ptr->as<cache::BatchControl>();
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  auto first = container();
  auto second = container();
  auto decode = make_handle(first);
  auto prefill = make_handle(second);
  ASSERT_TRUE(kv->note_handle(&decode).get());
  ASSERT_TRUE(kv->note_handle(&prefill).get());
  const auto a = control->seq_new();
  const auto b = control->seq_new();
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  ASSERT_EQ(step(*cache_ptr, decode, {*a, *b}), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&prefill), Error::Ok);
  const int first_updates = first.updates;
  const int second_updates = second.updates;
  decode.cuda_graph_state.phase = cu::CudaGraphPhase::Replay;
  prefill.cuda_graph_state.phase = cu::CudaGraphPhase::Warmup;
  prefill.cuda_graph_state.warmup_remaining = 0;
  ASSERT_TRUE(control->seq_rm(*a));
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  EXPECT_EQ(decode.cuda_graph_state.phase, cu::CudaGraphPhase::Warmup);
  EXPECT_EQ(decode.cuda_graph_state.warmup_remaining, 1);
  EXPECT_EQ(prefill.cuda_graph_state.warmup_remaining, 1);
  ASSERT_EQ(kv->rebind_for_execute(&decode), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&prefill), Error::Ok);
  EXPECT_EQ(first.updates, first_updates + 1);
  EXPECT_EQ(second.updates, second_updates + 1);
  EXPECT_EQ(first.bound["flat_k"].data, second.bound["flat_k"].data);
  kv->forget_handle(&decode);
  kv->forget_handle(&prefill);
}

TEST_F(CudaKVPoolTest, FailedColumnGrowthKeepsOldStorageAndBindingsUsable) {
  constexpr int kDeclaredColumns = 1 << 20;
  constexpr int kHugeDim = 1 << 20;
  cu::CudaKVPool pool(
      {{kHeads, kDim, kDeclaredColumns, true},
       {1, kHugeDim, kDeclaredColumns, true}},
      {},
      slimc10::ScalarType::BFloat16,
      kDeclaredColumns,
      kDeclaredColumns,
      2);
  auto fake = flat_and_ring_container();
  for (size_t index = 0; index < 4; ++index) {
    declare(
        fake,
        fake.fqns[index],
        slimc10::ScalarType::BFloat16,
        {1,
         kDeclaredColumns,
         index < 2 ? kHeads : 1,
         index < 2 ? kDim : kHugeDim});
  }
  auto handle = make_handle(fake);
  ASSERT_TRUE(pool.note_handle(&handle).get());
  ASSERT_EQ(pool.prepare(2, 0, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(pool.bind(&handle), Error::Ok);
  const auto history = iota_rows(2);
  ASSERT_EQ(
      cudaMemcpy(
          pool.layer_data(0, false),
          history.data(),
          history.size() * sizeof(uint16_t),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  void* old_k = pool.layer_data(0, false);
  void* old_large_k = pool.layer_data(1, false);
  const auto before_bytes = pool.allocated_bytes();
  const int updates = fake.updates;
  handle.cuda_graph_state.phase = cu::CudaGraphPhase::Replay;
  for (const bool trim_unused : {false, true}) {
    bool committed = true;
    EXPECT_EQ(
        pool.resize_columns(
            kDeclaredColumns,
            {{0, 0}},
            /*live_rows=*/1,
            cudaStreamPerThread,
            trim_unused,
            &committed),
        Error::MemoryAllocationFailed);
    EXPECT_FALSE(committed);
    EXPECT_EQ(pool.columns(), 2);
    EXPECT_EQ(pool.column_resize_count(), 0);
    EXPECT_EQ(pool.allocated_bytes(), before_bytes);
    EXPECT_EQ(pool.layer_data(0, false), old_k);
    EXPECT_EQ(pool.layer_data(1, false), old_large_k);
    EXPECT_EQ(handle.cuda_graph_state.phase, cu::CudaGraphPhase::Replay);
    ASSERT_EQ(pool.bind(&handle), Error::Ok);
    EXPECT_EQ(fake.updates, updates);
    EXPECT_EQ(read_rows(old_k, 2), history);
  }
  ASSERT_EQ(pool.prepare(2, 1, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(pool.mark_step_done(pool.stream()), Error::Ok);
  pool.forget_handle(&handle);
}

TEST_F(CudaBatchedSequenceCacheTest, BuilderValidatesAndIsRegistered) {
  auto no_seqs = sequence_config();
  no_seqs.max_seqs.reset();
  EXPECT_EQ(
      cu::make_cuda_batched_sequence_kv_cache(flat_and_ring(kWindow), no_seqs),
      nullptr);
  auto no_write = sequence_config();
  no_write.max_write.reset();
  EXPECT_EQ(
      cu::make_cuda_batched_sequence_kv_cache(flat_and_ring(kWindow), no_write),
      nullptr);
  auto built = cache::CacheFactory::global().build(
      cu::kCudaBackendId,
      cache::kind::kBatchedSequence,
      flat_and_ring(kWindow),
      sequence_config());
  ASSERT_TRUE(built.ok());
  EXPECT_NE(built.get()->as<cache::BatchControl>(), nullptr);
  EXPECT_NE(built.get()->as<cu::CudaKVCache>(), nullptr);
}

namespace {

class CudaPrivateCacheTest : public CudaBatchedSequenceCacheTest {
 protected:
  static constexpr int kOwners = 64;
  static constexpr int kPrivateWindow = 5;
  static constexpr int kPrivateWrite = 128;

  static cache::CacheConfig private_config(int max_write = kPrivateWrite) {
    auto cfg = config(256, 2, max_write);
    cfg.max_seqs = kOwners;
    return cfg;
  }

  static std::shared_ptr<cache::Cache> make_private(
      int max_write = kPrivateWrite) {
    return cu::make_cuda_batched_sequence_kv_cache(
        flat_and_ring(kPrivateWindow), private_config(max_write), false, true);
  }

  static FakeContainer private_container(int max_write = kPrivateWrite) {
    FakeContainer out{
        {"flat_k",
         "flat_v",
         "ring_k",
         "ring_v",
         "cols",
         "info",
         "steps",
         "stage_rows",
         "flat_layout",
         "ring_layout",
         "k_stage",
         "v_stage"},
        {"__et_offgraph_kv_layer_0_k",
         "__et_offgraph_kv_layer_0_v",
         "__et_offgraph_kv_layer_1_k",
         "__et_offgraph_kv_layer_1_v",
         "__et_offgraph_kv_seq_cols",
         "__et_offgraph_kv_seq_info",
         "__et_offgraph_kv_owner_step",
         "__et_offgraph_kv_stage_rows",
         "__et_offgraph_kv_layer_0_owner_layout",
         "__et_offgraph_kv_layer_1_owner_layout",
         "__et_offgraph_kv_layer_1_k_stage",
         "__et_offgraph_kv_layer_1_v_stage"},
        {},
        0};
    for (int i = 0; i < 4; ++i) {
      declare(
          out,
          out.fqns[i],
          slimc10::ScalarType::BFloat16,
          {1,
           (i < 2 ? 256 : kPrivateWindow + max_write - 1) * (kOwners + 1),
           kHeads,
           kDim});
    }
    declare(out, out.fqns[4], slimc10::ScalarType::Long, {max_write});
    declare(out, out.fqns[5], slimc10::ScalarType::Int, {2 * kOwners, 4});
    declare(out, out.fqns[6], slimc10::ScalarType::Int, {kOwners + 1, 3});
    declare(out, out.fqns[7], slimc10::ScalarType::Long, {max_write});
    for (int i : {8, 9}) {
      declare(out, out.fqns[i], slimc10::ScalarType::Int, {kOwners + 1, 2});
    }
    for (int i : {10, 11}) {
      declare(
          out,
          out.fqns[i],
          slimc10::ScalarType::BFloat16,
          {1, max_write, kHeads, kDim});
    }
    return out;
  }

  static std::vector<int32_t> read_ints(void* device, int count) {
    std::vector<int32_t> out(count);
    EXPECT_EQ(
        cudaMemcpy(
            out.data(),
            device,
            count * sizeof(int32_t),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    return out;
  }

  static std::vector<uint16_t> payload(int owner, int position, bool value) {
    std::vector<uint16_t> row(kRow);
    for (int lane = 0; lane < kRow; ++lane) {
      row[lane] = static_cast<uint16_t>(
          owner * 100 + position * 7 + lane + (value ? 4000 : 1));
    }
    return row;
  }

  // Stand in only for the GPU kernels. The production cache allocates real
  // CUDA storage, uploads metadata and archives/restores exact payload bytes.
  static Error write_step(
      cache::Cache& ptr,
      FakeContainer& fake,
      cu::CudaDelegateHandle& handle,
      const std::vector<int32_t>& owners) {
    auto* control = ptr.as<cache::BatchControl>();
    auto* kv = ptr.as<cu::CudaKVCache>();
    if (!control->declare_step(owners)) {
      return Error::InvalidArgument;
    }
    ET_CHECK_OK_OR_RETURN_ERROR(
        kv->prepare_step(owners.size(), cudaStreamPerThread));
    ET_CHECK_OK_OR_RETURN_ERROR(kv->rebind_for_execute(&handle));
    const auto flat =
        read_ints(fake.bound["flat_layout"].data, (kOwners + 1) * 2);
    const auto ring =
        read_ints(fake.bound["ring_layout"].data, (kOwners + 1) * 2);
    const auto steps = read_ints(fake.bound["steps"].data, (kOwners + 1) * 3);
    const auto stage = read_cols(fake.bound["stage_rows"].data, owners.size());
    std::vector<int> consumed(kOwners, 0);
    for (size_t token = 0; token < owners.size(); ++token) {
      const int owner = owners[token];
      const int position = steps[owner * 3 + 1] + consumed[owner]++;
      const int end = steps[owner * 3 + 1] + steps[owner * 3 + 2];
      for (bool value : {false, true}) {
        const auto row = payload(owner, position, value);
        const auto copy = [&](const char* name, int64_t index) {
          auto* data = static_cast<uint16_t*>(fake.bound[name].data);
          return cudaMemcpy(
                     data + index * kRow,
                     row.data(),
                     kRow * sizeof(uint16_t),
                     cudaMemcpyHostToDevice) == cudaSuccess;
        };
        if (!copy(value ? "flat_v" : "flat_k", flat[owner * 2] + position) ||
            !copy(value ? "v_stage" : "k_stage", stage[token])) {
          return Error::Internal;
        }
        if (position >= end - ring[owner * 2 + 1] &&
            !copy(
                value ? "ring_v" : "ring_k",
                ring[owner * 2] + position % ring[owner * 2 + 1])) {
          return Error::Internal;
        }
      }
    }
    return kv->commit_step(owners.size());
  }

  static void expect_history(FakeContainer& fake, int owner, int keep) {
    for (bool ring : {false, true}) {
      const auto layout = read_ints(
          fake.bound[ring ? "ring_layout" : "flat_layout"].data,
          (kOwners + 1) * 2);
      for (bool value : {false, true}) {
        auto* data =
            static_cast<uint16_t*>(fake.bound
                                       [ring ? (value ? "ring_v" : "ring_k")
                                             : (value ? "flat_v" : "flat_k")]
                                           .data);
        for (int position = ring ? std::max(0, keep - kPrivateWindow) : 0;
             position < keep;
             ++position) {
          const int physical =
              ring ? position % layout[owner * 2 + 1] : position;
          EXPECT_EQ(
              read_rows(data + (layout[owner * 2] + physical) * kRow, 1),
              payload(owner, position, value));
        }
      }
    }
  }
};

} // namespace

TEST_F(CudaPrivateCacheTest, MetricsRemainSafeAfterLaterHistoryCreationFails) {
  auto geometry = flat_and_ring(kPrivateWindow);
  auto invalid_ring = geometry.layers.back();
  invalid_ring.n_kv_heads = 1 << 30;
  invalid_ring.head_dim = 1 << 30;
  geometry.layers.push_back(invalid_ring);
  auto cache_ptr = cu::make_cuda_batched_sequence_kv_cache(
      geometry, private_config(), false, true);
  ASSERT_NE(cache_ptr, nullptr);
  auto* kv = cache_ptr->as<cu::CudaKVCache>();
  const auto metrics = kv->metrics();
  EXPECT_EQ(metrics.host_undo_bytes, 0);
  EXPECT_EQ(metrics.device_staging_bytes, 0);
  EXPECT_EQ(kv->prepare_step(1, cudaStreamPerThread), Error::InvalidArgument);
}

TEST_F(CudaPrivateCacheTest, SixtyFourOwnersHaveIndependentRowsAndSharedStage) {
  auto ptr = make_private();
  ASSERT_NE(ptr, nullptr);
  auto* ctl = ptr->as<cache::BatchControl>();
  auto* kv = ptr->as<cu::CudaKVCache>();
  auto fake = private_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  std::vector<int32_t> owners;
  for (int i = 0; i < kOwners; ++i) {
    ASSERT_EQ(ctl->seq_new(), i);
    owners.push_back(i);
  }
  EXPECT_EQ(kv->metrics().allocated_bytes, 0);
  ASSERT_EQ(write_step(*ptr, fake, handle, owners), Error::Ok);
  ASSERT_EQ(
      write_step(*ptr, fake, handle, std::vector<int32_t>(127, 0)), Error::Ok);
  const auto flat =
      read_ints(fake.bound["flat_layout"].data, (kOwners + 1) * 2);
  const auto ring =
      read_ints(fake.bound["ring_layout"].data, (kOwners + 1) * 2);
  EXPECT_EQ(flat[1], 128);
  EXPECT_EQ(ring[1], kPrivateWindow);
  for (int owner = 1; owner < kOwners; ++owner) {
    EXPECT_EQ(flat[owner * 2 + 1], 2);
    EXPECT_EQ(ring[owner * 2 + 1], 2);
  }
  const int64_t row_bytes = kRow * sizeof(uint16_t);
  EXPECT_EQ(
      kv->metrics().allocated_bytes,
      2 * (flat[kOwners * 2] + ring[kOwners * 2] + 2) * row_bytes);
  expect_history(fake, 0, 128);
  expect_history(fake, 63, 1);
  kv->forget_handle(&handle);
}

TEST_F(CudaPrivateCacheTest, FragmentationRetainsSlackAcrossDecodeSteps) {
  auto ptr = make_private();
  ASSERT_NE(ptr, nullptr);
  auto* ctl = ptr->as<cache::BatchControl>();
  auto* kv = ptr->as<cu::CudaKVCache>();
  auto fake = private_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  std::vector<int32_t> owners;
  for (int owner = 0; owner < kOwners; ++owner) {
    ASSERT_EQ(ctl->seq_new(), owner);
    owners.push_back(owner);
  }
  for (int length = 1; length <= 3; ++length) {
    ASSERT_EQ(write_step(*ptr, fake, handle, owners), Error::Ok);
  }
  const auto before = kv->metrics();
  const auto ring_before =
      read_ints(fake.bound["ring_layout"].data, (kOwners + 1) * 2);
  for (int owner = 0; owner < kOwners; ++owner) {
    ASSERT_EQ(ring_before[owner * 2 + 1], 4);
  }
  ASSERT_EQ(write_step(*ptr, fake, handle, owners), Error::Ok);
  EXPECT_EQ(kv->metrics().growth_count, before.growth_count);
  EXPECT_EQ(
      read_ints(fake.bound["ring_layout"].data, (kOwners + 1) * 2),
      ring_before);
  expect_history(fake, 0, 4);
  expect_history(fake, kOwners - 1, 4);
  kv->forget_handle(&handle);
}

TEST_F(CudaPrivateCacheTest, WrappedCloneAndRewindRestoreArchivedHistory) {
  auto ptr = make_private();
  auto* ctl = ptr->as<cache::BatchControl>();
  auto* kv = ptr->as<cu::CudaKVCache>();
  auto fake = private_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  ASSERT_EQ(ctl->seq_new(), 0);
  ASSERT_EQ(
      write_step(*ptr, fake, handle, std::vector<int32_t>(4, 0)), Error::Ok);
  ASSERT_EQ(
      write_step(*ptr, fake, handle, std::vector<int32_t>(8, 0)), Error::Ok);
  const auto source_layout =
      read_ints(fake.bound["ring_layout"].data, (kOwners + 1) * 2);
  ASSERT_EQ(source_layout[0], 0);
  ASSERT_EQ(source_layout[1], kPrivateWindow);
  ASSERT_EQ(ctl->seq_clone(0, 8), 1);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  // Private spans are densely packed by owner; only owners 0 and 1 have rows.
  // Device descriptors refresh at prepare, so inspect the committed allocation
  // at the clone's expected span without appending another token.
  const int clone_base = source_layout[0] + source_layout[1];
  for (bool value : {false, true}) {
    auto* data =
        static_cast<uint16_t*>(fake.bound[value ? "ring_v" : "ring_k"].data);
    for (int position = 4; position < 8; ++position) {
      EXPECT_EQ(
          read_rows(data + (clone_base + position % kPrivateWindow) * kRow, 1),
          payload(0, position, value));
    }
  }
  expect_history(fake, 0, 12);
  ASSERT_TRUE(ctl->seq_rm(0));
  ASSERT_TRUE(ctl->rewind(1, 6));
  for (bool value : {false, true}) {
    auto* data =
        static_cast<uint16_t*>(fake.bound[value ? "ring_v" : "ring_k"].data);
    for (int position = 2; position < 6; ++position) {
      EXPECT_EQ(
          read_rows(data + (clone_base + position % kPrivateWindow) * kRow, 1),
          payload(0, position, value));
    }
  }
  ASSERT_EQ(write_step(*ptr, fake, handle, {1}), Error::Ok);
  EXPECT_EQ(ctl->pos(1), 7);
  kv->forget_handle(&handle);
}

TEST_F(CudaPrivateCacheTest, ExactRewindFloorAndCloneMatchTheLegacyContract) {
  for (int max_write : {1, kPrivateWrite}) {
    auto ptr = make_private(max_write);
    auto* ctl = ptr->as<cache::BatchControl>();
    auto* kv = ptr->as<cu::CudaKVCache>();
    auto fake = private_container(max_write);
    auto handle = make_handle(fake);
    ASSERT_TRUE(kv->note_handle(&handle).get());
    ASSERT_EQ(ctl->seq_new(), 0);
    const int written = max_write == 1 ? 12 : 256;
    for (int position = 0; position < written; position += max_write) {
      ASSERT_EQ(
          write_step(*ptr, fake, handle, std::vector<int32_t>(max_write, 0)),
          Error::Ok);
    }
    const int floor = written - max_write;
    EXPECT_FALSE(ctl->rewind(0, floor - 1));
    EXPECT_FALSE(ctl->seq_clone(0, floor - 1));
    ASSERT_EQ(ctl->seq_clone(0, floor), 1);
    ASSERT_TRUE(ctl->rewind(0, floor));
    ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
    for (int owner : {0, 1}) {
      for (bool value : {false, true}) {
        auto* data = static_cast<uint16_t*>(
            fake.bound[value ? "ring_v" : "ring_k"].data);
        for (int position = floor - kPrivateWindow + 1; position < floor;
             ++position) {
          EXPECT_EQ(
              read_rows(
                  data +
                      (owner * kPrivateWindow + position % kPrivateWindow) *
                          kRow,
                  1),
              payload(0, position, value));
        }
      }
    }
    ASSERT_EQ(write_step(*ptr, fake, handle, {1}), Error::Ok);
    EXPECT_EQ(ctl->pos(1), floor + 1);
    EXPECT_EQ(ctl->pos(0), floor);
    kv->forget_handle(&handle);
  }
}

TEST_F(CudaPrivateCacheTest, AbortedForwardRetiresOnlyItsOwners) {
  auto ptr = make_private();
  auto* ctl = ptr->as<cache::BatchControl>();
  auto* kv = ptr->as<cu::CudaKVCache>();
  auto fake = private_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  ASSERT_EQ(ctl->seq_new(), 0);
  ASSERT_EQ(ctl->seq_new(), 1);
  ASSERT_EQ(ctl->seq_new(), 2);
  ASSERT_EQ(write_step(*ptr, fake, handle, {0, 1, 2}), Error::Ok);
  ASSERT_TRUE(ctl->declare_step({0, 2}));
  ASSERT_EQ(kv->prepare_step(2, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  const auto layout =
      read_ints(fake.bound["ring_layout"].data, (kOwners + 1) * 2);
  for (const int owner : {0, 2}) {
    // Simulate a partially executed Ring write, with both an old and the new
    // row corrupted. Admission must never reuse this owner's claimed history.
    ASSERT_EQ(
        cudaMemset(
            static_cast<uint16_t*>(fake.bound["ring_v"].data) +
                layout[owner * 2] * kRow,
            0xff,
            layout[owner * 2 + 1] * kRow * sizeof(uint16_t)),
        cudaSuccess);
  }
  EXPECT_FALSE(ctl->seq_rm(0));
  EXPECT_FALSE(ctl->seq_clone(0, 1));
  ASSERT_EQ(kv->abort_step(), Error::Ok);
  ASSERT_EQ(kv->abort_step(), Error::Ok);
  EXPECT_FALSE(ctl->declare_step({0}));
  EXPECT_FALSE(ctl->declare_step({1, 2}));
  EXPECT_FALSE(ctl->seq_clone(0, 1));
  EXPECT_FALSE(ctl->rewind(2, 1));
  expect_history(fake, 1, 1);
  ASSERT_EQ(write_step(*ptr, fake, handle, {1}), Error::Ok);
  expect_history(fake, 1, 2);
  ASSERT_EQ(ctl->seq_new(), 3);
  ASSERT_TRUE(ctl->seq_rm(0));
  ASSERT_EQ(ctl->seq_new(), 0);
  ASSERT_EQ(write_step(*ptr, fake, handle, {0, 3}), Error::Ok);
  expect_history(fake, 0, 1);
  expect_history(fake, 3, 1);
  ASSERT_TRUE(ctl->seq_rm(2));
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  ASSERT_EQ(write_step(*ptr, fake, handle, {0, 1, 3}), Error::Ok);
  kv->forget_handle(&handle);
}

TEST_F(CudaPrivateCacheTest, CachePhaseTimesAreCumulativeAcrossClear) {
  auto ptr = make_private();
  auto* ctl = ptr->as<cache::BatchControl>();
  auto* kv = ptr->as<cu::CudaKVCache>();
  auto fake = private_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  EXPECT_EQ(kv->metrics().prepare_time_ns, 0);
  EXPECT_EQ(kv->metrics().commit_time_ns, 0);
  ASSERT_EQ(ctl->seq_new(), 0);
  ASSERT_EQ(write_step(*ptr, fake, handle, {0}), Error::Ok);
  const auto before = kv->metrics();
  EXPECT_GT(before.prepare_time_ns, 0);
  EXPECT_GT(before.archive_transfer_time_ns, 0);
  EXPECT_GT(before.archive_append_time_ns, 0);
  EXPECT_GE(
      before.commit_time_ns,
      before.archive_transfer_time_ns + before.archive_append_time_ns);
  ctl->clear();
  const auto cleared = kv->metrics();
  EXPECT_EQ(cleared.prepare_time_ns, before.prepare_time_ns);
  EXPECT_EQ(cleared.commit_time_ns, before.commit_time_ns);
  EXPECT_EQ(cleared.archive_transfer_time_ns, before.archive_transfer_time_ns);
  EXPECT_EQ(cleared.archive_append_time_ns, before.archive_append_time_ns);
  ASSERT_EQ(ctl->seq_new(), 0);
  ASSERT_EQ(write_step(*ptr, fake, handle, {0}), Error::Ok);
  const auto after = kv->metrics();
  EXPECT_GT(after.prepare_time_ns, before.prepare_time_ns);
  EXPECT_GT(after.commit_time_ns, before.commit_time_ns);
  kv->forget_handle(&handle);
}

TEST_F(CudaPrivateCacheTest, CurrentCloneAndRewindPreserveResidentRows) {
  auto ptr = make_private();
  auto* ctl = ptr->as<cache::BatchControl>();
  auto* kv = ptr->as<cu::CudaKVCache>();
  auto fake = private_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  ASSERT_EQ(ctl->seq_new(), 0);
  ASSERT_EQ(
      write_step(*ptr, fake, handle, std::vector<int32_t>(12, 0)), Error::Ok);
  auto& graph = handle.cuda_graph_state;
  graph.phase = cu::CudaGraphPhase::Replay;
  const void* flat_before_clone = fake.bound["flat_k"].data;
  ASSERT_EQ(ctl->seq_clone(0, 12), 1);
  ASSERT_EQ(kv->rebind_for_execute(&handle), Error::Ok);
  if (fake.bound["flat_k"].data != flat_before_clone) {
    EXPECT_EQ(graph.phase, cu::CudaGraphPhase::Warmup);
    EXPECT_EQ(graph.warmup_remaining, 1);
  } else {
    EXPECT_EQ(graph.phase, cu::CudaGraphPhase::Replay);
  }
  for (bool value : {false, true}) {
    auto* data =
        static_cast<uint16_t*>(fake.bound[value ? "ring_v" : "ring_k"].data);
    for (int position = 8; position < 12; ++position) {
      EXPECT_EQ(
          read_rows(
              data + (kPrivateWindow + position % kPrivateWindow) * kRow, 1),
          payload(0, position, value));
    }
  }
  graph.phase = cu::CudaGraphPhase::Replay;
  const auto growth = kv->metrics().growth_count;
  const auto bytes = kv->metrics().allocated_bytes;
  ASSERT_TRUE(ctl->rewind(1, 12));
  EXPECT_EQ(graph.phase, cu::CudaGraphPhase::Replay);
  EXPECT_EQ(kv->metrics().growth_count, growth);
  EXPECT_EQ(kv->metrics().allocated_bytes, bytes);
  ASSERT_TRUE(ctl->rewind(1, 11));
  EXPECT_EQ(graph.phase, cu::CudaGraphPhase::Replay);
  for (bool value : {false, true}) {
    auto* data =
        static_cast<uint16_t*>(fake.bound[value ? "ring_v" : "ring_k"].data);
    for (int position = 7; position < 11; ++position) {
      EXPECT_EQ(
          read_rows(
              data + (kPrivateWindow + position % kPrivateWindow) * kRow, 1),
          payload(0, position, value));
    }
  }
  ASSERT_EQ(write_step(*ptr, fake, handle, {1}), Error::Ok);
  EXPECT_EQ(ctl->pos(0), 12);
  EXPECT_EQ(ctl->pos(1), 12);
  kv->forget_handle(&handle);
}

TEST_F(CudaPrivateCacheTest, FailedPreparationOrCompactionFenceRequiresClear) {
  for (bool compact : {false, true}) {
    SCOPED_TRACE(compact);
    auto ptr = make_private();
    auto* ctl = ptr->as<cache::BatchControl>();
    auto* kv = ptr->as<cu::CudaKVCache>();
    auto fake = private_container();
    auto handle = make_handle(fake);
    ASSERT_TRUE(kv->note_handle(&handle).get());
    ASSERT_EQ(ctl->seq_new(), 0);
    ASSERT_EQ(ctl->seq_new(), 1);
    ASSERT_EQ(write_step(*ptr, fake, handle, {0, 1}), Error::Ok);
    if (compact) {
      ASSERT_TRUE(ctl->seq_rm(1));
    }
    cu::CudaKVPoolTestPeer fences;
    if (compact) {
      fences.fail_device_fence = true;
      EXPECT_EQ(kv->compact_columns(0), Error::Internal);
    } else {
      ASSERT_TRUE(ctl->declare_step({0, 0}));
      fences.fail_stream_fence = true;
      EXPECT_EQ(kv->prepare_step(2, cudaStreamPerThread), Error::Internal);
      EXPECT_EQ(kv->abort_step(), Error::Ok);
    }
    EXPECT_FALSE(ctl->declare_step({0}));
    EXPECT_FALSE(ctl->seq_new());
    EXPECT_FALSE(ctl->seq_clone(0, 1));
    EXPECT_EQ(kv->compact_columns(0), Error::InvalidState);
    ctl->clear();
    ASSERT_EQ(ctl->seq_new(), 0);
    ASSERT_EQ(write_step(*ptr, fake, handle, {0}), Error::Ok);
    expect_history(fake, 0, 1);
    kv->forget_handle(&handle);
  }
}

TEST_F(CudaPrivateCacheTest, CompactLiveReusedOwnerDropsLongCapacity) {
  auto ptr = make_private();
  auto* ctl = ptr->as<cache::BatchControl>();
  auto* kv = ptr->as<cu::CudaKVCache>();
  auto fake = private_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  ASSERT_EQ(ctl->seq_new(), 0);
  ASSERT_EQ(ctl->seq_new(), 1);
  ASSERT_EQ(
      write_step(*ptr, fake, handle, std::vector<int32_t>(128, 0)), Error::Ok);
  ASSERT_EQ(
      write_step(*ptr, fake, handle, std::vector<int32_t>(12, 1)), Error::Ok);
  ASSERT_TRUE(ctl->seq_rm(0));
  ASSERT_EQ(ctl->seq_new(), 0);
  ASSERT_EQ(write_step(*ptr, fake, handle, {0}), Error::Ok);
  const auto before = kv->metrics().allocated_bytes;
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  EXPECT_LT(kv->metrics().allocated_bytes, before);
  // Upload the new descriptors without moving the short owner's positions.
  ASSERT_EQ(write_step(*ptr, fake, handle, {0}), Error::Ok);
  const auto flat =
      read_ints(fake.bound["flat_layout"].data, (kOwners + 1) * 2);
  const auto ring =
      read_ints(fake.bound["ring_layout"].data, (kOwners + 1) * 2);
  EXPECT_EQ(flat[1], 2);
  EXPECT_EQ(ring[1], 2);
  EXPECT_EQ(ring[3], kPrivateWindow);
  expect_history(fake, 0, 2);
  expect_history(fake, 1, 12);
  const auto stable = kv->metrics().allocated_bytes;
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  EXPECT_EQ(kv->metrics().allocated_bytes, stable);
  kv->forget_handle(&handle);
}

TEST_F(CudaPrivateCacheTest, CompactPreservesLiveDecodeSlackAndExactHistory) {
  auto ptr = make_private();
  auto* ctl = ptr->as<cache::BatchControl>();
  auto* kv = ptr->as<cu::CudaKVCache>();
  auto fake = private_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  ASSERT_EQ(ctl->seq_new(), 0);
  ASSERT_EQ(ctl->seq_new(), 1);
  ASSERT_EQ(
      write_step(*ptr, fake, handle, std::vector<int32_t>(128, 0)), Error::Ok);
  for (int length = 1; length <= 9; ++length) {
    ASSERT_EQ(write_step(*ptr, fake, handle, {1}), Error::Ok);
  }
  const auto before = kv->metrics().allocated_bytes;
  ASSERT_TRUE(ctl->seq_rm(0));
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  EXPECT_LT(kv->metrics().allocated_bytes, before);
  const auto growths = kv->metrics().growth_count;
  ASSERT_EQ(write_step(*ptr, fake, handle, {1}), Error::Ok);
  EXPECT_EQ(kv->metrics().growth_count, growths);
  const auto flat =
      read_ints(fake.bound["flat_layout"].data, (kOwners + 1) * 2);
  EXPECT_EQ(flat[3], 16);
  void* flat_data = fake.bound["flat_k"].data;
  void* ring_data = fake.bound["ring_k"].data;
  for (int length = 11; length <= 16; ++length) {
    ASSERT_EQ(write_step(*ptr, fake, handle, {1}), Error::Ok);
    EXPECT_EQ(kv->metrics().growth_count, growths);
    EXPECT_EQ(fake.bound["flat_k"].data, flat_data);
    EXPECT_EQ(fake.bound["ring_k"].data, ring_data);
    expect_history(fake, 1, length);
  }
  kv->forget_handle(&handle);
}

TEST_F(CudaPrivateCacheTest, IdleCompactionDropsClosedPrivateSpans) {
  auto ptr = make_private();
  auto* ctl = ptr->as<cache::BatchControl>();
  auto* kv = ptr->as<cu::CudaKVCache>();
  auto fake = private_container();
  auto handle = make_handle(fake);
  ASSERT_TRUE(kv->note_handle(&handle).get());
  std::vector<int32_t> owners;
  for (int i = 0; i < kOwners; ++i) {
    ASSERT_EQ(ctl->seq_new(), i);
    owners.push_back(i);
  }
  ASSERT_EQ(write_step(*ptr, fake, handle, owners), Error::Ok);
  const auto before = kv->metrics().allocated_bytes;
  for (int i = 1; i < kOwners; ++i) {
    ASSERT_TRUE(ctl->seq_rm(i));
  }
  ASSERT_EQ(kv->compact_columns(0), Error::Ok);
  EXPECT_EQ(kv->metrics().physical_columns, 2);
  EXPECT_LT(kv->metrics().allocated_bytes, before);
  ASSERT_EQ(write_step(*ptr, fake, handle, {0}), Error::Ok);
  expect_history(fake, 0, 2);
  kv->forget_handle(&handle);
}
