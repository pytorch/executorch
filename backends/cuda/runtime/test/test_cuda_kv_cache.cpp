/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/cuda/runtime/cuda_kv_cache.h>
#include <executorch/backends/cuda/runtime/cuda_kv_pool.h>
#include <executorch/backends/cuda/runtime/backend_options.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/cache/cell_cache.h>
#include <executorch/runtime/core/evalue.h>
#include <gtest/gtest.h>

#include <algorithm>
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
  std::unordered_map<size_t, size_t> data_size_override{};
  std::unordered_map<size_t, int32_t> dtype_override{};
  size_t last_update_pairs{0};
  // Compiled dtype and bytes by FQN, for constants make_cache() cannot
  // describe: storage built straight on a CudaKVPool, and side buffers.
  std::unordered_map<std::string, std::pair<int32_t, size_t>> declared{};
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

// Bytes the compiled program declares for one layer's K or V: BSHD at the
// maximum rows, i.e. the capacity, or window + max_write - 1 for a ring.
size_t declared_bytes(
    const cache::LayerGeometry& layer,
    const cache::CacheConfig& cfg) {
  const int64_t rows = layer.policy.kind == cache::LayerPolicy::Kind::Ring
      ? layer.policy.window + cfg.max_write.value_or(1) - 1
      : cfg.capacity;
  return static_cast<size_t>(rows) * layer.n_kv_heads * layer.head_dim *
      slimc10::elementSize(static_cast<slimc10::ScalarType>(cfg.kv_dtype));
}

size_t layer_of(const std::string& fqn) {
  const std::string prefix = "__et_offgraph_kv_layer_";
  return std::stoul(fqn.substr(prefix.size()));
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
  const auto it = fake->data_size_override.find(index);
  if (it != fake->data_size_override.end()) {
    *data_size = it->second;
    return Error::Ok;
  }
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

TEST_F(CudaKVCacheTest, FlatLayersGrowGeometricallyAndKeepHistory) {
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

  ASSERT_EQ(kv.prepare_step(3, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  constexpr int kRingRows = kWindow + kMaxWrite - 1;
  EXPECT_EQ(kv.metrics().flat_capacity, 4);
  EXPECT_EQ(kv.metrics().growth_count, 0);
  EXPECT_EQ(
      kv.metrics().allocated_bytes,
      2 * (4 + kRingRows) * kRow * static_cast<int64_t>(sizeof(uint16_t)));

  // Stand in for the kernels: the step writes rows [0, 3).
  const auto history = iota_rows(3);
  ASSERT_EQ(
      cudaMemcpy(
          container.bound["flat_k"].data,
          history.data(),
          history.size() * sizeof(uint16_t),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  void* first_flat = container.bound["flat_k"].data;
  void* first_ring = container.bound["ring_k"].data;
  ASSERT_EQ(kv.commit_step(3), Error::Ok);

  // 3 + 2 > 4: doubles to 8, moves the flat storage, keeps what was written.
  ASSERT_EQ(kv.prepare_step(2, cudaStreamPerThread), Error::Ok);
  ASSERT_EQ(kv.rebind_for_execute(&handle), Error::Ok);
  EXPECT_EQ(kv.metrics().flat_capacity, 8);
  EXPECT_EQ(kv.metrics().growth_count, 1);
  EXPECT_NE(container.bound["flat_k"].data, first_flat);
  EXPECT_EQ(container.bound["ring_k"].data, first_ring);
  EXPECT_EQ(read_rows(container.bound["flat_k"].data, 3), history);
  ASSERT_EQ(kv.commit_step(2), Error::Ok);

  // A step wider than doubling asks for: grows straight to what it needs.
  ASSERT_EQ(kv.prepare_step(9, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(kv.metrics().flat_capacity, 16);
  ASSERT_EQ(kv.commit_step(9), Error::Ok);
  ASSERT_EQ(kv.prepare_step(9, cudaStreamPerThread), Error::Ok);
  EXPECT_EQ(kv.metrics().flat_capacity, 32);
  EXPECT_EQ(kv.metrics().growth_count, 3);
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
  const size_t compiled_ring_bytes =
      static_cast<size_t>(kWindow + 8 - 1) * kRow * sizeof(uint16_t);
  container.data_size_override = {
      {2, compiled_ring_bytes}, {3, compiled_ring_bytes}};
  auto handle = make_handle(container);

  EXPECT_EQ(kv.note_handle(&handle).error(), Error::InvalidProgram);
  EXPECT_EQ(container.updates, 0);
}

TEST_F(CudaKVCacheTest, CacheNotMatchingTheCompiledCapacityIsRejected) {
  auto cache_ptr = make_cache(flat_and_ring(4), config(64, 4, 4));
  ASSERT_NE(cache_ptr, nullptr);
  auto& kv = *cache_ptr->as<cu::CudaKVCache>();
  auto container = flat_and_ring_container();
  const size_t compiled_flat_bytes = 128 * kRow * sizeof(uint16_t);
  container.data_size_override = {
      {0, compiled_flat_bytes}, {1, compiled_flat_bytes}};
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
  graph.note_graph_allocations();
  // Only the allocation the graph leaves outstanding is its to free.
  ASSERT_EQ(graph.graph_allocations.size(), 1);
  EXPECT_EQ(graph.graph_allocations[0], output);

  ASSERT_EQ(cudaGraphLaunch(graph.graph_exec, stream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_GE(graph_mem_in_use(), before + kBytes);

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
    declare(container, "__et_offgraph_kv_cells", slimc10::ScalarType::Long, {16});
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
  ASSERT_EQ(pool->mark_step_done(), Error::Ok);

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
  ASSERT_EQ(pool->mark_step_done(), Error::Ok);

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
  // CPU constants: the 128-byte cells buffer stays 128, but a 16-byte one
  // reads as 64. Both forms describe the same declared shape.
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
  container.declared["__et_offgraph_kv_read_len"] = {
      static_cast<int32_t>(slimc10::ScalarType::Long), 64};
  auto handle = make_handle(container);
  EXPECT_TRUE(pool.note_handle(&handle).get());

  // Rounded past the next multiple of 64 is a different shape.
  container.declared["__et_offgraph_kv_read_len"].second = 128;
  EXPECT_EQ(pool.note_handle(&handle).error(), Error::InvalidProgram);
  pool.forget_handle(&handle);
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
  declare(wrong_dtype, "__et_offgraph_kv_cells", slimc10::ScalarType::Int, {32});
  auto wrong_dtype_handle = make_handle(wrong_dtype);
  EXPECT_EQ(
      pool->note_handle(&wrong_dtype_handle).error(), Error::InvalidProgram);
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

  static std::shared_ptr<cache::Cache> make(int initial = 4) {
    return cu::make_cuda_cell_kv_cache(
        flat_and_ring(kWindow), config(kCells, initial, kMaxWrite));
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
    declare(container, "__et_offgraph_kv_cells", slimc10::ScalarType::Long, {kMaxWrite});
    declare(container, "__et_offgraph_kv_read_len", slimc10::ScalarType::Long, {1});
    for (const char* mask :
         {"__et_offgraph_kv_mask_w0", "__et_offgraph_kv_mask_w2"}) {
      declare(
          container, mask, slimc10::ScalarType::Bool, {1, 1, kMaxWrite, kCells});
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
  static std::vector<std::vector<int>> read_mask(void* device, int rows, int cols) {
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
  static Error step(
      cache::Cache& cache,
      cu::CudaDelegateHandle& handle,
      const std::vector<int32_t>& seq_ids) {
    auto* control = cache.as<cache::BatchControl>();
    auto* kv = cache.as<cu::CudaKVCache>();
    if (!control->declare_step(seq_ids)) {
      return Error::InvalidArgument;
    }
    const auto width = static_cast<int64_t>(seq_ids.size());
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
  EXPECT_EQ(read_longs(fake.bound["cells"].data, 5),
            std::vector<int64_t>({0, 1, 2, 3, 4}));
  EXPECT_EQ(read_longs(fake.bound["read_len"].data, 1), std::vector<int64_t>({5}));
  using Mask = std::vector<std::vector<int>>;
  EXPECT_EQ(
      read_mask(fake.bound["mask0"].data, 5, 5),
      Mask({{1, 0, 0, 0, 0},
            {1, 1, 0, 0, 0},
            {1, 1, 1, 0, 0},
            {0, 0, 0, 1, 0},
            {0, 0, 0, 1, 1}}));
  // The window-2 layer drops a's position 0 for its position-2 query.
  EXPECT_EQ(
      read_mask(fake.bound["mask2"].data, 5, 5),
      Mask({{1, 0, 0, 0, 0},
            {1, 1, 0, 0, 0},
            {0, 1, 1, 0, 0},
            {0, 0, 0, 1, 0},
            {0, 0, 0, 1, 1}}));
  EXPECT_EQ(fake.bound["mask0"].sizes, std::vector<int64_t>({1, 1, kMaxWrite, kCells}));
  EXPECT_EQ(fake.bound["cells"].dtype, slimc10::ScalarType::Long);
  EXPECT_EQ(fake.bound["mask0"].dtype, slimc10::ScalarType::Bool);
  // Pools are declared at every cell; allocated past the first step's reach.
  EXPECT_EQ(fake.bound["r_k"].sizes, std::vector<int64_t>({1, kCells, kHeads, kDim}));
  EXPECT_EQ(kv->metrics().flat_capacity, 5);

  // Then they decode together, in the other order.
  ASSERT_EQ(step(*cache_ptr, handle, {b, a}), Error::Ok);
  EXPECT_EQ(read_longs(fake.bound["cells"].data, 2), std::vector<int64_t>({5, 6}));
  EXPECT_EQ(read_longs(fake.bound["read_len"].data, 1), std::vector<int64_t>({7}));
  EXPECT_EQ(
      read_mask(fake.bound["mask0"].data, 2, 7),
      Mask({{0, 0, 0, 1, 1, 1, 0}, {1, 1, 1, 0, 0, 0, 1}}));
  EXPECT_EQ(control->pos(a), 4);
  EXPECT_EQ(control->pos(b), 3);
  const auto metrics = kv->metrics();
  EXPECT_EQ(metrics.logical_length, 7);
  EXPECT_EQ(metrics.flat_capacity, 10);
  EXPECT_EQ(metrics.growth_count, 1);
  kv->forget_handle(&handle);
}

TEST_F(CudaCellCacheTest, SwitchingSequencesKeepsACapturedGraph) {
  // Room for every step below but the last: the pool grows ahead of placement
  // to where the step could reach, used_end + width.
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
  ASSERT_EQ(step(*cache_ptr, prefill, {a, a, a, b, b, b}), Error::Ok);
  EXPECT_NE(graph.phase, cu::CudaGraphPhase::Replay);
  ASSERT_EQ(step(*cache_ptr, decode, {a}), Error::Ok);
  EXPECT_NE(decode_fake.bound["f_k"].data, bound.at("f_k").data);
  EXPECT_EQ(decode_fake.bound["cells"].data, bound.at("cells").data);
  EXPECT_EQ(decode_fake.bound["mask2"].data, bound.at("mask2").data);
  kv->forget_handle(&prefill);
  kv->forget_handle(&decode);
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

  auto* cells = static_cast<cache::CellCache*>(cache_ptr.get());
  ASSERT_TRUE(control->seq_rm(a));
  EXPECT_EQ(cells->free_cells(), kCells - 4);
  ASSERT_TRUE(control->seq_rm(*c));
  EXPECT_EQ(cells->free_cells(), kCells);
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
  // Declared two, the program's input carries three.
  ASSERT_TRUE(control->declare_step({a, a}));
  EXPECT_EQ(kv->prepare_step(3, cudaStreamPerThread), Error::InvalidState);
  // Wider than the step buffers the program declared.
  EXPECT_EQ(
      kv->prepare_step(kMaxWrite + 1, cudaStreamPerThread),
      Error::InvalidArgument);
  // More tokens than free cells never declares.
  EXPECT_FALSE(control->declare_step(std::vector<int32_t>(kCells + 1, a)));
  // A rejected step leaves the table untouched.
  EXPECT_EQ(control->pos(a), 0);
  kv->forget_handle(&handle);
}

TEST_F(CudaCellCacheTest, BuilderValidatesAndIsRegisteredForBatchedKinds) {
  auto no_max_write = config(kCells, 4, kMaxWrite);
  no_max_write.max_write.reset();
  EXPECT_EQ(cu::make_cuda_cell_kv_cache(flat_and_ring(kWindow), no_max_write), nullptr);
  EXPECT_EQ(
      cu::make_cuda_cell_kv_cache(
          flat_and_ring(kWindow), config(kCells, 4, kCells + 1)),
      nullptr);
  for (const char* kind : {cache::kind::kBatchedCell, cache::kind::kBatched}) {
    auto built = cache::CacheFactory::global().build(
        cu::kCudaBackendId, kind, flat_and_ring(kWindow), config(kCells, 4, kMaxWrite));
    ASSERT_TRUE(built.ok()) << kind;
    EXPECT_NE(built.get()->as<cache::BatchControl>(), nullptr) << kind;
    EXPECT_NE(built.get()->as<cu::CudaKVCache>(), nullptr) << kind;
  }
}
