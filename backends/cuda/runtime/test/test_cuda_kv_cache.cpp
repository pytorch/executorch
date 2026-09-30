/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/aoti/slim/core/slim_tensor.h>
#include <executorch/backends/cuda/runtime/cuda_kv_cache.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
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
};

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
  *dtype = it != fake->dtype_override.end() ? it->second
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
