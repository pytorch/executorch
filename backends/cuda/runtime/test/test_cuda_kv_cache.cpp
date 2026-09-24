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

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cu = ::executorch::backends::cuda;
namespace cache = ::executorch::extension::llm::cache;
namespace aoti = ::executorch::backends::aoti;
namespace slim = ::executorch::backends::aoti::slim;
namespace slimc10 = ::executorch::backends::aoti::slim::c10;
using ::executorch::runtime::Error;

namespace {

struct FakeContainer {
  std::vector<std::string> internal_names;
  std::vector<std::string> fqns;
  std::unordered_map<std::string, void*> pointers;
  std::unordered_map<std::string, slimc10::ScalarType> dtypes;
};

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

Error update_pairs(
    aoti::AOTInductorModelContainerHandle container,
    const aoti::AOTInductorConstantMapEntry* pairs,
    size_t count,
    bool,
    bool) {
  auto& pointers = reinterpret_cast<FakeContainer*>(container)->pointers;
  for (size_t index = 0; index < count; ++index) {
    auto* tensor = reinterpret_cast<slim::SlimTensor*>(pairs[index].handle);
    pointers[pairs[index].name] = tensor->data_ptr();
    reinterpret_cast<FakeContainer*>(container)->dtypes[pairs[index].name] =
        tensor->dtype();
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
  handle.update_user_managed_constant_buffer_pairs = update_pairs;
  return handle;
}

bool has_cuda_device() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

} // namespace

TEST(CudaKVCacheTest, AllocatesFullCapacityAndKeepsPointersStable) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device required";
  }

  cache::CacheConfig cfg;
  cfg.capacity = 32;
  // Deliberately below capacity: flat layers must ignore it and take the full
  // capacity, or the graph's static head stride addresses past the allocation.
  cfg.initial_capacity = 4;
  // The widest step below is 9 tokens, and a ring layer may not be handed
  // more than max_write at once.
  cfg.max_write = 9;
  cfg.kv_dtype = static_cast<int>(slimc10::ScalarType::BFloat16);
  cache::CacheGeometry geometry;
  geometry.layers = {
      {{cache::LayerPolicy::Kind::Flat, 0}, 2, 8},
      {{cache::LayerPolicy::Kind::Ring, 4}, 2, 8},
  };
  auto cache_ptr = cu::make_cuda_sequence_kv_cache(geometry, cfg);
  ASSERT_NE(cache_ptr, nullptr);
  auto& context = *cache_ptr->as<cu::CudaKVCache>();
  auto& control = *cache_ptr->as<cache::SequenceControl>();
  FakeContainer container{
      {"flat_k",
       "flat_v",
       "flat_capacity",
       "ring_k",
       "ring_v",
       "ring_capacity"},
      {"__et_offgraph_kv_layer_0_k",
       "__et_offgraph_kv_layer_0_v",
       "__et_offgraph_kv_layer_0_capacity",
       "__et_offgraph_kv_layer_1_k",
       "__et_offgraph_kv_layer_1_v",
       "__et_offgraph_kv_layer_1_capacity"},
      {},
      {}};
  auto handle = make_handle(container);
  {
    const auto serves = context.note_handle(&handle);
    ASSERT_EQ(serves.error(), Error::Ok);
    ASSERT_TRUE(serves.get());
  }
  EXPECT_EQ(context.metrics().allocated_bytes, 0);

  ASSERT_EQ(context.prepare_step(3), Error::Ok);
  ASSERT_EQ(context.rebind_for_execute(&handle), Error::Ok);
  const auto initial = context.metrics();
  EXPECT_EQ(initial.flat_capacity, 32);
  EXPECT_EQ(initial.growth_count, 0);
  EXPECT_EQ(initial.logical_length, 0);
  ASSERT_NE(container.pointers["flat_k"], nullptr);

  std::vector<uint16_t> values(32);
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<uint16_t>(index + 1);
  }
  ASSERT_EQ(
      cudaMemcpy(
          container.pointers["flat_k"],
          values.data(),
          values.size() * sizeof(uint16_t),
          cudaMemcpyHostToDevice),
      cudaSuccess);
  void* old_k = container.pointers["flat_k"];
  ASSERT_EQ(context.commit_step(3), Error::Ok);

  ASSERT_EQ(context.prepare_step(2), Error::Ok);
  ASSERT_EQ(context.rebind_for_execute(&handle), Error::Ok);
  // The allocation must not move: AOTI binds this pointer once and the strides
  // baked into the graph assume it stays put.
  EXPECT_EQ(container.pointers["flat_k"], old_k);
  const auto grown = context.metrics();
  EXPECT_EQ(grown.flat_capacity, 32);
  EXPECT_EQ(grown.growth_count, 0);
  EXPECT_EQ(grown.logical_length, 3);

  std::vector<uint16_t> copied(values.size());
  ASSERT_EQ(
      cudaMemcpy(
          copied.data(),
          container.pointers["flat_k"],
          copied.size() * sizeof(uint16_t),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(copied, values);

  ASSERT_EQ(context.commit_step(2), Error::Ok);
  ASSERT_EQ(context.prepare_step(1), Error::Ok);
  EXPECT_EQ(context.metrics().flat_capacity, 32);
  EXPECT_EQ(context.metrics().growth_count, 0);
  EXPECT_EQ(context.prepare_step(28), Error::InvalidArgument);
  EXPECT_EQ(context.metrics().logical_length, 5);
  ASSERT_EQ((control.clear(), Error::Ok), Error::Ok);
  const auto reset = context.metrics();
  EXPECT_EQ(reset.logical_length, 0);
  EXPECT_EQ(reset.flat_capacity, 32);
  EXPECT_EQ(reset.growth_count, 0);

  // Growth was the only thing that moved storage, and moving storage is what
  // invalidated a captured graph. Flat layers now take their full capacity up
  // front, so a step that fits capacity leaves every pointer where a captured
  // graph baked it in. A handle mid-replay is therefore served without
  // growing and without disturbing the capture.
  handle.cuda_graph_state.enable(3);
  handle.cuda_graph_state.phase = cu::CudaGraphPhase::Replay;
  ASSERT_EQ(context.prepare_step(9), Error::Ok);
  EXPECT_EQ(context.metrics().growth_count, 0);
  EXPECT_EQ(context.metrics().flat_capacity, 32);
  EXPECT_EQ(handle.cuda_graph_state.phase, cu::CudaGraphPhase::Replay);
  EXPECT_EQ(context.rebind_for_execute(&handle), Error::Ok);
  context.forget_handle(&handle);
}

TEST(CudaKVCacheTest, SupportedDenseDtypesControlStorageAndDescriptors) {
  if (!has_cuda_device()) {
    GTEST_SKIP() << "CUDA device required";
  }

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
    cache::CacheConfig cfg;
    cfg.capacity = 8;
    cfg.initial_capacity = 4;
    cfg.kv_dtype = static_cast<int>(dtype);
    cache::CacheGeometry geometry;
    geometry.layers = {{{cache::LayerPolicy::Kind::Flat, 0}, 2, 8}};
    auto cache_ptr = cu::make_cuda_sequence_kv_cache(geometry, cfg);
    ASSERT_NE(cache_ptr, nullptr);
    auto& context = *cache_ptr->as<cu::CudaKVCache>();
    FakeContainer container{
        {"flat_k", "flat_v", "flat_capacity"},
        {"__et_offgraph_kv_layer_0_k",
         "__et_offgraph_kv_layer_0_v",
         "__et_offgraph_kv_layer_0_capacity"},
        {},
        {}};
    auto handle = make_handle(container);
    {
      const auto serves = context.note_handle(&handle);
      ASSERT_EQ(serves.error(), Error::Ok);
      ASSERT_TRUE(serves.get());
    }

    ASSERT_EQ(context.prepare_step(4), Error::Ok);
    ASSERT_EQ(context.rebind_for_execute(&handle), Error::Ok);
    EXPECT_EQ(container.dtypes["flat_k"], dtype);
    EXPECT_EQ(container.dtypes["flat_v"], dtype);
    EXPECT_EQ(container.dtypes["flat_capacity"], slimc10::ScalarType::Long);
    const size_t element_size = slimc10::elementSize(dtype);
    EXPECT_EQ(
        context.metrics().allocated_bytes,
        2 * 2 * 8 * 8 * static_cast<int64_t>(element_size) +
            static_cast<int64_t>(sizeof(int64_t)));

    // The head stride is fixed at full capacity from the first allocation, so
    // a write stays where it was put instead of being relocated by a regrow.
    const size_t head_bytes = 8 * 8 * element_size;
    std::vector<uint8_t> values(2 * head_bytes);
    for (size_t index = 0; index < values.size(); ++index) {
      values[index] = static_cast<uint8_t>(index % 251 + 1);
    }
    ASSERT_EQ(
        cudaMemcpy(
            container.pointers["flat_k"],
            values.data(),
            values.size(),
            cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(context.commit_step(4), Error::Ok);

    ASSERT_EQ(context.prepare_step(1), Error::Ok);
    ASSERT_EQ(context.rebind_for_execute(&handle), Error::Ok);
    std::vector<uint8_t> copied(2 * head_bytes);
    ASSERT_EQ(
        cudaMemcpy(
            copied.data(),
            container.pointers["flat_k"],
            copied.size(),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(copied, values);
    context.forget_handle(&handle);
  }
}
