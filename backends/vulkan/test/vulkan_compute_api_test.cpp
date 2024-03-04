/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <ATen/native/vulkan/api/api.h>

#include <ATen/native/vulkan/impl/Common.h>
#include <ATen/native/vulkan/impl/Packing.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Arithmetic.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

using namespace at::native::vulkan;

//
// Utilities
//

#define CREATE_FLOAT_TEXTURE(sizes, allocate_memory) \
  vTensor(                                           \
      api::context(),                                \
      sizes,                                         \
      api::kFloat,                                   \
      api::StorageType::TEXTURE_3D,                  \
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,  \
      allocate_memory);

#define CREATE_FLOAT_BUFFER(sizes, allocate_memory) \
  vTensor(                                          \
      api::context(),                               \
      sizes,                                        \
      api::kFloat,                                  \
      api::StorageType::BUFFER,                     \
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, \
      allocate_memory);

void fill_vtensor(vTensor& vten, std::vector<float>& data) {
  api::StorageBuffer staging_buffer(api::context(), api::kFloat, data.size());

  copy_ptr_to_staging(data.data(), staging_buffer, vten.gpu_nbytes());

  if (vten.storage_type() == api::StorageType::BUFFER) {
    packing::record_nchw_to_buffer_op(
        api::context(), staging_buffer.buffer(), vten, {}, VK_NULL_HANDLE);
  } else {
    api::ShaderInfo compute_shader = packing::get_nchw_to_image_shader(vten);
    packing::record_nchw_to_image_op(
        api::context(),
        compute_shader,
        staging_buffer.buffer(),
        vten,
        {},
        VK_NULL_HANDLE);
  }
}

void fill_vtensor(ComputeGraph& graph, const IOValueRef idx, float val) {
  std::vector<float> data(graph.get_val(idx.value).toTensor().gpu_numel());
  std::fill(data.begin(), data.end(), val);

  graph.copy_into_staging(idx.staging, data.data(), data.size());
}

void extract_vtensor(vTensor& vten, std::vector<float>& data) {
  api::StorageBuffer staging_buffer(
      api::context(), api::kFloat, vten.gpu_numel());

  if (vten.storage_type() == api::StorageType::BUFFER) {
    packing::record_buffer_to_nchw_op(
        api::context(), vten, staging_buffer.buffer(), {}, VK_NULL_HANDLE);
  } else {
    api::ShaderInfo compute_shader = packing::get_image_to_nchw_shader(vten);
    packing::record_image_to_nchw_op(
        api::context(),
        compute_shader,
        vten,
        staging_buffer.buffer(),
        {},
        VK_NULL_HANDLE);
  }

  api::VulkanFence fence = api::context()->fences().get_fence();
  api::context()->submit_cmd_to_gpu(fence.get_submit_handle());
  fence.wait();

  copy_staging_to_ptr(staging_buffer, data.data(), vten.gpu_nbytes());
}

api::MemoryAllocation allocate_memory_for(const vTensor& vten) {
  return api::context()->adapter_ptr()->vma().create_allocation(
      vten.get_memory_requirements(), vten.get_allocation_create_info());
}

VmaTotalStatistics get_vma_stats() {
  return api::context()->adapter_ptr()->vma().get_memory_statistics();
}

size_t get_vma_allocation_count() {
  return get_vma_stats().total.statistics.allocationCount;
}

GraphConfig generate_graph_config() {
  const uint32_t submit_frequency = UINT32_MAX;

  const api::CommandPoolConfig cmd_config{
      4u, // cmdPoolInitialSize
      2u, // cmdPoolBatchSize
  };

  const api::DescriptorPoolConfig descriptor_pool_config{
      1024u, // descriptorPoolMaxSets
      1024u, // descriptorUniformBufferCount
      1024u, // descriptorStorageBufferCount
      1024u, // descriptorCombinedSamplerCount
      1024u, // descriptorStorageImageCount
      32u, // descriptorPileSizes
  };

  const api::QueryPoolConfig query_pool_config{};

  const api::ContextConfig context_config{
      submit_frequency, // cmdSubmitFrequency
      cmd_config, // cmdPoolConfig
      descriptor_pool_config, // descriptorPoolConfig
      query_pool_config, // queryPoolConfig
  };

  const GraphConfig graph_config{
      context_config,
  };

  return graph_config;
}

//
// Test Wrapper
//

class VulkanComputeAPITest : public ::testing::Test {
 public:
  void SetUp() override {
    // Make sure we are starting with a clean slate
    EXPECT_TRUE(get_vma_allocation_count() == 0);
  }

  void TearDown() override {
    api::context()->flush();

    // Make sure we are ending with a clean slate
    EXPECT_TRUE(get_vma_allocation_count() == 0);
  }
};

//
// Compute API Tests
//

TEST_F(VulkanComputeAPITest, buffer_copy_sanity_check) {
  // Simple test that copies data into a and reads from a
  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_BUFFER(sizes, /*allocate_memory = */ true);

  // Input data
  std::vector<float> data_in(a.gpu_numel());
  std::fill(data_in.begin(), data_in.end(), 2.524f);

  // Fill input tensor
  fill_vtensor(a, data_in);

  // Read back data
  std::vector<float> data_out(a.gpu_numel());
  extract_vtensor(a, data_out);

  // Check output
  for (const auto& d : data_out) {
    EXPECT_TRUE(d == 2.524f);
  }
}

TEST_F(VulkanComputeAPITest, buffer_deferred_allocation_test) {
  // Same as buffer_copy_sanity_check, but defers memory allocation

  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_BUFFER(sizes, /*allocate_memory = */ false);

  // For buffer storage, a small uniform buffer is allocated containing size and
  // stride data, which is why the check is for 1 allocation below.
  EXPECT_TRUE(get_vma_allocation_count() == 1);

  // Input data
  std::vector<float> data_in(a.gpu_numel());
  std::fill(data_in.begin(), data_in.end(), 1.234f);

  // Allocate memory at the last possible opportunity
  api::MemoryAllocation a_mem = allocate_memory_for(a);
  a.buffer().bind_allocation(a_mem);

  EXPECT_TRUE(get_vma_allocation_count() == 2);

  // Fill input tensor
  fill_vtensor(a, data_in);

  // Read back data
  std::vector<float> data_out(a.gpu_numel());
  extract_vtensor(a, data_out);

  // Check output
  for (const auto& d : data_out) {
    EXPECT_TRUE(d == 1.234f);
  }
}

TEST_F(VulkanComputeAPITest, texture_add_sanity_check) {
  // Simple test that performs a + b -> c

  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  // Input data
  std::vector<float> data_a(a.gpu_numel());
  std::fill(data_a.begin(), data_a.end(), 2.5f);
  std::vector<float> data_b(b.gpu_numel());
  std::fill(data_b.begin(), data_b.end(), 1.5f);

  // Add shader kernel
  api::ShaderInfo kernel = arithmetic::get_shader(arithmetic::OpType::ADD);

  // Fill input tensors
  fill_vtensor(a, data_a);
  fill_vtensor(b, data_b);

  // a + b -> c
  arithmetic::record_op(api::context(), kernel, a, b, c, 1.0f);

  // Extract output tensor
  std::vector<float> data_out(c.gpu_numel());
  extract_vtensor(c, data_out);

  // Check output
  for (const auto& d : data_out) {
    EXPECT_TRUE(d == 4.0f);
  }
}

TEST_F(VulkanComputeAPITest, texture_deferred_allocation_test) {
  // This test is the same as texture_add_sanity_check, except that the tensor
  // memory is allocated in a deferred fashion

  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);

  // No allocations made yet
  EXPECT_TRUE(get_vma_allocation_count() == 0);

  std::vector<float> data_a(a.gpu_numel());
  std::fill(data_a.begin(), data_a.end(), 2.5f);
  std::vector<float> data_b(b.gpu_numel());
  std::fill(data_b.begin(), data_b.end(), 1.5f);

  api::ShaderInfo kernel = arithmetic::get_shader(arithmetic::OpType::ADD);

  // Allocate memory at the last possible opportunity
  api::MemoryAllocation a_mem = allocate_memory_for(a);
  a.image().bind_allocation(a_mem);
  api::MemoryAllocation b_mem = allocate_memory_for(b);
  b.image().bind_allocation(b_mem);
  api::MemoryAllocation c_mem = allocate_memory_for(c);
  c.image().bind_allocation(c_mem);

  // One allocation for each tensor
  EXPECT_TRUE(get_vma_allocation_count() == 3);

  fill_vtensor(a, data_a);
  fill_vtensor(b, data_b);

  arithmetic::record_op(api::context(), kernel, a, b, c, 1.0f);

  std::vector<float> data_c(c.gpu_numel());
  extract_vtensor(c, data_c);

  for (const auto& val : data_c) {
    EXPECT_TRUE(val == 4.0f);
  }
}

TEST_F(VulkanComputeAPITest, texture_resource_aliasing_test) {
  // This test performs the following operations:
  // 1. a + b -> c
  // 2. c + d -> e
  // and share memory between tensors whenever possible.

  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor d = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor e = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);

  // No allocations made yet
  EXPECT_TRUE(get_vma_allocation_count() == 0);

  // a and d can share the same memory allocation
  api::MemoryAllocation a_d_mem = allocate_memory_for(a);
  a.image().bind_allocation(a_d_mem);
  d.image().bind_allocation(a_d_mem);
  // b and e can share the same memory allocation
  api::MemoryAllocation b_e_mem = allocate_memory_for(b);
  b.image().bind_allocation(b_e_mem);
  e.image().bind_allocation(b_e_mem);
  // c must have its own memory allocation
  api::MemoryAllocation c_mem = allocate_memory_for(c);
  c.image().bind_allocation(c_mem);

  // Only 3 allocations should be made
  EXPECT_TRUE(get_vma_allocation_count() == 3);

  // Specify input data
  std::vector<float> data_a(a.gpu_numel());
  std::fill(data_a.begin(), data_a.end(), 2.5f);
  std::vector<float> data_b(b.gpu_numel());
  std::fill(data_b.begin(), data_b.end(), 1.5f);
  std::vector<float> data_d(b.gpu_numel());
  std::fill(data_d.begin(), data_d.end(), 1.0f);

  // Get shader kernel for add
  api::ShaderInfo kernel = arithmetic::get_shader(arithmetic::OpType::ADD);

  // First, fill a and b with data
  fill_vtensor(a, data_a);
  fill_vtensor(b, data_b);

  // a + b -> c
  arithmetic::record_op(api::context(), kernel, a, b, c, 1.0f);

  // Now d can be filled with data
  fill_vtensor(d, data_d);

  // c + d -> e
  arithmetic::record_op(api::context(), kernel, c, d, e, 1.0f);

  // Extract data from e
  std::vector<float> data_e(e.gpu_numel());
  extract_vtensor(e, data_e);

  // Sanity check that the values are correct
  for (const auto& val : data_e) {
    EXPECT_TRUE(val == 5.0f);
  }
}

TEST_F(VulkanComputeAPITest, resource_bind_twice_fails) {
  // Check that binding a resource that already has memory associated with it
  // fails

  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  // Try to double bind a resource, which should fail
  api::MemoryAllocation a_mem = allocate_memory_for(a);
  EXPECT_THROW(a.image().bind_allocation(a_mem), api::Error);
}

TEST_F(VulkanComputeAPITest, resource_destructor_non_owning_memory) {
  // Check that the destructor of a vTensor that does not own its memory
  // does not free the memory

  api::MemoryAllocation memory;

  // Default MemoryAllocation constructor should not allocate memory
  EXPECT_TRUE(get_vma_allocation_count() == 0);

  std::vector<int64_t> sizes = {4, 4, 1};
  {
    vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);

    memory = allocate_memory_for(a);
    EXPECT_TRUE(get_vma_allocation_count() == 1);
    a.image().bind_allocation(memory);
  }

  // Check that the memory is still allocated
  EXPECT_TRUE(get_vma_allocation_count() == 1);
}

TEST_F(VulkanComputeAPITest, use_non_bound_textures_fails) {
  // Try to encode a command buffer with a vTensor that does not have memory

  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);

  // No allocations made yet
  EXPECT_TRUE(get_vma_allocation_count() == 0);

  std::vector<float> data_a(a.gpu_numel());
  std::fill(data_a.begin(), data_a.end(), 2.5f);

  // Encoding a command buffer with a vTensor without memory should throw
  EXPECT_THROW(fill_vtensor(a, data_a), api::Error);
}

//
// Compute Graph Tests
//

#define EXTRACT_TENSOR(name)                             \
  std::vector<float> data_##name(                        \
      graph.get_val(name.value).toTensor().gpu_numel()); \
  graph.copy_from_staging(name.staging, data_##name.data(), data_##name.size());

TEST(VulkanComputeGraphTest, test_simple_graph) {
  GraphConfig config = generate_graph_config();
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {4, 4, 4};
  std::vector<int64_t> size_small = {4, 4, 1};

  // Build graph

  IOValueRef a = graph.add_input_tensor(size_big, api::kFloat);
  IOValueRef b = graph.add_input_tensor(size_small, api::kFloat);

  IOValueRef out = {};

  out.value = add_arithmetic_node(graph, a.value, b.value, 1.0, VK_KERNEL(add));

  out.staging = graph.set_output_tensor(out.value);

  graph.encode_execute();

  // Run graph

  for (float i = 5.0f; i < 30.0f; i += 10.0f) {
    float val_a = i + 2.0f;
    float val_b = i + 1.5f;
    float val_c = val_a + val_b;

    fill_vtensor(graph, a, val_a);
    fill_vtensor(graph, b, val_b);

    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (const auto& val : data_out) {
      EXPECT_TRUE(val == val_c);
    }
  }
}

#define CREATE_WEIGHT_TENSOR(name, sizes, val)                          \
  std::vector<float> data_##name(api::utils::multiply_integers(sizes)); \
  std::fill(data_##name.begin(), data_##name.end(), val);               \
  ValueRef name = graph.add_tensorref(sizes, api::kFloat, data_##name.data());

TEST(VulkanComputeGraphTest, test_simple_prepacked_graph) {
  GraphConfig config = generate_graph_config();
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {4, 4, 4};
  std::vector<int64_t> size_small = {4, 4, 1};

  CREATE_WEIGHT_TENSOR(w1, size_small, 3.5f);
  CREATE_WEIGHT_TENSOR(w2, size_small, 3.0f);

  // Build graph

  IOValueRef a = graph.add_input_tensor(size_big, api::kFloat);

  ValueRef c = add_arithmetic_node(graph, a.value, w1, 1.0, VK_KERNEL(add));
  ValueRef e = add_arithmetic_node(graph, c, w2, 1.0, VK_KERNEL(mul));

  IOValueRef out = {};
  out.value = e;
  out.staging = graph.set_output_tensor(out.value);

  graph.encode_prepack();
  graph.prepack();

  graph.encode_execute();

  // Run graph

  for (float i = 5.0f; i < 30.0f; i += 10.0f) {
    float val_out = (i + 3.5f) * 3.0f;

    fill_vtensor(graph, a, i);

    // Execute graph
    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (const auto& val : data_out) {
      EXPECT_TRUE(val == val_out);
    }
  }
}

TEST(VulkanComputeGraphTest, test_simple_shared_objects) {
  GraphConfig config = generate_graph_config();
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {4, 4, 4};
  std::vector<int64_t> size_small = {4, 4, 1};

  // Build graph

  IOValueRef a = graph.add_input_tensor(
      size_big,
      api::kFloat,
      /*shared_object_idx = */ 2);
  IOValueRef b = graph.add_input_tensor(
      size_small,
      api::kFloat,
      /*shared_object_idx = */ 4);

  // Allocation count will be 2 (1 staging buffer for each input tensor)
  EXPECT_TRUE(get_vma_allocation_count() == 2);

  ValueRef c = add_arithmetic_node(
      graph,
      a.value,
      b.value,
      1.0,
      VK_KERNEL(add),
      /*shared_object_idx = */ 6);

  IOValueRef d = graph.add_input_tensor(
      size_small,
      api::kFloat,
      /*shared_object_idx = */ 2);

  // Allocation count will be 3 (1 staging buffer for each input tensor)
  EXPECT_TRUE(get_vma_allocation_count() == 3);

  ValueRef e = add_arithmetic_node(
      graph,
      c,
      d.value,
      1.0,
      VK_KERNEL(mul),
      /*shared_object_idx = */ 4);

  IOValueRef out = {};
  out.value = e;
  out.staging = graph.set_output_tensor(out.value);

  // Allocation count will be 4 (1 staging buffer for each I/O tensor)
  EXPECT_TRUE(get_vma_allocation_count() == 4);

  graph.encode_execute();

  // Allocation count will be 13:
  // 4 staging buffers for each I/O tensor
  // 6 uniform buffers to store args for each shader dispatch
  // 3 shared objects to back tensor memory
  EXPECT_TRUE(get_vma_allocation_count() == 13);

  // Run graph

  for (float i = 4.0f; i < 30.0f; i += 7.0f) {
    float val_a = i + 2.0f;
    float val_b = i + 1.5f;
    float val_d = i + 1.0f;
    float val_out = (val_a + val_b) * val_d;

    fill_vtensor(graph, a, val_a);
    fill_vtensor(graph, b, val_b);
    fill_vtensor(graph, d, val_d);

    // Execute graph
    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (const auto& val : data_out) {
      EXPECT_TRUE(val == val_out);
    }
  }
}
