/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <ATen/native/vulkan/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OpUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Arithmetic.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

using namespace at::native::vulkan;

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

//
// Simplified versions of ATen Vulkan legacy functions
//

void record_nchw_to_buffer_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst) {
  uint32_t buf_len = api::utils::safe_downcast<uint32_t>(v_dst.gpu_numel());
  api::utils::uvec3 global_size = {buf_len, 1u, 1u};
  api::utils::uvec3 local_size = {32u, 1u, 1u};

  api::UniformParamsBuffer cpu_buffer_metadata(
      context, v_dst.get_cpu_buffer_metadata());
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      VK_KERNEL(buffer_to_buffer),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_dst.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_dst.buffer_metadata(),
      src_buffer,
      cpu_buffer_metadata.buffer());
}

bool record_buffer_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer) {
  uint32_t buf_len = api::utils::safe_downcast<uint32_t>(v_src.numel());
  api::utils::uvec3 global_size = {buf_len, 1u, 1u};
  api::utils::uvec3 local_size = {4u, 1u, 1u};

  api::UniformParamsBuffer cpu_buffer_metadata(
      context, v_src.get_cpu_buffer_metadata());
  api::PipelineBarrier pipeline_barrier{};

  return context->submit_compute_job(
      VK_KERNEL(buffer_to_buffer),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      dst_buffer,
      cpu_buffer_metadata.buffer(),
      v_src.buffer(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_src.buffer_metadata());
}

void record_nchw_to_image_op(
    api::Context* const context,
    api::VulkanBuffer& src_buffer,
    vTensor& v_dst) {
  api::utils::uvec3 global_size = v_dst.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  api::UniformParamsBuffer params(context, create_staging_params(v_dst));
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      get_nchw_to_image_shader(v_dst),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      src_buffer,
      params.buffer());
}

bool record_image_to_nchw_op(
    api::Context* const context,
    vTensor& v_src,
    api::VulkanBuffer& dst_buffer) {
  api::utils::uvec3 global_size = v_src.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  api::UniformParamsBuffer params(context, create_staging_params(v_src));
  api::PipelineBarrier pipeline_barrier{};

  return context->submit_compute_job(
      get_image_to_nchw_shader(v_src),
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_src.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      dst_buffer,
      params.buffer());
}

void record_arithmetic_op(
    api::Context* const context,
    const api::ShaderInfo& compute_shader,
    vTensor& v_in1,
    vTensor& v_in2,
    vTensor& v_dst,
    const float alpha) {
  api::utils::uvec3 global_size = v_dst.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  ArithmeticParams block{
      get_size_as_ivec4(v_dst),
      get_size_as_ivec4(v_in1),
      get_size_as_ivec4(v_in2),
      alpha,
  };
  api::UniformParamsBuffer params(context, block);
  api::PipelineBarrier pipeline_barrier{};

  context->submit_compute_job(
      compute_shader,
      pipeline_barrier,
      global_size,
      local_size,
      VK_NULL_HANDLE,
      v_dst.image(
          pipeline_barrier,
          api::PipelineStage::COMPUTE,
          api::MemoryAccessType::WRITE),
      v_in1.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      v_in2.image(pipeline_barrier, api::PipelineStage::COMPUTE),
      params.buffer());
}

//
// Utilities
//

void fill_vtensor(vTensor& vten, std::vector<float>& data) {
  api::StorageBuffer staging_buffer(api::context(), api::kFloat, data.size());

  copy_ptr_to_staging(data.data(), staging_buffer, vten.gpu_nbytes());

  if (vten.storage_type() == api::StorageType::BUFFER) {
    record_nchw_to_buffer_op(api::context(), staging_buffer.buffer(), vten);
  } else {
    record_nchw_to_image_op(api::context(), staging_buffer.buffer(), vten);
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
    record_buffer_to_nchw_op(api::context(), vten, staging_buffer.buffer());
  } else {
    record_image_to_nchw_op(api::context(), vten, staging_buffer.buffer());
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

TEST_F(VulkanComputeAPITest, retrieve_custom_shader_test) {
  // Try to get shader from custom shader library
  const api::ShaderInfo& kernel = VK_KERNEL(test_shader);

  EXPECT_TRUE(kernel.kernel_name == "test_shader");
}

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
  api::ShaderInfo kernel = VK_KERNEL(add);

  // Fill input tensors
  fill_vtensor(a, data_a);
  fill_vtensor(b, data_b);

  // a + b -> c
  record_arithmetic_op(api::context(), kernel, a, b, c, 1.0f);

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

  api::ShaderInfo kernel = VK_KERNEL(add);

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

  record_arithmetic_op(api::context(), kernel, a, b, c, 1.0f);

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
  api::ShaderInfo kernel = VK_KERNEL(add);

  // First, fill a and b with data
  fill_vtensor(a, data_a);
  fill_vtensor(b, data_b);

  // a + b -> c
  record_arithmetic_op(api::context(), kernel, a, b, c, 1.0f);

  // Now d can be filled with data
  fill_vtensor(d, data_d);

  // c + d -> e
  record_arithmetic_op(api::context(), kernel, c, d, e, 1.0f);

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

TEST(VulkanComputeGraphTest, test_values_scalars) {
  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef idx;

  idx = graph.add_scalar<int64_t>(4);
  EXPECT_TRUE(graph.get_val(idx).toInt() == 4);

  idx = graph.add_scalar<double>(5.5f);
  EXPECT_TRUE(graph.get_val(idx).toDouble() == 5.5f);
}

TEST(VulkanComputeGraphTest, test_values_scalar_list_inplace_constructed) {
  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef idx = graph.add_scalar_list<int64_t>({1, 2, 3, 4});
  std::vector<int64_t>& arr = graph.get_val(idx).toIntList();
  EXPECT_TRUE(arr.size() == 4);
  for (int i = 0; i < 4; i++) {
    EXPECT_TRUE(arr[i] == i + 1);
  }
}

TEST(VulkanComputeGraphTest, test_values_scalar_list_outside_constructed) {
  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef idx;
  {
    std::vector<double> data = {5.0, 4.0, 3.0, 2.0, 1.0};
    idx = graph.add_scalar_list(std::move(data));
  }
  std::vector<double>& arr = graph.get_val(idx).toDoubleList();
  EXPECT_TRUE(arr.size() == 5);
  for (int i = 0; i < 5; i++) {
    EXPECT_TRUE(arr[i] == (5 - i));
  }
}

TEST(VulkanComputeGraphTest, test_values_string) {
  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef idx;
  {
    std::string data = "hello, world";
    idx = graph.add_string(std::move(data));
  }
  std::string& stored = graph.get_val(idx).toString();
  EXPECT_TRUE(stored == "hello, world");
}

TEST(VulkanComputeGraphTest, test_simple_graph) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {4, 4, 4};
  std::vector<int64_t> size_small = {4, 4, 1};

  // Build graph

  IOValueRef a = graph.add_input_tensor(size_big, api::kFloat);
  IOValueRef b = graph.add_input_tensor(size_small, api::kFloat);

  IOValueRef out = {};

  out.value = graph.add_tensor(size_big, api::kFloat);

  auto addFn = VK_GET_OP_FN("aten.add.Tensor");
  addFn(graph, {a.value, b.value, kDummyValueRef, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
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
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {4, 4, 4};
  std::vector<int64_t> size_small = {4, 4, 1};

  CREATE_WEIGHT_TENSOR(w1, size_small, 3.5f);
  CREATE_WEIGHT_TENSOR(w2, size_small, 3.0f);

  // Build graph

  IOValueRef a = graph.add_input_tensor(size_big, api::kFloat);

  ValueRef c = graph.add_tensor(size_big, api::kFloat);
  ValueRef e = graph.add_tensor(size_big, api::kFloat);

  auto addFn = VK_GET_OP_FN("aten.add.Tensor");
  addFn(graph, {a.value, w1, kDummyValueRef, c});

  auto mulFn = VK_GET_OP_FN("aten.mul.Tensor");
  mulFn(graph, {c, w2, e});

  IOValueRef out = {};
  out.value = e;
  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();

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
  GraphConfig config;
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

  // Allocation count will be 4:
  // 1 uniform buffer for each staging shader args
  // 1 staging buffer for each input tensor
  EXPECT_TRUE(get_vma_allocation_count() == 4);

  ValueRef c = graph.add_tensor(
      size_big,
      api::kFloat,
      /*shared_object_idx = */ 6);

  auto addFn = VK_GET_OP_FN("aten.add.Tensor");
  addFn(graph, {a.value, b.value, kDummyValueRef, c});

  IOValueRef d = graph.add_input_tensor(
      size_small,
      api::kFloat,
      /*shared_object_idx = */ 2);

  // Allocation count will be 7, three are new:
  // 1 uniform buffer for arithmetic shader args
  // 1 uniform buffer for staging shader args
  // 1 staging buffer for the input tensor
  EXPECT_TRUE(get_vma_allocation_count() == 7);

  ValueRef e = graph.add_tensor(
      size_big,
      api::kFloat,
      /*shared_object_idx = */ 4);

  auto mulFn = VK_GET_OP_FN("aten.mul.Tensor");
  mulFn(graph, {c, d.value, e});

  IOValueRef out = {};
  out.value = e;
  out.staging = graph.set_output_tensor(out.value);

  // Allocation count will be 10, three are new:
  // 1 uniform buffer for arithmetic shader
  // 1 uniform buffer for staging shader
  // 1 staging buffer for the input tensor
  EXPECT_TRUE(get_vma_allocation_count() == 10);

  graph.prepare();
  graph.encode_execute();

  // Allocation count will be 13, three shared objects are allocated for total:
  // 4 staging buffers for each I/O tensor
  // 6 uniform buffers to store params for each shader dispatch
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
