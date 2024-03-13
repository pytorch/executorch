/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <ATen/native/vulkan/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/test/utils/test_utils.h>

using namespace at::native::vulkan;

//
// Compute API Tests
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

TEST_F(VulkanComputeAPITest, retrieve_custom_shader_test) {
  // Try to get shader from custom shader library
  const api::ShaderInfo& kernel = VK_KERNEL(test_shader);

  EXPECT_TRUE(kernel.kernel_name == "test_shader");
}

TEST_F(VulkanComputeAPITest, update_params_between_submit) {
  api::context()->set_cmd(/*reusable = */ true);
  std::vector<int64_t> sizes = {4, 4, 2};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  struct Params final {
    api::utils::ivec3 size;
    int32_t fill;
    api::utils::vec4 values;
  };

  Params block{
      {2, 4, 1},
      0,
      {5.0, 5.0, 5.0, 5.0},
  };

  api::UniformParamsBuffer params(api::context(), block);

  {
    api::PipelineBarrier pipeline_barrier{};
    api::context()->submit_compute_job(
        VK_KERNEL(fill_texture__test),
        pipeline_barrier,
        {4, 4, 4},
        {4, 4, 4},
        VK_NULL_HANDLE,
        a.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        params.buffer());
  }

  api::StorageBuffer staging_buffer(api::context(), api::kFloat, a.gpu_numel());
  record_image_to_nchw_op(api::context(), a, staging_buffer.buffer());

  submit_to_gpu();
  check_staging_buffer(staging_buffer, 5.0f);

  Params new_block{
      {2, 4, 1},
      0,
      {4.0, 4.0, 4.0, 4.0},
  };

  params.update(new_block);

  submit_to_gpu();
  check_staging_buffer(staging_buffer, 4.0f);
}

TEST_F(VulkanComputeAPITest, texture_add_sanity_check) {
  // Simple test that performs a + b -> c

  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  // Fill input tensors
  fill_vtensor(a, 2.5f);
  fill_vtensor(b, 1.5f);

  // a + b -> c
  record_arithmetic_op(
      api::context(), VK_KERNEL(binary_add_nobroadcast__test), a, b, c);

  // Extract output tensor
  std::vector<float> data_out = extract_vtensor(c);

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

  api::ShaderInfo kernel = VK_KERNEL(binary_add_nobroadcast__test);

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

  record_arithmetic_op(api::context(), kernel, a, b, c);

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
  api::ShaderInfo kernel = VK_KERNEL(binary_add_nobroadcast__test);

  // First, fill a and b with data
  fill_vtensor(a, data_a);
  fill_vtensor(b, data_b);

  // a + b -> c
  record_arithmetic_op(api::context(), kernel, a, b, c);

  // Now d can be filled with data
  fill_vtensor(d, data_d);

  // c + d -> e
  record_arithmetic_op(api::context(), kernel, c, d, e);

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

TEST_F(VulkanComputeAPITest, tensor_reallocation_test) {
  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  execute_and_check_add(a, b, c, 3.0f, 5.0f);

  // Redo with new sizes
  std::vector<int64_t> new_sizes = {4, 6, 3};
  a.reallocate(new_sizes);
  b.reallocate(new_sizes);
  c.reallocate(new_sizes);

  // Flush everything
  api::context()->flush();

  execute_and_check_add(a, b, c, 12.0f, 10.0f);
}

TEST_F(
    VulkanComputeAPITest,
    tensor_reallocation_with_deferred_allocation_test) {
  std::vector<int64_t> sizes = {8, 8, 8};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);

  api::MemoryAllocation a_mem = allocate_memory_for(a);
  a.image().bind_allocation(a_mem);
  api::MemoryAllocation b_mem = allocate_memory_for(b);
  b.image().bind_allocation(b_mem);
  api::MemoryAllocation c_mem = allocate_memory_for(c);
  c.image().bind_allocation(c_mem);

  execute_and_check_add(a, b, c, 4.0f, 8.0f);

  std::vector<std::vector<int64_t>> new_sizes_list = {
      {4, 3, 5}, {4, 1, 7}, {8, 3, 2}, {8, 7, 2}};

  for (auto& new_sizes : new_sizes_list) {
    // Redo with new sizes
    a.reallocate(new_sizes);
    b.reallocate(new_sizes);
    c.reallocate(new_sizes);

    // Flush everything
    api::context()->flush();

    a.image().bind_allocation(a_mem);
    b.image().bind_allocation(b_mem);
    c.image().bind_allocation(c_mem);

    execute_and_check_add(
        a, b, c, float(new_sizes[1] + 4.5f), float(new_sizes[2] + 13.0f));
  }
}

TEST_F(VulkanComputeAPITest, texture_virtual_resize) {
  api::context()->set_cmd(/*reusable = */ true);
  std::vector<int64_t> sizes = {8, 12, 12};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  DEFINE_STAGING_BUFFER_AND_RECORD_TO_GPU_FOR(a)
  DEFINE_STAGING_BUFFER_AND_RECORD_TO_GPU_FOR(b)

  fill_staging(staging_buffer_a, 11.5f);
  fill_staging(staging_buffer_b, 12.5f);

  record_arithmetic_op(
      api::context(), VK_KERNEL(binary_add_nobroadcast__test), a, b, c);

  DEFINE_STAGING_BUFFER_AND_RECORD_FROM_GPU_FOR(c)

  submit_to_gpu();
  check_staging_buffer(staging_buffer_c, 24.0f);

  std::vector<std::vector<int64_t>> new_sizes_list = {
      {4, 2, 4}, {4, 3, 6}, {8, 12, 12}, {8, 1, 1}, {8, 11, 10}};

  for (auto& new_sizes : new_sizes_list) {
    a.virtual_resize(new_sizes);
    b.virtual_resize(new_sizes);
    c.virtual_resize(new_sizes);

    fill_staging(staging_buffer_a, float(new_sizes[1] + 1.5f), a.gpu_numel());
    fill_staging(staging_buffer_b, float(new_sizes[2] + 55.0f), b.gpu_numel());

    submit_to_gpu();
    check_staging_buffer(
        staging_buffer_c,
        float(new_sizes[1] + new_sizes[2] + 56.5f),
        c.gpu_numel());
  }
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

  std::vector<int64_t> size_big = {8, 64, 124};
  std::vector<int64_t> size_small = {8, 1, 124};

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

  std::vector<int64_t> size_big = {8, 73, 62};
  std::vector<int64_t> size_small = {8, 73, 1};

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

TEST(VulkanComputeGraphTest, test_simple_shared_objects_with_resize) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {12, 64, 64};
  std::vector<int64_t> size_small = {12, 64, 64};

  // Build graph

  IOValueRef a = graph.add_input_tensor(
      size_big,
      api::kFloat,
      /*shared_object_idx = */ 2);
  IOValueRef b = graph.add_input_tensor(
      size_small,
      api::kFloat,
      /*shared_object_idx = */ 4);

  // Allocation count will be 6:
  // 4: t.gpu_sizes_ubo(), t.cpu_sizes_ubo() for each staging shader
  // 2: staging buffer for each input tensor
  EXPECT_TRUE(get_vma_allocation_count() == 6);

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

  // Allocation count will be 11, 5 are new:
  // 2: out.gpu_sizes_ubo(), alpha UBO for arithmetic shader
  // 2: t.gpu_sizes_ubo(), t.cpu_sizes_ubo() uniform buffer for staging shader
  // 1: staging buffer for the input tensor
  EXPECT_TRUE(get_vma_allocation_count() == 11);

  ValueRef e = graph.add_tensor(
      size_big,
      api::kFloat,
      /*shared_object_idx = */ 4);

  auto mulFn = VK_GET_OP_FN("aten.mul.Tensor");
  mulFn(graph, {c, d.value, e});

  IOValueRef out = {};
  out.value = e;
  out.staging = graph.set_output_tensor(out.value);

  // Allocation count will be 15, 4 are new:
  // 1: alpha UBO for arithmetic shader
  // 2: t.gpu_sizes_ubo(), t.cpu_sizes_ubo() for staging shader
  // 1 staging buffer for the input tensor
  EXPECT_TRUE(get_vma_allocation_count() == 15);

  graph.prepare();
  graph.encode_execute();

  // Allocation count will be 18, 3 are new:
  // 3: shared memory allocations for tensors
  EXPECT_TRUE(get_vma_allocation_count() == 18);

  // Run graph

  std::vector<std::vector<int64_t>> new_sizes_list = {
      {8, 44, 34}, {4, 13, 56}, {8, 12, 64}, {12, 55, 33}, {4, 54, 10}};

  for (auto& new_sizes : new_sizes_list) {
    graph.get_val(a.value).toTensor().virtual_resize(new_sizes);
    graph.get_val(b.value).toTensor().virtual_resize(new_sizes);
    graph.get_val(c).toTensor().virtual_resize(new_sizes);
    graph.get_val(d.value).toTensor().virtual_resize(new_sizes);
    graph.get_val(e).toTensor().virtual_resize(new_sizes);

    float val_a = new_sizes[1] + 4.0f;
    float val_b = new_sizes[2] + 1.5f;
    float val_d = new_sizes[0] + 2.0f;
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

  std::vector<std::vector<int64_t>> new_sizes_list_2 = {
      {8, 44, 34}, {4, 13, 56}, {8, 12, 64}, {12, 55, 33}, {4, 54, 10}};

  for (auto& new_sizes : new_sizes_list_2) {
    graph.resize_input(0, new_sizes);
    graph.resize_input(1, new_sizes);
    graph.resize_input(2, new_sizes);
    graph.propagate_resize();

    // Check output shape
    EXPECT_TRUE(graph.get_val(out.value).toTensor().sizes() == new_sizes);

    float val_a = new_sizes[1] + 6.0f;
    float val_b = new_sizes[2] + 2.5f;
    float val_d = new_sizes[0] + 4.0f;
    float val_out = (val_a + val_b) * val_d;

    fill_vtensor(graph, a, val_a);
    fill_vtensor(graph, b, val_b);
    fill_vtensor(graph, d, val_d);

    // Execute graph
    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    int i = 0;
    for (const auto& val : data_out) {
      ASSERT_TRUE(val == val_out);
      ++i;
    }
  }
}
