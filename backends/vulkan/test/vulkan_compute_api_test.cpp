/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/test/utils/test_utils.h>

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

  ASSERT_TRUE(kernel.kernel_name == "test_shader");
}

TEST_F(VulkanComputeAPITest, update_params_between_submit) {
  api::context()->set_cmd(/*reusable = */ true);
  std::vector<int64_t> sizes = {4, 4, 2};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  std::stringstream kernel_name;
  kernel_name << "fill_texture__test";
  apply_dtype_suffix(kernel_name, a);

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
        VK_KERNEL_FROM_STR(kernel_name.str()),
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
  record_binary_op(api::context(), "add", a, b, c);

  // Extract output tensor
  std::vector<float> data_out = extract_vtensor(c);

  // Check output
  for (size_t i = 0; i < data_out.size(); ++i) {
    CHECK_VALUE(data_out, i, 4.0f);
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

  record_binary_op(api::context(), "add", a, b, c);

  std::vector<float> data_c(c.gpu_numel());
  extract_vtensor(c, data_c);

  for (size_t i = 0; i < data_c.size(); ++i) {
    CHECK_VALUE(data_c, i, 4.0f);
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

  // First, fill a and b with data
  fill_vtensor(a, data_a);
  fill_vtensor(b, data_b);

  // a + b -> c
  record_binary_op(api::context(), "add", a, b, c);

  // Now d can be filled with data
  fill_vtensor(d, data_d);

  // c + d -> e
  record_binary_op(api::context(), "add", c, d, e);

  // Extract data from e
  std::vector<float> data_e(e.gpu_numel());
  extract_vtensor(e, data_e);

  // Sanity check that the values are correct
  for (size_t i = 0; i < data_e.size(); ++i) {
    CHECK_VALUE(data_e, i, 5.0f);
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

  record_binary_op(api::context(), "add", a, b, c);

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
  const auto& arr = graph.get_val(idx).toIntList();
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
  const auto& arr = graph.get_val(idx).toDoubleList();
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
    for (size_t i = 0; i < graph.get_val(out.value).toTensor().numel(); ++i) {
      CHECK_VALUE(data_out, i, val_c);
    }
  }
}

#define CREATE_WEIGHT_TENSOR(name, sizes, dtype, val)                   \
  std::vector<float> data_##name(api::utils::multiply_integers(sizes)); \
  std::fill(data_##name.begin(), data_##name.end(), val);               \
  ValueRef name = graph.add_tensorref(sizes, dtype, data_##name.data());

TEST(VulkanComputeGraphTest, test_simple_prepacked_graph) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {8, 73, 62};
  std::vector<int64_t> size_small = {8, 73, 1};

  CREATE_WEIGHT_TENSOR(w1, size_small, api::kFloat, 3.5f);
  CREATE_WEIGHT_TENSOR(w2, size_small, api::kFloat, 3.0f);

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
    for (size_t i = 0; i < graph.get_val(out.value).toTensor().numel(); ++i) {
      CHECK_VALUE(data_out, i, val_out);
    }
  }
}

TEST(VulkanComputeGraphTest, test_simple_shared_objects_with_resize) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {12, 64, 64};
  std::vector<int64_t> size_small = {12, 64, 64};

  // Build graph and regularly check allocation counts

  IOValueRef a = graph.add_input_tensor(
      size_big,
      api::kFloat,
      /*shared_object_idx = */ 2);
  IOValueRef b = graph.add_input_tensor(
      size_small,
      api::kFloat,
      /*shared_object_idx = */ 4);

  // +4: t.gpu_sizes_ubo(), t.cpu_sizes_ubo() for each staging shader
  // +2: staging buffer for each input tensor
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

  // +3: out.gpu_sizes_ubo(), alpha UBO, broadcast UBO for arithmetic shader
  // +2: t.gpu_sizes_ubo(), t.cpu_sizes_ubo() uniform buffer for staging shader
  // +1: staging buffer for the input tensor
  EXPECT_TRUE(get_vma_allocation_count() == 12);

  ValueRef e = graph.add_tensor(
      size_big,
      api::kFloat,
      /*shared_object_idx = */ 4);

  auto mulFn = VK_GET_OP_FN("aten.mul.Tensor");
  mulFn(graph, {c, d.value, e});

  IOValueRef out = {};
  out.value = e;
  out.staging = graph.set_output_tensor(out.value);

  // +2: alpha UBO, broadcast UBO for arithmetic shader
  // +2: t.gpu_sizes_ubo(), t.cpu_sizes_ubo() for staging shader
  // +1 staging buffer for the input tensor
  EXPECT_TRUE(get_vma_allocation_count() == 17);

  graph.prepare();
  graph.encode_execute();

  // +3: shared memory allocations for tensors
  EXPECT_TRUE(get_vma_allocation_count() == 20);

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
    for (size_t i = 0; i < graph.get_val(out.value).toTensor().numel(); i++) {
      CHECK_VALUE(data_out, i, val_out);
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
    for (size_t i = 0; i < graph.get_val(out.value).toTensor().numel(); i++) {
      CHECK_VALUE(data_out, i, val_out);
    }
  }
}

TEST(VulkanComputeGraphTest, test_large_graph) {
  GraphConfig config;
  ComputeGraph graph(config);

  int64_t input_w = 256;
  int64_t input_h = 256;
  int64_t input_c = 8;

  std::vector<int64_t> size_big = {input_c, input_h, input_w};
  std::vector<int64_t> size_small = {input_c, input_h, 1};

  // Build graph

  IOValueRef a = graph.add_input_tensor(size_big, api::kFloat, 2);
  IOValueRef b = graph.add_input_tensor(size_small, api::kFloat, 4);

  ValueRef c = graph.add_tensor(size_big, api::kFloat, 6);

  auto addFn = VK_GET_OP_FN("aten.add.Tensor");
  addFn(graph, {a.value, b.value, kDummyValueRef, c});

  int n = 100;

  for (int i = 0; i < n; i++) {
    addFn(graph, {c, b.value, kDummyValueRef, a.value});

    addFn(graph, {a.value, b.value, kDummyValueRef, c});
  }

  IOValueRef out = {};
  out.value = c;
  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_execute();

  for (int i = 0; i < 10; i++) {
    float val_a = 1.0f;
    float val_b = 2.0f;

    float val_e = val_a + val_b * (2 * n + 1);

    fill_vtensor(graph, a, val_a);
    fill_vtensor(graph, b, val_b);

    graph.execute();

    EXTRACT_TENSOR(out);

    for (int i = 0; i < graph.get_val(out.value).toTensor().numel(); i++) {
      CHECK_VALUE(data_out, i, val_e);
    }
  }
}

class VulkanToFromGPUShaderTest : public ::testing::Test {
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

template <typename T>
void run_from_gpu_test(
    std::vector<int64_t>& sizes,
    api::GPUMemoryLayout memory_layout =
        api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
    api::ScalarType dtype = api::kFloat,
    api::StorageType storage_type = api::StorageType::TEXTURE_3D) {
  vTensor vten =
      vTensor(api::context(), sizes, api::kFloat, storage_type, memory_layout);

  std::stringstream kernel_name;
  kernel_name << "idx_fill_texture";
  apply_memory_layout_suffix(kernel_name, vten);
  apply_dtype_suffix(kernel_name, vten);

  {
    api::PipelineBarrier pipeline_barrier{};
    api::context()->submit_compute_job(
        VK_KERNEL_FROM_STR(kernel_name.str()),
        pipeline_barrier,
        vten.virtual_extents(),
        {4, 4, 4},
        VK_NULL_HANDLE,
        vten.image(
            pipeline_barrier,
            api::PipelineStage::COMPUTE,
            api::MemoryAccessType::WRITE),
        vten.gpu_sizes_ubo()->buffer(),
        vten.cpu_sizes_ubo()->buffer());
  }

  api::StorageBuffer staging_buffer(
      api::context(), api::kFloat, vten.gpu_numel());

  record_image_to_nchw_op(api::context(), vten, staging_buffer.buffer());

  submit_to_gpu();

  std::vector<T> data_out(staging_buffer.numel());
  copy_staging_to_ptr(
      staging_buffer, data_out.data(), sizeof(float) * staging_buffer.numel());

  for (int i = 0; i < vten.numel(); i++) {
    CHECK_VALUE(data_out, i, i);
  }
}

template <typename T>
void run_to_gpu_test(
    std::vector<int64_t>& sizes,
    api::GPUMemoryLayout memory_layout =
        api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
    api::ScalarType dtype = api::kFloat,
    api::StorageType storage_type = api::StorageType::TEXTURE_3D) {
  vTensor vten =
      vTensor(api::context(), sizes, api::kFloat, storage_type, memory_layout);

  // Create and fill input staging buffer
  api::StorageBuffer staging_buffer_in(
      api::context(), api::kFloat, vten.gpu_numel());

  std::vector<T> data_in(staging_buffer_in.numel());
  for (int i = 0; i < staging_buffer_in.numel(); i++) {
    data_in[i] = i;
  }
  copy_ptr_to_staging(data_in.data(), staging_buffer_in, vten.gpu_nbytes());

  // Output staging buffer
  api::StorageBuffer staging_buffer_out(
      api::context(), api::kFloat, vten.gpu_numel());

  // Copy data in and out of the tensor
  record_nchw_to_image_op(api::context(), staging_buffer_in.buffer(), vten);
  record_image_to_nchw_op(api::context(), vten, staging_buffer_out.buffer());

  // Execute command buffer
  submit_to_gpu();

  // Extract data from output staging buffer
  std::vector<T> data_out(staging_buffer_out.numel());
  copy_staging_to_ptr(
      staging_buffer_out,
      data_out.data(),
      sizeof(float) * staging_buffer_out.numel());

  // All indices should be equal to the input data
  for (int i = 0; i < vten.numel(); i++) {
    CHECK_VALUE(data_out, i, i);
  }
}

TEST(VulkanToFromGPUShaderTest, to_gpu_and_from_gpu_test_texture) {
  // The below tests will fill each texel element with the value of the linear
  // buffer index that corresponds to it. The texel at position (0, 0, 0) will
  // be filled with the values [0, 1, 2, 3], the texel at position (1, 0, 0)
  // will be filled with the values [4, 5, 6, 7], and so forth. The contents of
  // the texture are then written back to the CPU, and to check that the
  // transfer has ben performed correctly the value at each index of the CPU
  // data buffer should be equal to the index.
  //
  // The below test cases should ensure that the total number of elements does
  // not exceed 2048, or else the tests will fail for FP16 textures due to
  // precision issues. Half precision floating point formats can only represent
  // integers from 2048 to 4096 using intervals of 2.
  std::vector<std::vector<int64_t>> to_test = {
      // 2D sizes
      {17, 21},
      {67, 23},
      {55, 33},
      // 3D sizes
      {7, 9, 13},
      {21, 2, 19},
      {17, 17, 5},
      // 4D sizes
      {7, 3, 13, 7},
      {11, 9, 9, 1},
      {3, 3, 3, 3},
      {3, 1, 7, 13},
  };

#define RUN_TESTS(ctype, dtype)                                    \
  run_from_gpu_test<ctype>(                                        \
      sizes, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, dtype); \
  run_from_gpu_test<ctype>(                                        \
      sizes, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED, dtype);    \
  run_from_gpu_test<ctype>(                                        \
      sizes, api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED, dtype);   \
  run_to_gpu_test<ctype>(                                          \
      sizes, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, dtype); \
  run_to_gpu_test<ctype>(                                          \
      sizes, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED, dtype);    \
  run_to_gpu_test<ctype>(                                          \
      sizes, api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED, dtype);

  for (auto& sizes : to_test) {
    RUN_TESTS(float, api::kFloat)
    RUN_TESTS(float, api::kHalf)
  }
#undef RUN_TESTS
}

//
// Operator Smoke Tests
//

void test_binary_op(
    std::string op_name,
    std::vector<int64_t> sizes_big,
    std::vector<int64_t> sizes_small,
    api::ScalarType dtype,
    api::GPUMemoryLayout memory_layout,
    bool prepack = true) {
  GraphConfig config;
  ComputeGraph graph(config);

  IOValueRef arg2{};

  CREATE_WEIGHT_TENSOR(arg2_w, sizes_small, dtype, 2.5f);

  // Build graph

  IOValueRef arg1 = graph.add_input_tensor(sizes_big, dtype, memory_layout);

  if (prepack) {
    arg2.value = arg2_w;
  } else {
    arg2 = graph.add_input_tensor(sizes_small, dtype, memory_layout);
  }

  IOValueRef out;
  out.value = graph.add_tensor(sizes_big, dtype, memory_layout);

  std::stringstream ss;
  ss << "aten.";
  ss << op_name;
  ss << ".Tensor";
  VK_GET_OP_FN(ss.str())
  (graph, {arg1.value, arg2.value, kDummyValueRef, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_prepack();
  graph.prepack();
  graph.encode_execute();

  for (int i = 1; i < 4; i++) {
    float val_arg1 = i + 1.5;
    float val_arg2 = prepack ? 2.5f : i - 3.5;

    float val_out = val_arg1 + val_arg2;
    if (op_name == "sub") {
      val_out = val_arg1 - val_arg2;
    }
    if (op_name == "mul") {
      val_out = val_arg1 * val_arg2;
    }
    if (op_name == "div") {
      val_out = val_arg1 / val_arg2;
    }

    if (prepack) {
      execute_graph_and_check_output(graph, {val_arg1}, {val_out});
    } else {
      execute_graph_and_check_output(graph, {val_arg1, val_arg2}, {val_out});
    }
  }
}

#define CALL_TEST_FN_FORALL_CONDITIONS(_)                             \
  _(api::kFloat, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED, false)    \
  _(api::kFloat, api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED, false)   \
  _(api::kFloat, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, false) \
  _(api::kFloat, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED, true)     \
  _(api::kFloat, api::GPUMemoryLayout::TENSOR_HEIGHT_PACKED, true)    \
  _(api::kFloat, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, true)

#define CALL_TEST_FN_FOR_W_PACKED(_)                               \
  _(api::kFloat, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED, false) \
  _(api::kFloat, api::GPUMemoryLayout::TENSOR_WIDTH_PACKED, true)

#define CALL_TEST_FN_FOR_C_PACKED(_)                                  \
  _(api::kFloat, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, false) \
  _(api::kFloat, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, true)

TEST(VulkanComputeGraphOpsTest, add_smoke_test) {
#define RUN_TESTS(dtype, layout, prepack)                                  \
  test_binary_op("add", {17, 21}, {17, 21}, dtype, layout, prepack);       \
  test_binary_op("add", {17, 21}, {1, 1}, dtype, layout, prepack);         \
  test_binary_op("sub", {11, 22}, {11, 22}, dtype, layout, prepack);       \
  test_binary_op("sub", {11, 22}, {11, 1}, dtype, layout, prepack);        \
  test_binary_op("add", {7, 17, 17}, {7, 17, 17}, dtype, layout, prepack); \
  test_binary_op("add", {7, 17, 17}, {7, 1, 17}, dtype, layout, prepack);  \
  test_binary_op("sub", {9, 9, 7}, {9, 9, 7}, dtype, layout, prepack);     \
  test_binary_op("sub", {9, 9, 7}, {9, 1, 1}, dtype, layout, prepack);

  CALL_TEST_FN_FORALL_CONDITIONS(RUN_TESTS);

#undef RUN_TESTS
}

void test_mm(
    int B,
    int M,
    int K,
    int N,
    api::ScalarType dtype,
    api::GPUMemoryLayout memory_layout,
    bool prepack = true) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> mat1_size = {M, K};
  std::vector<int64_t> mat2_size = {K, N};
  std::vector<int64_t> out_size = {M, N};
  if (B > 1) {
    mat1_size.resize(3);
    mat1_size = {B, M, K};
    mat2_size.resize(3);
    mat2_size = {B, K, N};
    out_size.resize(3);
    out_size = {B, M, N};
  }

  IOValueRef mat2{};

  CREATE_WEIGHT_TENSOR(mat2_w, mat2_size, dtype, 2.0f);

  // Build graph

  IOValueRef mat1 = graph.add_input_tensor(mat1_size, dtype, memory_layout);

  if (prepack) {
    mat2.value = mat2_w;
  } else {
    mat2.value = graph.add_tensor(mat2_size, dtype, memory_layout);
    mat2.staging = graph.set_input_tensor(mat2.value);
  }

  IOValueRef out;
  out.value = graph.add_tensor(out_size, dtype, memory_layout);

  VK_GET_OP_FN("aten.mm.default")(graph, {mat1.value, mat2.value, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_prepack();
  graph.prepack();
  graph.encode_execute();

  for (int i = 1; i < 4; i++) {
    if (prepack) {
      float val_mat1 = i;
      float val_out = K * (val_mat1 * 2.0f);
      execute_graph_and_check_output(graph, {val_mat1}, {val_out});
    } else {
      float val_mat1 = i;
      float val_mat2 = i + 1;
      float val_out = K * (val_mat1 * val_mat2);
      execute_graph_and_check_output(graph, {val_mat1, val_mat2}, {val_out});
    }
  }
}

TEST(VulkanComputeGraphOpsTest, mm_smoke_test) {
#define RUN_TESTS(dtype, layout, prepack) \
  test_mm(                                \
      /*B = */ 1,                         \
      /*M = */ 31,                        \
      /*K = */ 127,                       \
      /*N = */ 23,                        \
      dtype,                              \
      layout,                             \
      prepack);                           \
  test_mm(                                \
      /*B = */ 5,                         \
      /*M = */ 31,                        \
      /*K = */ 127,                       \
      /*N = */ 23,                        \
      dtype,                              \
      layout,                             \
      prepack);                           \
  test_mm(                                \
      /*B = */ 7,                         \
      /*M = */ 13,                        \
      /*K = */ 89,                        \
      /*N = */ 17,                        \
      dtype,                              \
      layout,                             \
      prepack);                           \
  test_mm(                                \
      /*B = */ 1,                         \
      /*M = */ 13,                        \
      /*K = */ 89,                        \
      /*N = */ 17,                        \
      dtype,                              \
      layout,                             \
      prepack);

  CALL_TEST_FN_FOR_W_PACKED(RUN_TESTS);
  CALL_TEST_FN_FOR_C_PACKED(RUN_TESTS);

#undef RUN_TESTS
}

void test_max_pool2d(
    const std::vector<int64_t>& in_size,
    const int64_t base_val,
    std::vector<int64_t>& kernel) {
  GraphConfig config;
  ComputeGraph graph(config);

  // Build graph

  std::vector<int64_t> out_size(in_size);
  int h = in_size.size() - 2;
  int w = in_size.size() - 1;
  out_size[h] = in_size[h] - kernel[0] + 1;
  out_size[w] = in_size[w] - kernel[1] + 1;

  IOValueRef in_ioval = graph.add_input_tensor(
      in_size, api::kFloat, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  IOValueRef out_ioval;
  out_ioval.value = graph.add_tensor(
      out_size, api::kFloat, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  IOValueRef idx_ioval;
  idx_ioval.value = graph.add_tensor(
      out_size, api::kInt, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  ValueRef out = graph.add_value_list({out_ioval.value, idx_ioval.value});

  std::vector<int64_t> kernel_copy(kernel);
  VK_GET_OP_FN("aten.max_pool2d_with_indices.default")
  (graph,
   {in_ioval.value,
    graph.add_scalar_list<int64_t>(std::move(kernel)),
    graph.add_scalar_list<int64_t>({1, 1}),
    graph.add_scalar_list<int64_t>({0, 0}),
    graph.add_scalar_list<int64_t>({1, 1}),
    graph.add_scalar(false),
    out});

  out_ioval.staging = graph.set_output_tensor(out_ioval.value);
  idx_ioval.staging = graph.set_output_tensor(idx_ioval.value);

  graph.prepare();
  graph.encode_prepack();
  graph.prepack();
  graph.encode_execute();

  // Run graph

  fill_vtensor(graph, graph.inputs().at(0), base_val, /*iota = */ true);

  vTensor& t_in = graph.get_val(in_ioval.value).toTensor();
  std::vector<float> input_data(t_in.gpu_numel());
  graph.copy_from_staging(
      in_ioval.staging, input_data.data(), input_data.size());

  graph.execute();

  vTensor& t_out = graph.get_val(out_ioval.value).toTensor();
  std::vector<float> output_data(t_out.gpu_numel());
  graph.copy_from_staging(
      out_ioval.staging, output_data.data(), output_data.size());
  vTensor& t_idx = graph.get_val(idx_ioval.value).toTensor();
  std::vector<int> index_data(t_idx.gpu_numel());
  graph.copy_from_staging(
      idx_ioval.staging, index_data.data(), index_data.size());

  // Check results

  int h_offset = kernel_copy[0] - 1;
  int w_offset = kernel_copy[1] - 1;
  int h_out = api::utils::val_at(-2, t_out.sizes());
  int w_out = api::utils::val_at(-1, t_out.sizes());
  int w_in = api::utils::val_at(-1, t_in.sizes());
  for (size_t i = 0; i < h_out; ++i) {
    for (size_t j = 0; j < w_out; ++j) {
      size_t idx_out = i * w_out + j;
      size_t idx_in = (i + h_offset) * w_in + (j + w_offset);
      CHECK_VALUE(index_data, idx_out, idx_in);
      CHECK_VALUE(output_data, idx_out, input_data[idx_in]);
    }
  }
}

TEST(VulkanComputeGraphOpsTest, max_pool2d_smoke_test) {
  std::vector<int64_t> kernel = {2, 3};
  test_max_pool2d(
      /*in_size = */ {1, 4, 6},
      /*base_val = */ 10.0f,
      kernel);
}

void test_conv2d(
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& padded_sizes,
    const std::vector<int64_t>& gpu_sizes,
    const bool transposed,
    const std::vector<float>& data_out_expected) {
  vTensor vten = vTensor(
      api::context(),
      gpu_sizes,
      api::kFloat,
      api::StorageType::TEXTURE_2D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);

  // Create and fill input staging buffer
  const int64_t in_numel = api::utils::multiply_integers(original_sizes);
  api::StorageBuffer staging_buffer_in(api::context(), api::kFloat, in_numel);

  std::vector<float> data_in(in_numel);
  for (int i = 0; i < in_numel; i++) {
    data_in[i] = i + 1;
  }
  copy_ptr_to_staging(
      data_in.data(), staging_buffer_in, sizeof(float) * in_numel);

  // Output staging buffer
  const int64_t out_numel =
      padded_sizes[0] * padded_sizes[1] * original_sizes[2] * original_sizes[3];
  api::StorageBuffer staging_buffer_out(api::context(), api::kFloat, out_numel);

  // Copy data in and out of the tensor
  record_conv2d_prepack_weights_op(
      api::context(),
      staging_buffer_in.buffer(),
      vten,
      original_sizes,
      padded_sizes,
      transposed);
  record_image_to_nchw_op(api::context(), vten, staging_buffer_out.buffer());

  // Execute command buffer
  submit_to_gpu();

  // Extract data from output staging buffer
  std::vector<float> data_out(out_numel);
  copy_staging_to_ptr(
      staging_buffer_out, data_out.data(), sizeof(float) * out_numel);

  // Check data matches results copied from ATen-VK
  for (int i = 0; i < vten.numel(); i++) {
    CHECK_VALUE(data_out, i, data_out_expected[i]);
  }
}

TEST(VulkanComputeGraphOpsTest, conv2d_prepack_test) {
  test_conv2d(
      /*original_sizes = */ {2, 3, 1, 2},
      /*padded_sizes = */ {4, 4},
      /*gpu_sizes = */ {4, 1, 8},
      /*transposed = */ false,
      /*data_out_expected = */ {1, 3, 5,  0,  2, 4, 6, 0, 7, 9, 11,
                                0, 8, 10, 12, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0,  0,  0, 0, 0, 0, 0, 0});
  test_conv2d(
      /*original_sizes = */ {2, 3, 1, 2},
      /*padded_sizes = */ {4, 4},
      /*gpu_sizes = */ {4, 1, 8},
      /*transposed = */ true,
      /*data_out_expected = */ {2, 8, 0, 0, 1, 7, 0,  0, 4, 10, 0,
                                0, 3, 9, 0, 0, 6, 12, 0, 0, 5,  11,
                                0, 0, 0, 0, 0, 0, 0,  0, 0, 0});
}
