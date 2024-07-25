/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include <executorch/runtime/core/portable_type/half.h>

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/test/utils/test_utils.h>

using namespace vkcompute::api;

std::vector<std::vector<int64_t>> standard_sizes_to_test = {
    // 2D
    {7, 11},
    {13, 6},
    // 3D
    {2, 9, 7},
    {9, 15, 19},
    {7, 11, 24},
    {13, 8, 11},
    {12, 11, 19},
    // 4D
    {2, 2, 3, 5},
    {9, 13, 11, 17},
    {17, 14, 18, 20},
    {7, 13, 12, 21},
    {3, 8, 13, 17},
};

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
    context()->flush();

    // Make sure we are ending with a clean slate
    EXPECT_TRUE(get_vma_allocation_count() == 0);
  }
};

TEST_F(VulkanComputeAPITest, print_adapter) {
  std::cout << *(context()->adapter_ptr()) << std::endl;
}

std::vector<int64_t> get_reference_strides(
    const std::vector<int64_t>& sizes,
    const utils::GPUMemoryLayout layout,
    const bool texel_strides) {
  int64_t C = utils::val_at(-3, sizes);
  int64_t H = utils::val_at(-2, sizes);
  int64_t W = utils::val_at(-1, sizes);

  switch (layout) {
    case utils::kWidthPacked:
      if (texel_strides) {
        W = utils::div_up(W, INT64_C(4));
      }
      switch (sizes.size()) {
        case 1:
          return {1};
        case 2:
          return {W, 1};
        case 3:
          return {H * W, W, 1};
        case 4:
          return {C * H * W, H * W, W, 1};
        default:
          return {};
      }
      break;
    case utils::kHeightPacked:
      if (texel_strides) {
        H = utils::div_up(H, INT64_C(4));
      }
      switch (sizes.size()) {
        case 1:
          return {1};
        case 2:
          return {1, H};
        case 3:
          return {W * H, 1, H};
        case 4:
          return {C * W * H, W * H, 1, H};
        default:
          return {};
      }
    case utils::kChannelsPacked:
      if (texel_strides) {
        C = utils::div_up(C, INT64_C(4));
      }
      switch (sizes.size()) {
        case 1:
          return {1};
        case 2:
          return {W, 1};
        case 3:
          return {1, W * C, C};
        case 4:
          return {H * W * C, 1, W * C, C};
        default:
          return {};
      }
  }
  return {};
}

TEST_F(VulkanComputeAPITest, calculate_tensor_strides_test) {
  for (const auto& sizes : standard_sizes_to_test) {
    if (sizes.size() < 3) {
      continue;
    }
    for (const auto& layout :
         {utils::kWidthPacked, utils::kHeightPacked, utils::kChannelsPacked}) {
      // texel_strides = true
      {
        std::vector<int64_t> strides = calculate_strides(sizes, layout);
        std::vector<int64_t> ref_strides =
            get_reference_strides(sizes, layout, true);

        ASSERT_TRUE(strides == ref_strides);
      }

      // texel_strides = false
      {
        std::vector<int64_t> strides = calculate_strides(sizes, layout, false);
        std::vector<int64_t> ref_strides =
            get_reference_strides(sizes, layout, false);
        ASSERT_TRUE(strides == ref_strides);
      }
    }
  }
}

TEST_F(VulkanComputeAPITest, retrieve_custom_shader_test) {
  // Try to get shader from custom shader library
  const vkapi::ShaderInfo& kernel = VK_KERNEL(test_shader);

  ASSERT_TRUE(kernel.kernel_name == "test_shader");
}

TEST_F(VulkanComputeAPITest, spec_var_classes_test) {
  // Check equality operator
  ASSERT_TRUE(SV(1.5f) == SV(1.5f));
  ASSERT_FALSE(SV(15.0f) == SV(15));
  ASSERT_FALSE(SV(1u) == SV(true));

  size_t sv_size = sizeof(vkapi::SpecVar);

  vkapi::SpecVarList spec_vars = {};
  ASSERT_TRUE(spec_vars.size() == 0);
  spec_vars = {SV(1.1f), SV(32), SV(45)};
  ASSERT_TRUE(spec_vars.size() == 3);
  vkapi::SpecVarList spec_vars_other = {SV(2.6f), SV(true), SV(78u), SV(5.5f)};
  spec_vars.append(spec_vars_other);
  ASSERT_TRUE(spec_vars.size() == 7);

  // Check validity of the data
  const vkapi::SpecVar* data = spec_vars.data();
  ASSERT_TRUE(*(reinterpret_cast<const float*>(data + 3)) == 2.6f);
  ASSERT_TRUE(*(reinterpret_cast<const int32_t*>(data + 1)) == 32);
  ASSERT_TRUE(*(reinterpret_cast<const int32_t*>(data + 5)) == 78u);

  // Check validity of the map entries
  std::vector<VkSpecializationMapEntry> entries =
      spec_vars.generate_map_entries();

  for (size_t i = 0; i < spec_vars.size(); ++i) {
    ASSERT_TRUE(entries[i].constantID == i);
    ASSERT_TRUE(entries[i].offset == sv_size * i);
    if (i != 4) {
      ASSERT_TRUE(entries[i].size == 4);
    } else {
      ASSERT_TRUE(entries[i].size == 1);
    }
  }

  // Check copy
  vkapi::SpecVarList spec_vars_copy(spec_vars);
  ASSERT_TRUE(spec_vars_copy.size() == 7);

  // Check validity of the copied data
  const vkapi::SpecVar* copy_data = spec_vars_copy.data();
  ASSERT_TRUE(*(reinterpret_cast<const bool*>(copy_data + 4)) == true);
  ASSERT_TRUE(*(reinterpret_cast<const int32_t*>(copy_data + 2)) == 45);
  ASSERT_TRUE(*(reinterpret_cast<const float*>(copy_data + 6)) == 5.5f);
}

TEST_F(VulkanComputeAPITest, spec_var_shader_test) {
  size_t len = 16;
  StorageBuffer buffer(context(), vkapi::kFloat, len);

  float scale = 3.0f;
  float offset = 1.5f;

  {
    ParamsBuffer params(context(), int32_t(len));
    uint32_t len_div4 = utils::div_up(uint32_t(len), uint32_t(4));
    vkapi::PipelineBarrier pipeline_barrier{};
    context()->submit_compute_job(
        VK_KERNEL(fill_buffer),
        pipeline_barrier,
        {64, 1, 1},
        {len_div4, 1, 1},
        {SV(scale), SV(offset)},
        VK_NULL_HANDLE,
        0,
        buffer.buffer(),
        params.buffer());
  }

  submit_to_gpu();

  std::vector<float> data(len);
  copy_staging_to_ptr(buffer, data.data(), buffer.nbytes());

  for (size_t i = 0; i < len; ++i) {
    CHECK_VALUE(data, i, scale * i + offset);
  }
}

TEST_F(VulkanComputeAPITest, update_params_between_submit) {
  context()->set_cmd(/*reusable = */ true);
  std::vector<int64_t> sizes = {4, 4, 2};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  std::string kernel_name("fill_texture__test");
  add_dtype_suffix(kernel_name, a);

  struct Params final {
    utils::ivec3 size;
    int32_t fill;
    utils::vec4 values;
  };

  Params block{
      {2, 4, 1},
      0,
      {5.0, 5.0, 5.0, 5.0},
  };

  ParamsBuffer params(context(), block);

  {
    vkapi::PipelineBarrier pipeline_barrier{};
    vkapi::SpecVarList specialization_constants = {};
    context()->submit_compute_job(
        VK_KERNEL_FROM_STR(kernel_name),
        pipeline_barrier,
        {4, 4, 4},
        {4, 4, 4},
        specialization_constants,
        VK_NULL_HANDLE,
        0,
        a.image(
            pipeline_barrier,
            vkapi::PipelineStage::COMPUTE,
            vkapi::MemoryAccessType::WRITE),
        params.buffer());
  }

  StorageBuffer staging_buffer(context(), vkapi::kFloat, a.gpu_numel());
  record_image_to_nchw_op(context(), a, staging_buffer.buffer());

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

template <typename T, vkapi::ScalarType dtype>
void test_storage_buffer_type(const size_t len) {
  StorageBuffer buffer(context(), dtype, len);

  std::string kernel_name("idx_fill_buffer");
  switch (dtype) {
    case vkapi::kFloat:
      kernel_name += "_float";
      break;
    case vkapi::kHalf:
      kernel_name += "_half";
      break;
    case vkapi::kQInt8:
      kernel_name += "_int8";
      break;
    case vkapi::kQUInt8:
      kernel_name += "_uint8";
      break;
    default:
      throw std::runtime_error("Unsupported dtype");
      break;
  }

  ParamsBuffer params(context(), int32_t(len));

  {
    uint32_t len_div4 = utils::div_up(uint32_t(len), uint32_t(4));
    vkapi::PipelineBarrier pipeline_barrier{};
    vkapi::SpecVarList specialization_constants = {};
    context()->submit_compute_job(
        VK_KERNEL_FROM_STR(kernel_name),
        pipeline_barrier,
        {64, 1, 1},
        {len_div4, 1, 1},
        specialization_constants,
        VK_NULL_HANDLE,
        0,
        buffer.buffer(),
        params.buffer());
  }

  submit_to_gpu();

  std::vector<T> data(len);
  copy_staging_to_ptr(buffer, data.data(), buffer.nbytes());

  for (size_t i = 0; i < len; ++i) {
    CHECK_VALUE(data, i, T(i));
  }
}

TEST_F(VulkanComputeAPITest, test_buffer_float) {
  test_storage_buffer_type<float, vkapi::kFloat>(16);
}

TEST_F(VulkanComputeAPITest, test_buffer_float16) {
  if (!context()->adapter_ptr()->has_full_float16_buffers_support()) {
    GTEST_SKIP();
  }
  test_storage_buffer_type<torch::executor::Half, vkapi::kHalf>(16);
}

TEST_F(VulkanComputeAPITest, test_buffer_int8) {
  if (!context()->adapter_ptr()->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  test_storage_buffer_type<int8_t, vkapi::kQInt8>(16);
}

TEST_F(VulkanComputeAPITest, test_zero_size_tensor) {
  // Simple test that performs a + b -> c

  std::vector<int64_t> sizes = {0, 5, 7};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  // Fill input tensors
  fill_vtensor(a, 2.5f);
  fill_vtensor(b, 1.5f);

  // a + b -> c
  record_binary_op(context(), "add", a, b, c);

  // Extract output tensor
  std::vector<float> data_out = extract_vtensor(c);

  // Assert all tensors are empty
  ASSERT_TRUE(a.numel() == 0);
  ASSERT_TRUE(b.numel() == 0);
  ASSERT_TRUE(c.numel() == 0);
  ASSERT_TRUE(a.nbytes() == 0);
  ASSERT_TRUE(b.nbytes() == 0);
  ASSERT_TRUE(c.nbytes() == 0);

  // Check output
  for (size_t i = 0; i < data_out.size(); ++i) {
    CHECK_VALUE(data_out, i, 4.0f);
  }
}

template <typename T>
void run_buffer_tensor_sanity_check(vTensor& tensor) {
  fill_vtensor(tensor, 0.0f, true);

  record_scalar_add_buffer(context(), tensor, 2.0f);
  std::vector<float> data_out = extract_vtensor(tensor);

  // Check output
  for (size_t i = 0; i < tensor.numel(); ++i) {
    CHECK_VALUE(data_out, i, i + 2.0f);
  }
}

TEST_F(VulkanComputeAPITest, buffer_tensor_sanity_check) {
  for (const auto& sizes : standard_sizes_to_test) {
    for (const auto& dtype : {vkapi::kFloat, vkapi::kHalf, vkapi::kChar}) {
      if (dtype == vkapi::kHalf &&
          !context()->adapter_ptr()->has_full_float16_buffers_support()) {
        continue;
      }
      if (dtype == vkapi::kHalf && utils::multiply_integers(sizes) >= 2048) {
        continue;
      }
      if (dtype == vkapi::kChar &&
          !context()->adapter_ptr()->has_full_int8_buffers_support()) {
        continue;
      }
      if (dtype == vkapi::kChar && utils::multiply_integers(sizes) >= 128) {
        continue;
      }
      for (const auto& layout :
           {utils::kWidthPacked,
            utils::kHeightPacked,
            utils::kChannelsPacked}) {
        vTensor a = vTensor(context(), sizes, dtype, utils::kBuffer, layout);
        switch (dtype) {
          case vkapi::kFloat:
            run_buffer_tensor_sanity_check<float>(a);
            break;
          case vkapi::kHalf:
            run_buffer_tensor_sanity_check<torch::executor::Half>(a);
            break;
          case vkapi::kChar:
            run_buffer_tensor_sanity_check<int8_t>(a);
            break;
          default:
            VK_THROW("Unsupported dtype");
        }
      }
    }
  }
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
  record_binary_op(context(), "add", a, b, c);

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

  // No allocations made so far
  EXPECT_TRUE(get_vma_allocation_count() == 0);

  std::vector<float> data_a(a.gpu_numel());
  std::fill(data_a.begin(), data_a.end(), 2.5f);
  std::vector<float> data_b(b.gpu_numel());
  std::fill(data_b.begin(), data_b.end(), 1.5f);

  // Allocate memory at the last possible opportunity
  vkapi::Allocation a_mem = allocate_memory_for(a);
  a.image().bind_allocation(a_mem);
  vkapi::Allocation b_mem = allocate_memory_for(b);
  b.image().bind_allocation(b_mem);
  vkapi::Allocation c_mem = allocate_memory_for(c);
  c.image().bind_allocation(c_mem);

  // One allocation for each tensor
  EXPECT_TRUE(get_vma_allocation_count() == 3);

  fill_vtensor(a, data_a);
  fill_vtensor(b, data_b);

  record_binary_op(context(), "add", a, b, c);

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

  // No allocations made so far
  EXPECT_TRUE(get_vma_allocation_count() == 0);

  // a and d can share the same memory allocation
  vkapi::Allocation a_d_mem = allocate_memory_for(a);
  a.image().bind_allocation(a_d_mem);
  d.image().bind_allocation(a_d_mem);
  // b and e can share the same memory allocation
  vkapi::Allocation b_e_mem = allocate_memory_for(b);
  b.image().bind_allocation(b_e_mem);
  e.image().bind_allocation(b_e_mem);
  // c must have its own memory allocation
  vkapi::Allocation c_mem = allocate_memory_for(c);
  c.image().bind_allocation(c_mem);

  // 3 allocations should be made
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
  record_binary_op(context(), "add", a, b, c);

  // Now d can be filled with data
  fill_vtensor(d, data_d);

  // c + d -> e
  record_binary_op(context(), "add", c, d, e);

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
  vkapi::Allocation a_mem = allocate_memory_for(a);
  EXPECT_THROW(a.image().bind_allocation(a_mem), vkapi::Error);
}

TEST_F(VulkanComputeAPITest, resource_destructor_non_owning_memory) {
  // Check that the destructor of a vTensor that does not own its memory
  // does not free the memory

  vkapi::Allocation memory;

  // Default Allocation constructor should not allocate memory
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
  // Try to encode a command buffer with a vTensor that does not have
  // memory

  std::vector<int64_t> sizes = {4, 4, 1};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);

  // No allocations yet
  EXPECT_TRUE(get_vma_allocation_count() == 0);

  std::vector<float> data_a(a.gpu_numel());
  std::fill(data_a.begin(), data_a.end(), 2.5f);

  // Encoding a command buffer with a vTensor without memory should throw
  EXPECT_THROW(fill_vtensor(a, data_a), vkapi::Error);
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
  context()->flush();

  execute_and_check_add(a, b, c, 12.0f, 10.0f);
}

TEST_F(
    VulkanComputeAPITest,
    tensor_reallocation_with_deferred_allocation_test) {
  std::vector<int64_t> sizes = {8, 8, 8};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ false);

  vkapi::Allocation a_mem = allocate_memory_for(a);
  a.image().bind_allocation(a_mem);
  vkapi::Allocation b_mem = allocate_memory_for(b);
  b.image().bind_allocation(b_mem);
  vkapi::Allocation c_mem = allocate_memory_for(c);
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
    context()->flush();

    a.image().bind_allocation(a_mem);
    b.image().bind_allocation(b_mem);
    c.image().bind_allocation(c_mem);

    execute_and_check_add(
        a, b, c, float(new_sizes[1] + 4.5f), float(new_sizes[2] + 13.0f));
  }
}

TEST_F(VulkanComputeAPITest, texture_virtual_resize) {
  context()->set_cmd(/*reusable = */ true);
  std::vector<int64_t> sizes = {8, 12, 12};
  vTensor a = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor b = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);
  vTensor c = CREATE_FLOAT_TEXTURE(sizes, /*allocate_memory = */ true);

  DEFINE_STAGING_BUFFER_AND_RECORD_TO_GPU_FOR(a)
  DEFINE_STAGING_BUFFER_AND_RECORD_TO_GPU_FOR(b)

  fill_staging(staging_buffer_a, 11.5f);
  fill_staging(staging_buffer_b, 12.5f);

  record_binary_op(context(), "add", a, b, c);

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

#define EXTRACT_TENSOR(name)                                                 \
  std::vector<float> data_##name(graph.get_tensor(name.value)->gpu_numel()); \
  graph.copy_from_staging(name.staging, data_##name.data(), data_##name.size());

TEST(VulkanComputeGraphTest, test_values_scalars) {
  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef idx;

  idx = graph.add_scalar<int64_t>(4);
  EXPECT_TRUE(graph.get_int(idx) == 4);

  idx = graph.add_scalar<double>(5.5f);
  EXPECT_TRUE(graph.get_double(idx) == 5.5f);
}

TEST(VulkanComputeGraphTest, test_values_scalar_list_inplace_constructed) {
  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef idx = graph.add_scalar_list<int64_t>({1, 2, 3, 4});
  const auto arr = graph.get_int_list(idx);
  EXPECT_TRUE(arr->size() == 4);
  for (int i = 0; i < 4; i++) {
    EXPECT_TRUE(arr->at(i) == i + 1);
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
  const auto& arr = graph.get_double_list(idx);
  EXPECT_TRUE(arr->size() == 5);
  for (int i = 0; i < 5; i++) {
    EXPECT_TRUE(arr->at(i) == (5 - i));
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
  std::string stored = graph.get_string(idx);
  EXPECT_TRUE(stored == "hello, world");
}

TEST(VulkanComputeGraphTest, test_zero_dim_tensor) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {7, 3, 5};
  std::vector<int64_t> size_small = {};

  // Build graph

  IOValueRef a = graph.add_input_tensor(size_big, vkapi::kFloat);
  IOValueRef b = graph.add_input_tensor(size_small, vkapi::kFloat);

  IOValueRef out = {};

  out.value = graph.add_tensor(size_big, vkapi::kFloat);

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
    for (size_t i = 0; i < graph.get_tensor(out.value)->numel(); ++i) {
      CHECK_VALUE(data_out, i, val_c);
    }
  }
}

TEST(VulkanComputeGraphTest, test_simple_graph_with_buffer) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> sizes = {7, 13, 19};

  // Build graph

  IOValueRef a = graph.add_input_tensor(sizes, vkapi::kFloat, utils::kBuffer);

  IOValueRef out = {};

  out.value = graph.add_tensor(sizes, vkapi::kFloat, utils::kBuffer);

  auto addFn = VK_GET_OP_FN("aten.abs.default");
  addFn(graph, {a.value, out.value, kDummyValueRef, kDummyValueRef});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_execute();

  // Run graph

  for (float i = 5.0f; i < 30.0f; i += 10.0f) {
    float val = -i + 2.0f;
    float expected_val = std::abs(val);

    fill_vtensor(graph, a, val);

    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (size_t i = 0; i < graph.get_tensor(out.value)->numel(); ++i) {
      CHECK_VALUE(data_out, i, expected_val);
    }
  }
}

TEST(VulkanComputeGraphTest, test_simple_graph) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {8, 64, 124};
  std::vector<int64_t> size_small = {8, 1, 124};

  // Build graph

  IOValueRef a = graph.add_input_tensor(size_big, vkapi::kFloat);
  IOValueRef b = graph.add_input_tensor(size_small, vkapi::kFloat);

  IOValueRef out = {};

  out.value = graph.add_tensor(size_big, vkapi::kFloat);

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
    for (size_t i = 0; i < graph.get_tensor(out.value)->numel(); ++i) {
      CHECK_VALUE(data_out, i, val_c);
    }
  }
}

#define CREATE_WEIGHT_TENSOR(name, sizes, dtype, val)              \
  std::vector<float> data_##name(utils::multiply_integers(sizes)); \
  std::fill(data_##name.begin(), data_##name.end(), val);          \
  ValueRef name = graph.add_tensorref(sizes, dtype, data_##name.data());

TEST(VulkanComputeGraphTest, test_simple_prepacked_graph) {
  GraphConfig config;
  config.enable_querypool = true;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {8, 73, 62};
  std::vector<int64_t> size_small = {8, 73, 1};

  CREATE_WEIGHT_TENSOR(w1, size_small, vkapi::kFloat, 3.5f);
  CREATE_WEIGHT_TENSOR(w2, size_small, vkapi::kFloat, 3.0f);

  // Build graph

  IOValueRef a = graph.add_input_tensor(size_big, vkapi::kFloat);

  ValueRef c = graph.add_tensor(size_big, vkapi::kFloat);
  ValueRef e = graph.add_tensor(size_big, vkapi::kFloat);

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
    for (size_t i = 0; i < graph.get_tensor(out.value)->numel(); ++i) {
      CHECK_VALUE(data_out, i, val_out);
    }

    if (graph.context()->querypool()) {
      graph.context()->querypool().extract_results();
      graph.context()->querypool().print_results();
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
      vkapi::kFloat,
      /*shared_object_idx = */ 2);
  IOValueRef b = graph.add_input_tensor(
      size_small,
      vkapi::kFloat,
      /*shared_object_idx = */ 4);

  // +2: t.sizes_ubo() for each staging shader
  // +2: staging buffer for each input tensor
  EXPECT_TRUE(get_vma_allocation_count() == 4);

  ValueRef c = graph.add_tensor(
      size_big,
      vkapi::kFloat,
      /*shared_object_idx = */ 6);

  auto addFn = VK_GET_OP_FN("aten.add.Tensor");
  addFn(graph, {a.value, b.value, kDummyValueRef, c});

  IOValueRef d = graph.add_input_tensor(
      size_small,
      vkapi::kFloat,
      /*shared_object_idx = */ 2);

  // +2: alpha UBO, broadcast UBO for arithmetic shader
  // +1: t.sizes_ubo() uniform buffer for staging shader
  // +1: staging buffer for the input tensor
  EXPECT_TRUE(get_vma_allocation_count() == 9);

  ValueRef e = graph.add_tensor(
      size_big,
      vkapi::kFloat,
      /*shared_object_idx = */ 4);

  auto mulFn = VK_GET_OP_FN("aten.mul.Tensor");
  mulFn(graph, {c, d.value, e});

  IOValueRef out = {};
  out.value = e;
  out.staging = graph.set_output_tensor(out.value);

  // +2: alpha UBO, broadcast UBO for arithmetic shader
  // +1: t.sizes_ubo() for staging shader
  // +1 staging buffer for the input tensor
  EXPECT_TRUE(get_vma_allocation_count() == 13);

  graph.prepare();
  graph.encode_execute();

  // +3: shared memory allocations for tensors
  EXPECT_TRUE(get_vma_allocation_count() == 16);

  // Run graph

  std::vector<std::vector<int64_t>> new_sizes_list = {
      {8, 44, 34}, {4, 13, 56}, {8, 12, 64}, {12, 55, 33}, {4, 54, 10}};

  for (auto& new_sizes : new_sizes_list) {
    graph.get_tensor(a.value)->virtual_resize(new_sizes);
    graph.get_tensor(b.value)->virtual_resize(new_sizes);
    graph.get_tensor(c)->virtual_resize(new_sizes);
    graph.get_tensor(d.value)->virtual_resize(new_sizes);
    graph.get_tensor(e)->virtual_resize(new_sizes);

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
    for (size_t i = 0; i < graph.get_tensor(out.value)->numel(); i++) {
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
    EXPECT_TRUE(graph.get_tensor(out.value)->sizes() == new_sizes);

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
    for (size_t i = 0; i < graph.get_tensor(out.value)->numel(); i++) {
      CHECK_VALUE(data_out, i, val_out);
    }
  }
}

TEST(VulkanComputeGraphTest, test_large_graph) {
  auto build_start_time = std::chrono::system_clock::now();
  GraphConfig config;
  ComputeGraph graph(config);

  int64_t input_w = 256;
  int64_t input_h = 256;
  int64_t input_c = 8;

  std::vector<int64_t> size_big = {input_c, input_h, input_w};
  std::vector<int64_t> size_small = {input_c, input_h, 1};

  std::vector<int64_t> size_big_alt = {input_c / 2, input_h / 2, input_w / 2};
  std::vector<int64_t> size_small_alt = {input_c / 2, input_h / 2, 1};

  // Build graph

  IOValueRef a = graph.add_input_tensor(size_big, vkapi::kFloat, 2);
  IOValueRef b = graph.add_input_tensor(size_small, vkapi::kFloat, 4);

  ValueRef c = graph.add_tensor(size_big, vkapi::kFloat, 6);

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

  auto build_end_time = std::chrono::system_clock::now();

  auto build_time = std::chrono::duration_cast<std::chrono::microseconds>(
      build_end_time - build_start_time);

  std::stringstream ss;
  for (int i = 0; i < 10; i++) {
    auto resize_start_time = std::chrono::system_clock::now();
    if (i % 2 == 0) {
      graph.resize_input(0, size_big_alt);
      graph.resize_input(1, size_small_alt);
    } else {
      graph.resize_input(0, size_big);
      graph.resize_input(1, size_small);
    }
    graph.propagate_resize();
    auto resize_end_time = std::chrono::system_clock::now();

    auto resize_time = std::chrono::duration_cast<std::chrono::microseconds>(
        resize_end_time - resize_start_time);

    float val_a = 1.0f;
    float val_b = 2.0f;

    float val_e = val_a + val_b * (2 * n + 1);

    auto inference_start_time = std::chrono::system_clock::now();

    fill_vtensor(graph, a, val_a);
    fill_vtensor(graph, b, val_b);

    graph.execute();

    EXTRACT_TENSOR(out);

    auto inference_end_time = std::chrono::system_clock::now();

    auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
        inference_end_time - inference_start_time);

    for (int i = 0; i < graph.get_tensor(out.value)->numel(); i++) {
      CHECK_VALUE(data_out, i, val_e);
    }

    ss << "[          ] Resize:    " << std::setw(10) << std::right
       << resize_time.count() << " us" << std::endl;
    ss << "[          ] Inference: " << std::setw(10) << std::right
       << inference_time.count() << " us" << std::endl;
  }
  ss << "[          ] Model Load:" << std::setw(10) << std::right
     << build_time.count() << " us" << std::endl;
  std::cout << ss.str();
}

TEST(VulkanComputeGraphTest, test_etvk_copy_offset_node) {
  GraphConfig config;
  ComputeGraph graph(config);

  int64_t n = 6;
  int64_t c = 12;
  int64_t h = 4;
  int64_t w = 8;
  utils::GPUMemoryLayout memory_layout =
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;

  std::vector<int64_t> size = {n, c, h, w};

  IOValueRef a = graph.add_input_tensor(size, vkapi::kFloat, memory_layout);

  IOValueRef out = {};
  out.value = graph.add_tensor(size, vkapi::kFloat, memory_layout);

  // Notice that copy_node operates on in texture's x, y, z dimension. In the
  // comment, we provide the cooresponding coordinate in nchw.

  // src_offset is (n=0, c=4, h=1, w=1)
  ValueRef src_offset_ref = graph.add_scalar_list<int64_t>({1, 1, 1});

  // dst_offset is (n=1, c=8, h=2, w=0) in nchw coordinate
  // Argument is {x, y, z}.
  // x = 0 since w = 0
  // y = 2 since h = 2
  // z = c / 4 + 2 since
  //   1. there c/4 planes per batch, n=1 means we are on the first batch;
  //   2. +2 because c = 8, with channel packing it means two texels.
  ValueRef dst_offset_ref = graph.add_scalar_list<int64_t>({0, 2, c / 4 + 2});

  // range is (n=1, c=8, h=2, w=4)
  // Argument is {x, y, z}.
  // x = 4 since w = 4
  // y = 2 since h = 2
  // z = 2 since we are only copying 8 channels, hence 2 texel. n = 1 can be a
  // bit misleading here, since it gives the impression that we are copying the
  // entire channel. However, remember when we copy, we are trying to
  // dst[dst_offset:dst_offset + range] = src[src_offset:src_offset + range],
  // range must be non zero.
  ValueRef range_ref = graph.add_scalar_list<int64_t>({4, 2, 2});

  auto copyFn = VK_GET_OP_FN("etvk.copy_offset");
  copyFn(
      graph, {a.value, range_ref, src_offset_ref, dst_offset_ref, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_execute();

  fill_vtensor(graph, a, 0.0f, /*iota = */ true);

  graph.execute();

  EXTRACT_TENSOR(out);
  EXTRACT_TENSOR(a);

  // We will examine the results in the dst_range
  // The value in the cooresponding coordinate should match between the source
  // and destination tensor. We loop thru the range, calculate both the src and
  // dst index using the offsets, and compare the values in the extracted
  // vector. They should match.
  int n_idx = 0;
  // at each nested loop, index range from dst_offset to dst_offset + range

  for (int c_idx = 0; c_idx < 8; c_idx++) {
    for (int h_idx = 0; h_idx < 2; h_idx++) {
      for (int w_idx = 0; w_idx < 4; w_idx++) {
        auto dst_idx =
            get_buf_idx(graph, out, {n_idx + 1, c_idx + 8, h_idx + 2, w_idx});
        auto src_idx =
            get_buf_idx(graph, a, {n_idx, c_idx + 4, h_idx + 1, w_idx + 1});

        EXPECT_TRUE(data_out[dst_idx] == data_a[src_idx]);
      }
    }
  }
}

TEST(VulkanComputeGraphTest, test_etvk_copy_channel_offset_node) {
  GraphConfig config;
  ComputeGraph graph(config);

  int64_t n = 2;
  int64_t c = 12;
  int64_t h = 4;
  int64_t w = 8;
  utils::GPUMemoryLayout memory_layout =
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;

  std::vector<int64_t> size = {n, c, h, w};

  IOValueRef a = graph.add_input_tensor(size, vkapi::kFloat, memory_layout);

  IOValueRef out = {};
  out.value = graph.add_tensor(size, vkapi::kFloat, memory_layout);

  int64_t src_offset = 2;
  int64_t dst_offset = 3;
  int64_t range = 7;

  ValueRef src_offset_ref = graph.add_scalar<int64_t>(src_offset);
  ValueRef dst_offset_ref = graph.add_scalar<int64_t>(dst_offset);
  ValueRef range_ref = graph.add_scalar<int64_t>(range);

  auto copyFn = VK_GET_OP_FN("etvk.copy_channel_offset");
  copyFn(
      graph, {a.value, range_ref, src_offset_ref, dst_offset_ref, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_execute();

  fill_vtensor(graph, a, 0.0f, true);

  graph.execute();

  EXTRACT_TENSOR(out);
  EXTRACT_TENSOR(a);

  for (int n_idx = 0; n_idx < n; n_idx++) {
    for (int c_idx = 0; c_idx < range; c_idx++) {
      for (int h_idx = 0; h_idx < h; h_idx++) {
        for (int w_idx = 0; w_idx < w; w_idx++) {
          auto src_idx =
              get_buf_idx(graph, a, {n_idx, c_idx + src_offset, h_idx, w_idx});
          auto dst_idx = get_buf_idx(
              graph, out, {n_idx, c_idx + dst_offset, h_idx, w_idx});
          EXPECT_TRUE(data_out[dst_idx] == data_a[src_idx]);
        }
      }
    }
  }
}

TEST(
    VulkanComputeGraphTest,
    test_etvk_copy_channel_offset_node_clean_boundary) {
  // Tricky part for channel copy is handling the boundary across multiple copy.
  // For example, when we concat two [3, 1, 1] nchw-tensors along the channel
  // dimension, due to channel packing, elements from different source texel
  // will be packed into same destination texel at the boundaries.
  GraphConfig config;
  ComputeGraph graph(config);

  int64_t n = 2;
  int64_t c = 12;
  int64_t h = 4;
  int64_t w = 8;
  utils::GPUMemoryLayout memory_layout =
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;

  std::vector<int64_t> size = {n, c, h, w};

  IOValueRef zero = graph.add_input_tensor(size, vkapi::kFloat, memory_layout);
  IOValueRef a = graph.add_input_tensor(size, vkapi::kFloat, memory_layout);
  IOValueRef b = graph.add_input_tensor(size, vkapi::kFloat, memory_layout);

  IOValueRef out = {};
  out.value = graph.add_tensor(size, vkapi::kFloat, memory_layout);

  auto copyFn = VK_GET_OP_FN("etvk.copy_channel_offset");

  // Make sure entire out tensor is zeroed. The zero tensor will be filled with
  // zero later.
  copyFn(
      graph,
      {zero.value,
       graph.add_scalar<int64_t>(c),
       graph.add_scalar<int64_t>(0),
       graph.add_scalar<int64_t>(0),
       out.value});

  int64_t a_src_offset = 0;
  int64_t a_dst_offset = 2;
  int64_t a_range = 5;
  // a will write to channge [2, 7)
  copyFn(
      graph,
      {a.value,
       graph.add_scalar<int64_t>(a_range),
       graph.add_scalar<int64_t>(a_src_offset),
       graph.add_scalar<int64_t>(a_dst_offset),
       out.value});

  // b will write to channel [6, 11)
  // Intentional for b to override channel=6
  int64_t b_src_offset = 0;
  int64_t b_dst_offset = 6;
  int64_t b_range = 5;

  copyFn(
      graph,
      {b.value,
       graph.add_scalar<int64_t>(b_range),
       graph.add_scalar<int64_t>(b_src_offset),
       graph.add_scalar<int64_t>(b_dst_offset),
       out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_execute();

  float a_value = 1.0f;
  float b_value = 2.0f;
  float zero_value = 0.0f;
  fill_vtensor(graph, a, a_value);
  fill_vtensor(graph, b, b_value);
  fill_vtensor(graph, zero, zero_value);

  graph.execute();

  EXTRACT_TENSOR(out);

  for (int n_idx = 0; n_idx < n; n_idx++) {
    // c_idx only up to a_range-1 because the expected overwrite by b
    for (int c_idx = a_dst_offset; c_idx < a_dst_offset + a_range - 1;
         c_idx++) {
      for (int h_idx = 0; h_idx < h; h_idx++) {
        for (int w_idx = 0; w_idx < w; w_idx++) {
          auto dst_idx = get_buf_idx(graph, out, {n_idx, c_idx, h_idx, w_idx});
          EXPECT_TRUE(data_out[dst_idx] == a_value);
        }
      }
    }
  }

  for (int n_idx = 0; n_idx < n; n_idx++) {
    for (int c_idx = b_dst_offset; c_idx < b_dst_offset + b_range; c_idx++) {
      for (int h_idx = 0; h_idx < h; h_idx++) {
        for (int w_idx = 0; w_idx < w; w_idx++) {
          auto dst_idx = get_buf_idx(graph, out, {n_idx, c_idx, h_idx, w_idx});
          EXPECT_TRUE(data_out[dst_idx] == b_value);
        }
      }
    }
  }

  // Also verify that data before a_dst_offset and after b_dst_offset + b_range
  // are untouched.
  for (int n_idx = 0; n_idx < n; n_idx++) {
    for (int c_idx = 0; c_idx < a_dst_offset; c_idx++) {
      for (int h_idx = 0; h_idx < h; h_idx++) {
        for (int w_idx = 0; w_idx < w; w_idx++) {
          auto dst_idx = get_buf_idx(graph, out, {n_idx, c_idx, h_idx, w_idx});
          EXPECT_TRUE(data_out[dst_idx] == zero_value);
        }
      }
    }
  }

  for (int n_idx = 0; n_idx < n; n_idx++) {
    for (int c_idx = b_dst_offset + b_range; c_idx < c; c_idx++) {
      for (int h_idx = 0; h_idx < h; h_idx++) {
        for (int w_idx = 0; w_idx < w; w_idx++) {
          auto dst_idx = get_buf_idx(graph, out, {n_idx, c_idx, h_idx, w_idx});
          EXPECT_TRUE(data_out[dst_idx] == zero_value);
        }
      }
    }
  }
}

TEST(VulkanComputeGraphTest, test_etvk_copy_offset_int_node) {
  GraphConfig config;
  ComputeGraph graph(config);

  int64_t n = 6;
  int64_t c = 12;
  int64_t h = 4;
  int64_t w = 8;
  utils::GPUMemoryLayout memory_layout =
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;

  std::vector<int64_t> size = {n, c, h, w};

  IOValueRef a = graph.add_input_tensor(size, vkapi::kInt, memory_layout);

  IOValueRef out = {};
  out.value = graph.add_tensor(size, vkapi::kInt, memory_layout);

  // Notice that copy_node operates on in texture's x, y, z dimension. In the
  // comment, we provide the cooresponding coordinate in nchw.

  // src_offset is (n=0, c=4, h=1, w=1)
  ValueRef src_offset_ref = graph.add_scalar_list<int64_t>({1, 1, 1});

  // dst_offset is (n=1, c=8, h=2, w=0) in nchw coordinate
  // Argument is {x, y, z}.
  // x = 0 since w = 0
  // y = 2 since h = 2
  // z = c / 4 + 2 since
  //   1. there c/4 planes per batch, n=1 means we are on the first batch;
  //   2. +2 because c = 8, with channel packing it means two texels.
  ValueRef dst_offset_ref = graph.add_scalar_list<int64_t>({0, 2, c / 4 + 2});

  // range is (n=1, c=8, h=2, w=4)
  // Argument is {x, y, z}.
  // x = 4 since w = 4
  // y = 2 since h = 2
  // z = 2 since we are only copying 8 channels, hence 2 texel. n = 1 can be a
  // bit misleading here, since it gives the impression that we are copying the
  // entire channel. However, remember when we copy, we are trying to
  // dst[dst_offset:dst_offset + range] = src[src_offset:src_offset + range],
  // range must be non zero.
  ValueRef range_ref = graph.add_scalar_list<int64_t>({4, 2, 2});

  auto copyFn = VK_GET_OP_FN("etvk.copy_offset");
  copyFn(
      graph, {a.value, range_ref, src_offset_ref, dst_offset_ref, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_execute();

  fill_vtensor(graph, a, 0, /*iota = */ true);

  graph.execute();

  EXTRACT_TENSOR(out);
  EXTRACT_TENSOR(a);

  // We will examine the results in the dst_range
  // The value in the cooresponding coordinate should match between the source
  // and destination tensor. We loop thru the range, calculate both the src and
  // dst index using the offsets, and compare the values in the extracted
  // vector. They should match.
  int n_idx = 0;
  // at each nested loop, index range from dst_offset to dst_offset + range

  for (int c_idx = 0; c_idx < 8; c_idx++) {
    for (int h_idx = 0; h_idx < 2; h_idx++) {
      for (int w_idx = 0; w_idx < 4; w_idx++) {
        auto dst_idx =
            get_buf_idx(graph, out, {n_idx + 1, c_idx + 8, h_idx + 2, w_idx});
        auto src_idx =
            get_buf_idx(graph, a, {n_idx, c_idx + 4, h_idx + 1, w_idx + 1});

        EXPECT_TRUE(data_out[dst_idx] == data_a[src_idx]);
      }
    }
  }
}

TEST(VulkanComputeGraphTest, test_etvk_copy_channel_offset_int_node) {
  GraphConfig config;
  ComputeGraph graph(config);

  int64_t n = 2;
  int64_t c = 12;
  int64_t h = 4;
  int64_t w = 8;
  utils::GPUMemoryLayout memory_layout =
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED;

  std::vector<int64_t> size = {n, c, h, w};

  IOValueRef a = graph.add_input_tensor(size, vkapi::kFloat, memory_layout);

  IOValueRef out = {};
  out.value = graph.add_tensor(size, vkapi::kFloat, memory_layout);

  int64_t src_offset = 2;
  int64_t dst_offset = 3;
  int64_t range = 7;

  ValueRef src_offset_ref = graph.add_scalar<int64_t>(src_offset);
  ValueRef dst_offset_ref = graph.add_scalar<int64_t>(dst_offset);
  ValueRef range_ref = graph.add_scalar<int64_t>(range);

  auto copyFn = VK_GET_OP_FN("etvk.copy_channel_offset");
  copyFn(
      graph, {a.value, range_ref, src_offset_ref, dst_offset_ref, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_execute();

  fill_vtensor(graph, a, 0.0f, true);

  graph.execute();

  EXTRACT_TENSOR(out);
  EXTRACT_TENSOR(a);

  for (int n_idx = 0; n_idx < n; n_idx++) {
    for (int c_idx = 0; c_idx < range; c_idx++) {
      for (int h_idx = 0; h_idx < h; h_idx++) {
        for (int w_idx = 0; w_idx < w; w_idx++) {
          auto src_idx =
              get_buf_idx(graph, a, {n_idx, c_idx + src_offset, h_idx, w_idx});
          auto dst_idx = get_buf_idx(
              graph, out, {n_idx, c_idx + dst_offset, h_idx, w_idx});
          EXPECT_TRUE(data_out[dst_idx] == data_a[src_idx]);
        }
      }
    }
  }
}

TEST(VulkanComputeGraphTest, test_view_change_packing) {
  std::vector<std::pair<utils::GPUMemoryLayout, utils::GPUMemoryLayout>>
      layout_pairs = {
          {utils::kWidthPacked, utils::kChannelsPacked},
          {utils::kWidthPacked, utils::kHeightPacked},
          {utils::kWidthPacked, utils::kWidthPacked},
          {utils::kHeightPacked, utils::kChannelsPacked},
          {utils::kHeightPacked, utils::kHeightPacked},
          {utils::kHeightPacked, utils::kHeightPacked},
          {utils::kChannelsPacked, utils::kChannelsPacked},
          {utils::kChannelsPacked, utils::kHeightPacked},
          {utils::kChannelsPacked, utils::kHeightPacked},
      };

  int64_t n = 3;
  int64_t c = 2;
  int64_t h = 2;
  int64_t w = 5;
  std::vector<int64_t> size = {n, c, h, w};

  for (auto layout_pair : layout_pairs) {
    GraphConfig config;
    ComputeGraph graph(config);

    IOValueRef in =
        graph.add_input_tensor(size, vkapi::kFloat, layout_pair.first);

    IOValueRef out = {};
    out.value = graph.add_tensor(size, vkapi::kFloat, layout_pair.second);

    auto viewFn = VK_GET_OP_FN("aten.view_copy.default");
    viewFn(graph, {in.value, graph.add_none(), out.value});

    out.staging = graph.set_output_tensor(out.value);

    graph.prepare();
    graph.encode_execute();

    fill_vtensor(graph, in, 0.0, true);

    graph.execute();

    EXTRACT_TENSOR(out);

    // The extracted data is a flattened nchw buffer. Hence, should expect the
    // all elements inside the out array to match the index.
    for (int i = 0; i < graph.get_tensor(out.value)->numel(); i++) {
      CHECK_VALUE(data_out, i, i);
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
    context()->flush();

    // Make sure we are ending with a clean slate
    EXPECT_TRUE(get_vma_allocation_count() == 0);
  }
};

template <typename T>
void run_from_gpu_test(
    std::vector<int64_t>& sizes,
    utils::GPUMemoryLayout memory_layout =
        utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
    vkapi::ScalarType dtype = vkapi::kFloat,
    utils::StorageType storage_type = utils::StorageType::TEXTURE_3D) {
  if (dtype == vkapi::kHalf && !context()->adapter_ptr()->has_16bit_storage()) {
    return;
  }
  if ((dtype == vkapi::kChar || dtype == vkapi::kQInt8) &&
      !context()->adapter_ptr()->has_full_int8_buffers_support()) {
    return;
  }
  vTensor vten = vTensor(context(), sizes, dtype, storage_type, memory_layout);

  std::string kernel_name("idx_fill_texture");
  add_memory_layout_suffix(kernel_name, vten);
  add_dtype_suffix(kernel_name, vten);

  {
    vkapi::PipelineBarrier pipeline_barrier{};
    vkapi::SpecVarList specialization_constants = {vten.packed_dim_whcn_idx()};
    context()->submit_compute_job(
        VK_KERNEL_FROM_STR(kernel_name),
        pipeline_barrier,
        vten.image_extents(),
        {4, 4, 4},
        specialization_constants,
        VK_NULL_HANDLE,
        0,
        vten.image(
            pipeline_barrier,
            vkapi::PipelineStage::COMPUTE,
            vkapi::MemoryAccessType::WRITE),
        vten.sizes_ubo());
  }

  StorageBuffer staging_buffer(context(), dtype, vten.gpu_numel());

  record_image_to_nchw_op(context(), vten, staging_buffer.buffer());

  submit_to_gpu();

  std::vector<T> data_out(staging_buffer.numel());
  copy_staging_to_ptr(staging_buffer, data_out.data(), staging_buffer.nbytes());

  for (int i = 0; i < vten.numel(); i++) {
    CHECK_VALUE(data_out, i, i);
  }
}

template <typename T>
void run_to_gpu_test(
    std::vector<int64_t>& sizes,
    utils::GPUMemoryLayout memory_layout =
        utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
    vkapi::ScalarType dtype = vkapi::kFloat,
    utils::StorageType storage_type = utils::StorageType::TEXTURE_3D) {
  if (dtype == vkapi::kHalf && !context()->adapter_ptr()->has_16bit_storage()) {
    return;
  }
  if ((dtype == vkapi::kChar || dtype == vkapi::kQInt8) &&
      !context()->adapter_ptr()->has_full_int8_buffers_support()) {
    return;
  }

  vTensor vten = vTensor(context(), sizes, dtype, storage_type, memory_layout);

  // Create and fill input staging buffer
  StorageBuffer staging_buffer_in(context(), dtype, vten.gpu_numel());

  std::vector<T> data_in(staging_buffer_in.numel());
  for (int i = 0; i < staging_buffer_in.numel(); i++) {
    data_in[i] = i;
  }
  copy_ptr_to_staging(data_in.data(), staging_buffer_in, vten.gpu_nbytes());

  // Output staging buffer
  StorageBuffer staging_buffer_out(context(), dtype, vten.gpu_numel());

  // Copy data in and out of the tensor
  record_nchw_to_image_op(context(), staging_buffer_in.buffer(), vten);
  record_image_to_nchw_op(context(), vten, staging_buffer_out.buffer());

  // Execute command buffer
  submit_to_gpu();

  // Extract data from output staging buffer
  std::vector<T> data_out(staging_buffer_out.numel());
  copy_staging_to_ptr(
      staging_buffer_out, data_out.data(), staging_buffer_out.nbytes());

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

  // These sizes are set such that the total number of elements is less than
  // 128 which is the maximum representable value for int8.
  std::vector<std::vector<int64_t>> to_test_int8 = {
      // 2D sizes
      {14, 7},
      // 3D sizes
      {3, 7, 5},
      {4, 2, 11},
      // 4D sizes
      {3, 3, 3, 3},
      {7, 1, 6, 3},
  };

#define RUN_TESTS(ctype, dtype)                                      \
  run_to_gpu_test<ctype>(                                            \
      sizes, utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, dtype); \
  run_to_gpu_test<ctype>(                                            \
      sizes, utils::GPUMemoryLayout::TENSOR_WIDTH_PACKED, dtype);    \
  run_to_gpu_test<ctype>(                                            \
      sizes, utils::GPUMemoryLayout::TENSOR_HEIGHT_PACKED, dtype);

  for (auto& sizes : to_test) {
    RUN_TESTS(float, vkapi::kFloat)
    RUN_TESTS(torch::executor::Half, vkapi::kHalf)
  }

  for (auto& sizes : to_test_int8) {
    RUN_TESTS(int8_t, vkapi::kChar);
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
    vkapi::ScalarType dtype,
    utils::GPUMemoryLayout memory_layout,
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

#define CALL_TEST_FN_FORALL_CONDITIONS(_)                                 \
  _(vkapi::kFloat, utils::GPUMemoryLayout::TENSOR_WIDTH_PACKED, false)    \
  _(vkapi::kFloat, utils::GPUMemoryLayout::TENSOR_HEIGHT_PACKED, false)   \
  _(vkapi::kFloat, utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, false) \
  _(vkapi::kFloat, utils::GPUMemoryLayout::TENSOR_WIDTH_PACKED, true)     \
  _(vkapi::kFloat, utils::GPUMemoryLayout::TENSOR_HEIGHT_PACKED, true)    \
  _(vkapi::kFloat, utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, true)

#define CALL_TEST_FN_FOR_W_PACKED(_)                                   \
  _(vkapi::kFloat, utils::GPUMemoryLayout::TENSOR_WIDTH_PACKED, false) \
  _(vkapi::kFloat, utils::GPUMemoryLayout::TENSOR_WIDTH_PACKED, true)

#define CALL_TEST_FN_FOR_C_PACKED(_)                                      \
  _(vkapi::kFloat, utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, false) \
  _(vkapi::kFloat, utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, true)

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
    vkapi::ScalarType dtype,
    utils::GPUMemoryLayout memory_layout,
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
      in_size, vkapi::kFloat, utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  IOValueRef out_ioval;
  out_ioval.value = graph.add_tensor(
      out_size, vkapi::kFloat, utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  IOValueRef idx_ioval;
  idx_ioval.value = graph.add_tensor(
      out_size, vkapi::kInt, utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
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

  vTensorPtr t_in = graph.get_tensor(in_ioval.value);
  std::vector<float> input_data(t_in->gpu_numel());
  graph.copy_from_staging(
      in_ioval.staging, input_data.data(), input_data.size());

  graph.execute();

  vTensorPtr t_out = graph.get_tensor(out_ioval.value);
  std::vector<float> output_data(t_out->gpu_numel());
  graph.copy_from_staging(
      out_ioval.staging, output_data.data(), output_data.size());
  vTensorPtr t_idx = graph.get_tensor(idx_ioval.value);
  std::vector<int> index_data(t_idx->gpu_numel());
  graph.copy_from_staging(
      idx_ioval.staging, index_data.data(), index_data.size());

  // Check results

  int h_offset = kernel_copy[0] - 1;
  int w_offset = kernel_copy[1] - 1;
  int h_out = utils::val_at(-2, t_out->sizes());
  int w_out = utils::val_at(-1, t_out->sizes());
  int w_in = utils::val_at(-1, t_in->sizes());
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
      context(),
      gpu_sizes,
      vkapi::kFloat,
      utils::StorageType::TEXTURE_2D,
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);

  // Create and fill input staging buffer
  const int64_t in_numel = utils::multiply_integers(original_sizes);
  StorageBuffer staging_buffer_in(context(), vkapi::kFloat, in_numel);

  std::vector<float> data_in(in_numel);
  for (int i = 0; i < in_numel; i++) {
    data_in[i] = i + 1;
  }
  copy_ptr_to_staging(
      data_in.data(), staging_buffer_in, sizeof(float) * in_numel);

  // Output staging buffer
  const int64_t out_numel =
      padded_sizes[0] * padded_sizes[1] * original_sizes[2] * original_sizes[3];
  StorageBuffer staging_buffer_out(context(), vkapi::kFloat, out_numel);

  // Copy data in and out of the tensor
  record_conv2d_prepack_weights_op(
      context(), staging_buffer_in.buffer(), vten, original_sizes, transposed);
  record_image_to_nchw_op(context(), vten, staging_buffer_out.buffer());

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
