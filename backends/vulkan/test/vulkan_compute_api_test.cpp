/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <bitset>
#include <utility>
#include <vector>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/QPackUtils.h>

#include <executorch/backends/vulkan/test/utils/test_utils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DispatchNode.h>

using namespace vkcompute;
using namespace vkcompute::api;

std::vector<float>
transpose_matrix(std::vector<float>& mat, const int H, const int W) {
  std::vector<float> out(W * H);
  for (int out_y = 0; out_y < H; ++out_y) {
    for (int out_x = 0; out_x < W; ++out_x) {
      out[out_x * H + out_y] = mat[out_y * W + out_x];
    }
  }
  return out;
}

std::vector<float> compute_reference_matmul(
    std::vector<float>& mat1,
    std::vector<float>& mat2,
    const int M,
    const int K,
    const int N) {
  std::vector<float> out(M * N);
  for (int out_y = 0; out_y < M; ++out_y) {
    for (int out_x = 0; out_x < N; ++out_x) {
      out[out_y * N + out_x] = 0;
      for (int k = 0; k < K; ++k) {
        out[out_y * N + out_x] += mat1[out_y * K + k] * mat2[k * N + out_x];
      }
    }
  }
  return out;
}

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
    const bool unsqueezed = false) {
  int64_t C = utils::val_at(-3, sizes);
  int64_t H = utils::val_at(-2, sizes);
  int64_t W = utils::val_at(-1, sizes);

  int64_t numel = utils::multiply_integers(sizes);

  switch (layout) {
    case utils::kWidthPacked:
      switch (sizes.size()) {
        case 1:
          if (unsqueezed)
            return {numel, numel, numel, 1};
          return {1};
        case 2:
          if (unsqueezed)
            return {numel, numel, W, 1};
          return {W, 1};
        case 3:
          if (unsqueezed)
            return {numel, H * W, W, 1};
          return {H * W, W, 1};
        case 4:
          return {C * H * W, H * W, W, 1};
        default:
          return {};
      }
      break;
    case utils::kHeightPacked:
      switch (sizes.size()) {
        case 1:
          if (unsqueezed)
            return {numel, numel, numel, 1};
          return {1};
        case 2:
          if (unsqueezed)
            return {numel, numel, 1, H};
          return {1, H};
        case 3:
          if (unsqueezed)
            return {numel, H * W, 1, H};
          return {W * H, 1, H};
        case 4:
          return {C * W * H, W * H, 1, H};
        default:
          return {};
      }
    case utils::kChannelsPacked:
      switch (sizes.size()) {
        case 1:
          if (unsqueezed)
            return {numel, numel, numel, 1};
          return {1};
        case 2:
          if (unsqueezed)
            return {numel, numel, W, 1};
          return {W, 1};
        case 3:
          if (unsqueezed)
            return {numel, 1, W * C, C};
          return {1, W * C, C};
        case 4:
          return {H * W * C, 1, W * C, C};
        default:
          return {};
      }
  }
  return {};
}

TEST_F(VulkanComputeAPITest, empty_init_shader_info_test) {
  vkapi::ShaderInfo empty_shader_info;
  EXPECT_FALSE(empty_shader_info);
  EXPECT_TRUE(empty_shader_info.src_code.bin == nullptr);
  EXPECT_TRUE(empty_shader_info.src_code.size == 0u);
}

TEST_F(VulkanComputeAPITest, calculate_dim_order_test) {
  // ndim, GPUMemoryLayout, expected dim order pairs
  std::vector<std::tuple<size_t, int32_t, std::vector<int64_t>>> test_cases = {
      {1, WHCN::kWidthDim, {0}},
      {1, WHCN::kHeightDim, {0}},
      {1, WHCN::kChannelsDim, {0}},
      {2, WHCN::kWidthDim, {0, 1}},
      {2, WHCN::kHeightDim, {1, 0}},
      {2, WHCN::kChannelsDim, {0, 1}},
      {3, WHCN::kWidthDim, {0, 1, 2}},
      {3, WHCN::kHeightDim, {0, 2, 1}},
      {3, WHCN::kChannelsDim, {1, 2, 0}},
      {4, WHCN::kWidthDim, {0, 1, 2, 3}},
      {4, WHCN::kHeightDim, {0, 1, 3, 2}},
      {4, WHCN::kChannelsDim, {0, 2, 3, 1}},
  };

  for (const auto& test_case : test_cases) {
    const size_t& ndim = std::get<0>(test_case);
    const int32_t packed_dim = std::get<1>(test_case);
    const auto& expected_dim_order = std::get<2>(test_case);
    std::vector<int64_t> dim_order = calculate_dim_order(ndim, packed_dim);

    ASSERT_TRUE(dim_order == expected_dim_order);
  }
}

TEST_F(VulkanComputeAPITest, calculate_tensor_strides_test) {
  vTensor v_tensor_to_resize(
      context(),
      {25, 25, 25, 25},
      vkapi::kFloat,
      utils::kBuffer,
      utils::kWidthPacked,
      /*allocate_memory = */ false);

  for (const auto& sizes : standard_sizes_to_test) {
    if (sizes.size() < 3) {
      continue;
    }
    for (const auto& layout :
         {utils::kWidthPacked, utils::kHeightPacked, utils::kChannelsPacked}) {
      {
        const int32_t packed_dim = static_cast<int32_t>(layout);
        std::vector<int64_t> dim_order =
            calculate_dim_order(sizes.size(), packed_dim);
        std::vector<int64_t> strides = calculate_strides(sizes, dim_order);
        std::vector<int64_t> ref_strides = get_reference_strides(sizes, layout);
        ASSERT_TRUE(strides == ref_strides);

        int64_t numel = utils::multiply_integers(sizes);
        std::vector<int64_t> unsqueezed_strides =
            unsqueeze_strides(strides, numel);
        std::vector<int64_t> ref_unsqueezed_strides =
            get_reference_strides(sizes, layout, true);

        ASSERT_TRUE(unsqueezed_strides == ref_unsqueezed_strides);

        // Create new vTensor and check that the strides are correct
        vTensor new_v_tensor(
            context(),
            sizes,
            vkapi::kFloat,
            utils::kBuffer,
            layout,
            /*allocate_memory = */ false);

        ASSERT_TRUE(new_v_tensor.strides() == ref_strides);
        ASSERT_TRUE(
            new_v_tensor.unsqueezed_strides() == ref_unsqueezed_strides);

        // Resize vtensor and check that updated metadata is correct
        v_tensor_to_resize.virtual_reconfigure(sizes, dim_order);
        ASSERT_TRUE(v_tensor_to_resize.strides() == ref_strides);
        ASSERT_TRUE(
            v_tensor_to_resize.unsqueezed_strides() == ref_unsqueezed_strides);
      }
    }
  }
}

TEST_F(VulkanComputeAPITest, virtual_transpose_test) {
  std::vector<int64_t> sizes = {7, 9, 11, 13};
  // (dim0, dim1), new_sizes, new_dim_order, new_axis_map, new_packed_dim_idx
  std::vector<std::vector<std::vector<int64_t>>> test_cases = {
      {{2, 3}, {7, 9, 13, 11}, {0, 1, 3, 2}, {1, 0, 2, 2}, {1}},
      {{2, 1}, {7, 11, 9, 13}, {0, 2, 1, 3}, {0, 2, 1, 1}, {0}},
      {{1, 3}, {7, 13, 11, 9}, {0, 3, 2, 1}, {2, 1, 0, 0}, {2}},
  };

  for (const auto& test_case : test_cases) {
    const int dim0 = test_case.at(0).at(0);
    const int dim1 = test_case.at(0).at(1);

    const auto& expected_sizes = test_case.at(1);
    const auto& expected_dim_order = test_case.at(2);
    const auto& expected_axis_map = test_case.at(3);
    const int expected_packed_dim = test_case.at(4).at(0);

    {
      vTensor a_buffer = vTensor(
          context(), sizes, vkapi::kFloat, utils::kBuffer, utils::kWidthPacked);

      a_buffer.virtual_transpose(dim0, dim1);
      EXPECT_TRUE(a_buffer.sizes() == expected_sizes);
      EXPECT_TRUE(a_buffer.dim_order() == expected_dim_order);
    }

    {
      vTensor a_texture = vTensor(
          context(),
          sizes,
          vkapi::kFloat,
          utils::kTexture3D,
          utils::kWidthPacked);
      a_texture.virtual_transpose(dim0, dim1);
      EXPECT_TRUE(a_texture.sizes() == expected_sizes);
      EXPECT_TRUE(a_texture.axis_map() == expected_axis_map);
      EXPECT_TRUE(a_texture.packed_dim() == expected_packed_dim);
    }
  }
}

TEST_F(VulkanComputeAPITest, view_of_view_test) {
  constexpr int N = 3;
  constexpr int C = 5;
  constexpr int H = 17;
  constexpr int W = 19;

  std::vector<int64_t> sizes = {N, C, H, W};

  vTensor t1 = vTensor(
      context(), sizes, vkapi::kFloat, utils::kTexture3D, utils::kWidthPacked);

  vTensor t2 = vTensor(t1);
  EXPECT_TRUE(t2.sizes() == sizes);
  vTensor t3 = vTensor(t2);
  EXPECT_TRUE(t2.sizes() == sizes);

  t2.virtual_transpose(1, 2);
  std::vector<int64_t> expected_t2_sizes = {N, H, C, W};
  EXPECT_TRUE(t2.sizes() == expected_t2_sizes);

  // Because t3 was created before t2's metadata was updated, we need to first
  // update t3's metadata to match t2's metadata. Then the transpose will yield
  // the correct metadata.
  t3.virtual_clone(t2);
  t3.virtual_transpose(2, 3);
  std::vector<int64_t> expected_t3_sizes = {N, H, W, C};
  EXPECT_TRUE(t3.sizes() == expected_t3_sizes);
}

utils::ivec3 make_temp_ivec3(int x, int y, int z) {
  return utils::ivec3{x, y, z};
}

TEST_F(VulkanComputeAPITest, vec_test) {
  {
    utils::vec3 v3({1, 2, 3});
    ASSERT_TRUE(v3[0] == 1);
    ASSERT_TRUE(v3[1] == 2);
    ASSERT_TRUE(v3[2] == 3);
    v3 = {4, 5, 6};
    ASSERT_TRUE(v3[0] == 4);
    ASSERT_TRUE(v3[1] == 5);
    ASSERT_TRUE(v3[2] == 6);
  }

  {
    utils::uvec4 uv4({4, 3, 2, 1});
    ASSERT_TRUE(uv4[0] == 4);
    ASSERT_TRUE(uv4[1] == 3);
    ASSERT_TRUE(uv4[2] == 2);
    ASSERT_TRUE(uv4[3] == 1);
    uv4 = {11, 13, 12, 88};
    ASSERT_TRUE(uv4[0] == 11);
    ASSERT_TRUE(uv4[1] == 13);
    ASSERT_TRUE(uv4[2] == 12);
    ASSERT_TRUE(uv4[3] == 88);
  }

  // Test copy from same type
  {
    utils::ivec3 v{5, 6, 8};
    utils::ivec3 v2 = v;

    ASSERT_TRUE(v2[0] == 5);
    ASSERT_TRUE(v2[1] == 6);
    ASSERT_TRUE(v2[2] == 8);
  }

  // Test copy from different type
  {
    utils::uvec3 v{5, 6, 8};
    utils::ivec3 v2 = v;

    ASSERT_TRUE(v2[0] == 5);
    ASSERT_TRUE(v2[1] == 6);
    ASSERT_TRUE(v2[2] == 8);
  }

  // Test construction from temporary vec
  {
    utils::uvec3 v{make_temp_ivec3(4, 5, 10)};
    ASSERT_TRUE(v[0] == 4);
    ASSERT_TRUE(v[1] == 5);
    ASSERT_TRUE(v[2] == 10);
  }

  // Test initalization from temporary vec
  {
    utils::uvec3 v = make_temp_ivec3(4, 5, 10);
    ASSERT_TRUE(v[0] == 4);
    ASSERT_TRUE(v[1] == 5);
    ASSERT_TRUE(v[2] == 10);
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
  StagingBuffer buffer(context(), vkapi::kFloat, len);

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
  buffer.copy_to(data.data(), buffer.nbytes());

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

  StagingBuffer staging_buffer(
      context(), vkapi::kFloat, a.staging_buffer_numel());
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
  StagingBuffer buffer(context(), dtype, len);

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
  buffer.copy_to(data.data(), buffer.nbytes());

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
  test_storage_buffer_type<executorch::aten::Half, vkapi::kHalf>(16);
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
            run_buffer_tensor_sanity_check<executorch::aten::Half>(a);
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

TEST_F(VulkanComputeAPITest, tensor_alias_test) {
  for (utils::StorageType storage_type : {utils::kTexture3D, utils::kBuffer}) {
    std::vector<int64_t> sizes = {9, 9};

    const size_t alloc_count_before = get_vma_allocation_count();

    vTensor original = vTensor(context(), sizes, vkapi::kFloat, storage_type);

    vTensor copy = vTensor(original);

    // Two tensors but only one additional allocation.
    EXPECT_TRUE(get_vma_allocation_count() == alloc_count_before + 1);
    EXPECT_TRUE(copy.is_view_of(original));

    // Fill original tensor with some data
    fill_vtensor(original, 2.5f, true);

    std::vector<float> data_out(copy.staging_buffer_numel());
    // Extract the copy tensor; should contain the data of the original tensor
    extract_vtensor(copy, data_out);

    for (size_t i = 0; i < original.numel(); ++i) {
      CHECK_VALUE(data_out, i, 2.5f + i);
    }
  }
}

TEST_F(VulkanComputeAPITest, tensor_no_copy_transpose_test) {
  constexpr int M = 11;
  constexpr int K = 23;
  constexpr int N = 17;
  std::vector<int64_t> mat1_sizes = {M, K};
  std::vector<int64_t> mat2_sizes = {N, K};
  std::vector<int64_t> out_sizes = {M, N};

  for (const auto storage_type : {utils::kTexture3D, utils::kBuffer}) {
    vTensor mat1 = vTensor(
        context(),
        mat1_sizes,
        vkapi::kFloat,
        storage_type,
        utils::kWidthPacked);
    vTensor mat2 = vTensor(
        context(),
        mat2_sizes,
        vkapi::kFloat,
        storage_type,
        utils::kWidthPacked);
    vTensor out = vTensor(
        context(), out_sizes, vkapi::kFloat, storage_type, utils::kWidthPacked);

    // Generate data
    std::vector<float> mat1_data =
        create_random_float_buffer(mat1.staging_buffer_numel());
    std::vector<float> mat2_data =
        create_random_float_buffer(mat2.staging_buffer_numel());

    // Create direct view and modify sizes and strides later
    vTensor mat2_t = vTensor(mat2);
    // Update sizes and strides of mat2_t to be that of a transposed tensor
    mat2_t.virtual_transpose(0, 1);

    EXPECT_TRUE(mat2_t.packed_dim() == WHCN::kHeightDim);

    std::vector<float> mat2_t_data = transpose_matrix(mat2_data, N, K);
    std::vector<float> ref_out =
        compute_reference_matmul(mat1_data, mat2_t_data, M, K, N);

    // Fill original tensor with some data
    fill_vtensor(mat1, mat1_data);
    fill_vtensor(mat2, mat2_data);

    if (storage_type == utils::kTexture3D) {
      record_matmul_texture3d(context(), out, mat1, mat2_t);
    } else {
      record_reference_matmul(context(), out, mat1, mat2_t);
    }

    std::vector<float> data_out(out.staging_buffer_numel());
    // Extract the copy tensor; should contain the data of the original tensor
    extract_vtensor(out, data_out);

    for (size_t i = 0; i < ref_out.size(); ++i) {
      EXPECT_TRUE(check_close(data_out[i], ref_out[i]));
    }
  }
}

TEST_F(VulkanComputeAPITest, tensor_no_copy_slice_test) {
  constexpr int L = 31;

  // S{N} refers to slice {N}
  constexpr int L_S1 = 17;
  constexpr int O_S1 = 5;

  constexpr int L_S2 = 7;
  constexpr int O_S2 = 3;

  std::vector<int64_t> dim_order = {0};

  std::vector<int64_t> t_sizes = {L};
  std::vector<int64_t> s1_sizes = {L_S1};
  std::vector<int64_t> s2_sizes = {L_S2};

  vTensor orig = CREATE_FLOAT_BUFFER(t_sizes, /*allocate_memory=*/true);

  fill_vtensor(orig, 0);

  vTensor s1 = vTensor(orig, s1_sizes, dim_order, O_S1);
  vTensor s2 = vTensor(s1, s2_sizes, dim_order, O_S2);

  record_scalar_add_buffer(api::context(), s1, 4.5f);
  record_scalar_add_buffer(api::context(), s2, 7.5f);

  std::vector<float> orig_data(orig.staging_buffer_numel());
  extract_vtensor(orig, orig_data);

  int id = 0;
  while (id < O_S1) {
    EXPECT_TRUE(orig_data[id] == 0);
    ++id;
  }
  while (id < O_S1 + O_S2) {
    EXPECT_TRUE(orig_data[id] == 4.5);
    ++id;
  }
  while (id < O_S1 + O_S2 + L_S2) {
    EXPECT_TRUE(orig_data[id] == 12);
    ++id;
  }
  while (id < O_S1 + L_S1) {
    EXPECT_TRUE(orig_data[id] == 4.5);
    ++id;
  }
  while (id < L) {
    EXPECT_TRUE(orig_data[id] == 0);
    ++id;
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

  std::vector<float> data_a(a.staging_buffer_numel());
  std::fill(data_a.begin(), data_a.end(), 2.5f);
  std::vector<float> data_b(b.staging_buffer_numel());
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

  std::vector<float> data_c(c.staging_buffer_numel());
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
  std::vector<float> data_a(a.staging_buffer_numel());
  std::fill(data_a.begin(), data_a.end(), 2.5f);
  std::vector<float> data_b(b.staging_buffer_numel());
  std::fill(data_b.begin(), data_b.end(), 1.5f);
  std::vector<float> data_d(b.staging_buffer_numel());
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
  std::vector<float> data_e(e.staging_buffer_numel());
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

  std::vector<float> data_a(a.staging_buffer_numel());
  std::fill(data_a.begin(), data_a.end(), 2.5f);

  // Encoding a command buffer with a vTensor without memory should throw
  EXPECT_THROW(fill_vtensor(a, data_a), vkapi::Error);
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

    fill_staging(
        staging_buffer_a, float(new_sizes[1] + 1.5f), a.staging_buffer_numel());
    fill_staging(
        staging_buffer_b,
        float(new_sizes[2] + 55.0f),
        b.staging_buffer_numel());

    submit_to_gpu();
    check_staging_buffer(
        staging_buffer_c,
        float(new_sizes[1] + new_sizes[2] + 56.5f),
        c.staging_buffer_numel());
  }
}

//
// Compute Graph Tests
//

#define EXTRACT_TENSOR(name)                                 \
  std::vector<float> data_##name(                            \
      graph.get_tensor(name.value)->staging_buffer_numel()); \
  graph.copy_from_staging(name.staging, data_##name.data(), data_##name.size());

// The purpose of this test is simply to track the size of various classes over
// time, in the interest of making sure that they doesn't grow too large.
TEST_F(VulkanComputeAPITest, print_object_sizes) {
#define PRINT_SIZE(name) \
  std::cout << #name << " size: " << sizeof(name) << " B" << std::endl
  PRINT_SIZE(vTensor);
  PRINT_SIZE(Value);
  PRINT_SIZE(StagingBuffer);
  PRINT_SIZE(ComputeGraph);
  PRINT_SIZE(DispatchNode);
#undef PRINT_SIZE

  // The actual sizes of each object is dependent on the platform. However, we
  // can alert ourselves to any significant changes in the sizes of these
  // objects by checking the `sizeof()` the class against some loose thresholds.

  // Current known size on 64 bit system: 1040 B
  EXPECT_TRUE(sizeof(vTensor) < 1200);
  // Current known size on 64 bit system: 48 B
  EXPECT_TRUE(sizeof(Value) < 56);
  // Current known size on 64 bit system: 120 B
  EXPECT_TRUE(sizeof(StagingBuffer) < 500);
  // Current known size on 64 bit system: 384 B
  EXPECT_TRUE(sizeof(ComputeGraph) < 500);
  // Current known size on 64 bit system: 248 B
  EXPECT_TRUE(sizeof(DispatchNode) < 500);
}

TEST_F(VulkanComputeAPITest, test_tensor_creation_from_vulkan_image) {
  const auto w = 16;
  const auto h = 12;
  const auto d = 1;
  const utils::uvec3 image_extents = {w, h, d};

  vkapi::Adapter* adapter_ptr = context()->adapter_ptr();

  vkapi::ImageSampler::Properties sampler_props{
      VK_FILTER_NEAREST,
      VK_SAMPLER_MIPMAP_MODE_NEAREST,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
  };

  VkFormat image_format = VK_FORMAT_R32G32B32A32_SFLOAT;
  VkImageType image_type = VK_IMAGE_TYPE_3D;
  VkImageViewType image_view_type = VK_IMAGE_VIEW_TYPE_3D;

  VkSampler sampler = adapter_ptr->sampler_cache().retrieve(sampler_props);

  auto image = adapter_ptr->vma().create_image(
      context()->device(),
      vkapi::create_extent3d(image_extents),
      image_format,
      image_type,
      context()->preferred_image_tiling(),
      image_view_type,
      sampler_props,
      sampler,
      /*allow_transfer = */ true,
      /*allocate_memory = */ true);

  auto tensor = vTensor(context(), image);

  const auto exp_sizes = std::vector<int64_t>{w, h, d * 4};
  EXPECT_TRUE(tensor.sizes() == exp_sizes);
  EXPECT_TRUE(tensor.packed_dim() == 2);

  const auto exp_numel = w * h * d * 4;
  EXPECT_TRUE(tensor.numel() == exp_numel);
  EXPECT_TRUE(tensor.padded_numel() == exp_numel);
}

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

TEST(VulkanComputeGraphTest, empty_init_graphnode_test) {
  ExecuteNode node(nullptr, {});

  GraphConfig config;
  ComputeGraph graph(config);

  // Encode an empty ExecuteNode and check that command buffer encoding does not
  // crash.
  graph.execute_nodes().emplace_back(new ExecuteNode(nullptr, {}));
  EXPECT_NO_FATAL_FAILURE(graph.encode_execute());
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

TEST(VulkanComputeGraphTest, test_simple_graph_with_view) {
  constexpr int W = 7;
  constexpr int H = 7;
  // slice height
  constexpr int S_H = 2;
  // slice offset
  constexpr int S_O = 3;

  GraphConfig config;
  config.set_storage_type_override(utils::kBuffer);
  ComputeGraph graph(config);

  std::vector<int64_t> dim_order = {0, 1};

  std::vector<int64_t> orig_sizes = {H, W};
  std::vector<int64_t> slice_sizes = {S_H, W};
  const int offset = S_O * W;

  // Build graph

  IOValueRef orig = graph.add_input_tensor(orig_sizes, vkapi::kFloat);
  ValueRef slice =
      graph.add_tensor_view(orig.value, slice_sizes, dim_order, offset);

  EXPECT_TRUE(graph.val_is_view_of(slice, orig.value));

  IOValueRef out = {};

  out.value = graph.add_tensor(slice_sizes, vkapi::kFloat);

  auto opFn = VK_GET_OP_FN("aten.abs.default");
  opFn(graph, {slice, out.value, kDummyValueRef, kDummyValueRef});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_execute();

  // Run graph

  for (float i = 5.0f; i < 30.0f; i += 10.0f) {
    float start_val = -130 + i;

    fill_vtensor(graph, orig, start_val, true);

    graph.execute();

    EXTRACT_TENSOR(out);

    for (size_t i = 0; i < graph.get_tensor(out.value)->numel(); ++i) {
      const float expected_val = std::abs(start_val) - float(offset) - i;
      CHECK_VALUE(data_out, i, expected_val);
    }
  }
}

TEST(VulkanComputeGraphTest, test_graph_view_of_view) {
  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);

  constexpr int N = 3;
  constexpr int C = 5;
  constexpr int H = 17;
  constexpr int W = 19;

  std::vector<int64_t> orig_sizes = {N, C, H, W};

  // Test a common view of view usage pattern. In delegate execution, the values
  // of the graph are created first; then operators are added. As a result,
  // creating views of views is a bit tricky because metadata updates to a view
  // does not update the metadata of the view's views. Nonetheless, view
  // operators have an implicit assumption that the metadata of the output is
  // equivalent to the metadata of the input. Therefore, view operators must
  // account for unseen updates to the input view by first calling
  // `virtual_clone()` to make the output equivalent to the input before.
  // modifying metadata.

  ValueRef t1 = graph.add_tensor(orig_sizes, vkapi::kFloat);
  ValueRef t2 = graph.add_tensor_view(t1);
  ValueRef t3 = graph.add_tensor_view(t2);

  ValueRef channels = graph.add_scalar<int64_t>(1);
  ValueRef height = graph.add_scalar<int64_t>(2);
  ValueRef width = graph.add_scalar<int64_t>(3);

  auto opFn = VK_GET_OP_FN("aten.transpose.int");

  opFn(graph, {t1, channels, height, t2});
  std::vector<int64_t> t2_sizes = graph.sizes_of(t2);
  std::vector<int64_t> expected_t2_sizes = {N, H, C, W};
  EXPECT_TRUE(t2_sizes == expected_t2_sizes);

  opFn(graph, {t2, height, width, t3});
  std::vector<int64_t> t3_sizes = graph.sizes_of(t3);
  std::vector<int64_t> expected_t3_sizes = {N, H, W, C};
  EXPECT_TRUE(t3_sizes == expected_t3_sizes);
}

TEST(VulkanComputeGraphTest, test_simple_graph) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {1, 8, 8};
  std::vector<int64_t> size_small = {1, 1, 8};

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

TEST(VulkanComputeGraphTest, test_simple_graph_with_symint) {
  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);

  std::vector<int64_t> sizes = {8, 64, 124};

  // Build graph

  ValueRef scalar = graph.add_symint(1);
  IOValueRef a = graph.add_input_tensor(sizes, vkapi::kFloat);

  IOValueRef out = {};
  out.value = a.value;

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR("scalar_add_texture"),
      graph.create_global_wg_size(a.value),
      graph.create_local_wg_size(a.value),
      // Inputs and Outputs
      {{out.value, vkapi::MemoryAccessType::WRITE}},
      // Shader params buffers
      {graph.logical_limits_ubo(a.value),
       graph.get_or_create_int_param_buffer(scalar)},
      // Specialization Constants
      {},
      // Resizing Logic
      nullptr,
      {}));

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_execute();

  // Run graph

  for (float i = 5.0f; i < 30.0f; i += 10.0f) {
    int scalar_val = i - 3.0f;
    graph.set_symint(scalar, scalar_val);

    int32_t scalar_val_read = graph.read_symint(scalar);
    EXPECT_TRUE(scalar_val_read == scalar_val);

    float val_a = i + 2.0f;
    float val_out = val_a + scalar_val;

    fill_vtensor(graph, a, val_a);

    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (size_t i = 0; i < graph.get_tensor(out.value)->numel(); ++i) {
      CHECK_VALUE(data_out, i, val_out);
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

  ValueRef w1_packed = graph.add_tensor(size_small, vkapi::kFloat);
  ValueRef w2_packed = graph.add_tensor(size_small, vkapi::kFloat);

  auto prepackFn = VK_GET_OP_FN("et_vk.prepack.default");
  prepackFn(graph, {w1, w1_packed});
  prepackFn(graph, {w2, w2_packed});

  auto addFn = VK_GET_OP_FN("aten.add.Tensor");
  addFn(graph, {a.value, w1_packed, kDummyValueRef, c});

  auto mulFn = VK_GET_OP_FN("aten.mul.Tensor");
  mulFn(graph, {c, w2_packed, e});

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
  size_t expected_vma_allocation_count = 0;

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
  expected_vma_allocation_count += 4;
  EXPECT_EQ(get_vma_allocation_count(), expected_vma_allocation_count);

  ValueRef c = graph.add_tensor(
      size_big,
      vkapi::kFloat,
      /*shared_object_idx = */ 6);

  auto addFn = VK_GET_OP_FN("aten.add.Tensor");
  addFn(graph, {a.value, b.value, kDummyValueRef, c});

  // no new allocations if binary op uses push constants
  EXPECT_EQ(get_vma_allocation_count(), expected_vma_allocation_count);

  IOValueRef d = graph.add_input_tensor(
      size_small,
      vkapi::kFloat,
      /*shared_object_idx = */ 2);

  // +1: t.sizes_ubo() uniform buffer for staging shader
  // +1: staging buffer for the input tensor
  expected_vma_allocation_count += 2;
  EXPECT_EQ(get_vma_allocation_count(), expected_vma_allocation_count);

  ValueRef e = graph.add_tensor(
      size_big,
      vkapi::kFloat,
      /*shared_object_idx = */ 4);

  auto mulFn = VK_GET_OP_FN("aten.mul.Tensor");
  mulFn(graph, {c, d.value, e});

  // no new allocations if binary op uses push constants
  EXPECT_EQ(get_vma_allocation_count(), expected_vma_allocation_count);

  IOValueRef out = {};
  out.value = e;
  out.staging = graph.set_output_tensor(out.value);

  // +1: staging buffer input tensor
  // +1: staging buffer for the output tensor
  expected_vma_allocation_count += 2;
  EXPECT_EQ(get_vma_allocation_count(), expected_vma_allocation_count);

  graph.prepare();
  graph.encode_execute();

  // +3: shared memory allocations for tensors
  expected_vma_allocation_count += 3;
  EXPECT_EQ(get_vma_allocation_count(), expected_vma_allocation_count);

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

TEST(VulkanComputeGraphTest, test_simple_graph_with_tmp_tensors) {
  GraphConfig config;
  ComputeGraph graph(config);

  std::vector<int64_t> size_big = {8, 64, 124};
  std::vector<int64_t> size_small = {8, 1, 124};

  // Build graph

  IOValueRef a = graph.add_input_tensor(
      size_big, vkapi::kFloat, /*shared_object_idx = */ 0);
  IOValueRef b = graph.add_input_tensor(
      size_small, vkapi::kFloat, /*shared_object_idx = */ 1);

  IOValueRef out = {};

  out.value =
      graph.add_tensor(size_big, vkapi::kFloat, /*shared_object_idx = */ 2);

  // Perform the following compute
  //
  // a, b, out;
  // {
  //   inter;
  //   {
  //     tmp = a + b
  //     tmp2 = tmp + a
  //     inter = tmp2 + b
  //   }
  //   {
  //     tmp = inter + b;
  //     tmp2 = tmp + a
  //     out = tmp2 + b;
  //   }
  // }
  {
    TmpTensor inter(&graph, size_big, vkapi::kFloat);
    EXPECT_TRUE(inter.sobj_idx == 3);
    {
      TmpTensor tmp(&graph, size_big, vkapi::kFloat);
      EXPECT_TRUE(tmp.sobj_idx == 4);
      VK_GET_OP_FN("aten.add.Tensor")
      (graph, {a, b, kDummyValueRef, tmp});

      TmpTensor tmp2(&graph, size_big, vkapi::kFloat);
      EXPECT_TRUE(tmp2.sobj_idx == 5);
      VK_GET_OP_FN("aten.add.Tensor")
      (graph, {tmp, a, kDummyValueRef, tmp2});

      VK_GET_OP_FN("aten.add.Tensor")
      (graph, {tmp2, b, kDummyValueRef, inter});
    }
    {
      TmpTensor tmp(&graph, size_big, vkapi::kFloat);
      EXPECT_TRUE(tmp.sobj_idx == 4);
      VK_GET_OP_FN("aten.add.Tensor")
      (graph, {inter, b, kDummyValueRef, tmp});

      TmpTensor tmp2(&graph, size_big, vkapi::kFloat);
      EXPECT_TRUE(tmp2.sobj_idx == 5);
      VK_GET_OP_FN("aten.add.Tensor")
      (graph, {tmp, a, kDummyValueRef, tmp2});

      VK_GET_OP_FN("aten.add.Tensor")
      (graph, {tmp2, b, kDummyValueRef, out});
    }
  }

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_execute();

  // Run graph

  for (float i = 5.0f; i < 30.0f; i += 10.0f) {
    float val_a = i + 2.0f;
    float val_b = i + 1.5f;
    float val_tmp = val_a + val_b;
    float val_tmp2 = val_tmp + val_a;
    float val_inter = val_tmp2 + val_b;
    float val_tmp_2 = val_inter + val_b;
    float val_tmp2_2 = val_tmp_2 + val_a;
    float val_out = val_tmp2_2 + val_b;

    fill_vtensor(graph, a, val_a);
    fill_vtensor(graph, b, val_b);

    graph.execute();

    EXTRACT_TENSOR(out);

    // Sanity check that the values are correct
    for (size_t i = 0; i < graph.get_tensor(out.value)->numel(); ++i) {
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

void test_clone(
    std::vector<int64_t> sizes,
    utils::StorageType src_storage,
    utils::GPUMemoryLayout src_layout,
    utils::StorageType dst_storage,
    utils::GPUMemoryLayout dst_layout) {
  GraphConfig config;
  ComputeGraph graph(config);

  IOValueRef a =
      graph.add_input_tensor(sizes, vkapi::kFloat, src_storage, src_layout);

  IOValueRef out = {};
  out.value = graph.add_tensor(sizes, vkapi::kFloat, dst_storage, dst_layout);

  auto copyFn = VK_GET_OP_FN("aten.clone.default");
  copyFn(graph, {a.value, kDummyValueRef, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_execute();

  fill_vtensor(graph, a, 0.0f, /*iota = */ true);

  graph.propagate_resize();
  graph.execute();

  EXTRACT_TENSOR(out);
  EXTRACT_TENSOR(a);

  for (int i = 0; i < graph.numel_of(a.value); ++i) {
    EXPECT_TRUE(data_out[i] == data_a[i]);
  }
}

TEST(VulkanComputeGraphTest, test_clone) {
  std::vector<std::pair<utils::GPUMemoryLayout, utils::GPUMemoryLayout>> cases{
      {utils::kWidthPacked, utils::kWidthPacked},
      {utils::kWidthPacked, utils::kChannelsPacked},
      {utils::kChannelsPacked, utils::kChannelsPacked},
  };

  for (std::vector<int64_t> sizes : standard_sizes_to_test) {
    for (auto& [src_layout, dst_layout] : cases) {
      test_clone(
          sizes, utils::kTexture3D, src_layout, utils::kBuffer, dst_layout);
      test_clone(
          sizes, utils::kBuffer, src_layout, utils::kTexture3D, dst_layout);
      test_clone(
          sizes, utils::kTexture3D, src_layout, utils::kTexture3D, dst_layout);
    }
  }
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
  if (dtype == vkapi::kHalf &&
      !context()->adapter_ptr()->supports_16bit_storage_buffers()) {
    return;
  }
  vTensor vten = vTensor(context(), sizes, dtype, storage_type, memory_layout);

  std::string kernel_name("idx_fill_texture");
  add_dtype_suffix(kernel_name, vten);

  int32_t offset = -50;

  {
    vkapi::PipelineBarrier pipeline_barrier{};
    context()->submit_compute_job(
        VK_KERNEL_FROM_STR(kernel_name),
        pipeline_barrier,
        vten.logical_limits(),
        {4, 4, 4},
        {vten.packed_dim(), offset},
        VK_NULL_HANDLE,
        0,
        vten.image(
            pipeline_barrier,
            vkapi::PipelineStage::COMPUTE,
            vkapi::MemoryAccessType::WRITE),
        vten.sizes_ubo());
  }

  StagingBuffer staging_buffer(context(), dtype, vten.staging_buffer_numel());

  if (dtype == vkapi::kChar &&
      !context()->adapter_ptr()->has_full_int8_buffers_support()) {
    record_bitw8_image_to_nchw_nobitw8buffer_op(
        context(), vten, staging_buffer);
  } else {
    record_image_to_nchw_op(context(), vten, staging_buffer.buffer());
  }

  submit_to_gpu();

  std::vector<T> data_out(staging_buffer.numel());
  staging_buffer.copy_to(data_out.data(), staging_buffer.nbytes());

  for (int i = 0; i < vten.numel(); i++) {
    CHECK_VALUE(data_out, i, i + offset);
  }
}

template <typename T>
void round_trip_test(
    std::vector<int64_t>& sizes,
    utils::GPUMemoryLayout memory_layout =
        utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
    vkapi::ScalarType dtype = vkapi::kFloat,
    utils::StorageType storage_type = utils::StorageType::TEXTURE_3D) {
  if (dtype == vkapi::kHalf &&
      !context()->adapter_ptr()->supports_16bit_storage_buffers()) {
    return;
  }

  vTensor vten = vTensor(context(), sizes, dtype, storage_type, memory_layout);

  // Create and fill input staging buffer
  StagingBuffer staging_buffer_in(
      context(), dtype, vten.staging_buffer_numel());

  std::vector<T> data_in(staging_buffer_in.numel());
  for (int i = 0; i < staging_buffer_in.numel(); i++) {
    data_in[i] = T(i * -1);
  }
  staging_buffer_in.copy_from(data_in.data(), vten.staging_buffer_nbytes());

  // Output staging buffer
  StagingBuffer staging_buffer_out(
      context(), dtype, vten.staging_buffer_numel());

  record_nchw_to_image_op(context(), staging_buffer_in.buffer(), vten);

  // Copy data in and out of the tensor
  if (dtype == vkapi::kChar &&
      !context()->adapter_ptr()->has_full_int8_buffers_support()) {
    record_bitw8_image_to_nchw_nobitw8buffer_op(
        context(), vten, staging_buffer_out);
  } else {
    record_image_to_nchw_op(context(), vten, staging_buffer_out.buffer());
  }

  // Execute command buffer
  submit_to_gpu();

  // Extract data from output staging buffer
  std::vector<T> data_out(staging_buffer_out.numel());
  staging_buffer_out.copy_to(data_out.data(), staging_buffer_out.nbytes());

  // All indices should be equal to the input data
  for (int i = 0; i < vten.numel(); i++) {
    CHECK_VALUE(data_out, i, data_in[i]);
  }
}

template <typename T>
void compute_graph_round_trip_test(
    std::vector<int64_t>& sizes,
    utils::GPUMemoryLayout memory_layout =
        utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
    vkapi::ScalarType dtype = vkapi::kFloat,
    utils::StorageType storage_type = utils::StorageType::TEXTURE_3D) {
  if (dtype == vkapi::kHalf &&
      !context()->adapter_ptr()->supports_16bit_storage_buffers()) {
    return;
  }

  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef r_tensor =
      graph.add_tensor(sizes, dtype, storage_type, memory_layout);
  ValueRef r_staging_in = graph.set_input_tensor(r_tensor);
  ValueRef r_staging_out = graph.set_output_tensor(r_tensor);

  graph.prepare();
  graph.encode_execute();

  vTensorPtr tensor = graph.get_tensor(r_tensor);

  std::vector<T> data_in(tensor->numel());
  for (int i = 0; i < data_in.size(); i++) {
    data_in[i] = T(i * -1);
  }
  graph.copy_into_staging(r_staging_in, data_in.data(), data_in.size());

  graph.execute();

  std::vector<T> data_out(tensor->staging_buffer_numel());
  graph.copy_from_staging(r_staging_out, data_out.data(), data_out.size());

  for (int i = 0; i < data_in.size(); i++) {
    CHECK_VALUE(data_out, i, data_in[i]);
  }
}

TEST(VulkanToFromGPUShaderTest, round_trip_tests) {
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
  round_trip_test<ctype>(                                            \
      sizes, utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, dtype); \
  round_trip_test<ctype>(                                            \
      sizes, utils::GPUMemoryLayout::TENSOR_WIDTH_PACKED, dtype);    \
  round_trip_test<ctype>(                                            \
      sizes, utils::GPUMemoryLayout::TENSOR_HEIGHT_PACKED, dtype);   \
  compute_graph_round_trip_test<ctype>(                              \
      sizes, utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED, dtype); \
  compute_graph_round_trip_test<ctype>(                              \
      sizes, utils::GPUMemoryLayout::TENSOR_WIDTH_PACKED, dtype);    \
  compute_graph_round_trip_test<ctype>(                              \
      sizes, utils::GPUMemoryLayout::TENSOR_HEIGHT_PACKED, dtype);

  for (auto& sizes : to_test) {
    RUN_TESTS(float, vkapi::kFloat)
    RUN_TESTS(executorch::aten::Half, vkapi::kHalf)
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
    utils::GPUMemoryLayout memory_layout) {
  GraphConfig config;
  ComputeGraph graph(config);

  IOValueRef arg2{};

  // Build graph

  IOValueRef arg1 = graph.add_input_tensor(sizes_big, dtype, memory_layout);
  arg2 = graph.add_input_tensor(sizes_small, dtype, memory_layout);

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
    float val_arg2 = i - 3.5;

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

    execute_graph_and_check_output(graph, {val_arg1, val_arg2}, {val_out});
  }
}

#define CALL_TEST_FN_FORALL_CONDITIONS(_)                   \
  _(vkapi::kFloat, utils::kTexture3D, utils::kWidthPacked)  \
  _(vkapi::kFloat, utils::kTexture3D, utils::kHeightPacked) \
  _(vkapi::kFloat, utils::kTexture3D, utils::kChannelsPacked)

#define CALL_TEST_FN_FOR_W_PACKED(_)                              \
  _(vkapi::kFloat, utils::kTexture3D, utils::kWidthPacked, false) \
  _(vkapi::kFloat, utils::kTexture3D, utils::kWidthPacked, true)  \
  _(vkapi::kFloat, utils::kBuffer, utils::kWidthPacked, false)    \
  _(vkapi::kFloat, utils::kBuffer, utils::kWidthPacked, true)

#define CALL_TEST_FN_FOR_C_PACKED(_)                                 \
  _(vkapi::kFloat, utils::kTexture3D, utils::kChannelsPacked, false) \
  _(vkapi::kFloat, utils::kTexture3D, utils::kChannelsPacked, true)  \
  _(vkapi::kFloat, utils::kBuffer, utils::kChannelsPacked, false)    \
  _(vkapi::kFloat, utils::kBuffer, utils::kChannelsPacked, true)

TEST(VulkanComputeGraphOpsTest, add_smoke_test) {
#define RUN_TESTS(dtype, storage, layout)                         \
  test_binary_op("add", {17, 21}, {17, 21}, dtype, layout);       \
  test_binary_op("add", {17, 21}, {1, 1}, dtype, layout);         \
  test_binary_op("sub", {11, 22}, {11, 22}, dtype, layout);       \
  test_binary_op("sub", {11, 22}, {11, 1}, dtype, layout);        \
  test_binary_op("add", {7, 17, 17}, {7, 17, 17}, dtype, layout); \
  test_binary_op("add", {7, 17, 17}, {7, 1, 17}, dtype, layout);  \
  test_binary_op("sub", {9, 9, 7}, {9, 9, 7}, dtype, layout);     \
  test_binary_op("sub", {9, 9, 7}, {9, 1, 1}, dtype, layout);

  CALL_TEST_FN_FORALL_CONDITIONS(RUN_TESTS);

#undef RUN_TESTS
}

void test_mm(
    int B,
    int M,
    int K,
    int N,
    vkapi::ScalarType dtype,
    utils::StorageType storage_type,
    utils::GPUMemoryLayout memory_layout,
    bool prepack = true) {
  GraphConfig config;
  config.set_storage_type_override(storage_type);
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
#define RUN_TESTS(dtype, storage_type, layout, prepack) \
  test_mm(                                              \
      /*B = */ 1,                                       \
      /*M = */ 31,                                      \
      /*K = */ 127,                                     \
      /*N = */ 23,                                      \
      dtype,                                            \
      storage_type,                                     \
      layout,                                           \
      prepack);                                         \
  test_mm(                                              \
      /*B = */ 5,                                       \
      /*M = */ 31,                                      \
      /*K = */ 127,                                     \
      /*N = */ 23,                                      \
      dtype,                                            \
      storage_type,                                     \
      layout,                                           \
      prepack);                                         \
  test_mm(                                              \
      /*B = */ 7,                                       \
      /*M = */ 13,                                      \
      /*K = */ 89,                                      \
      /*N = */ 17,                                      \
      dtype,                                            \
      storage_type,                                     \
      layout,                                           \
      prepack);                                         \
  test_mm(                                              \
      /*B = */ 1,                                       \
      /*M = */ 13,                                      \
      /*K = */ 89,                                      \
      /*N = */ 17,                                      \
      dtype,                                            \
      storage_type,                                     \
      layout,                                           \
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
  std::vector<float> input_data(t_in->staging_buffer_numel());
  graph.copy_from_staging(
      in_ioval.staging, input_data.data(), input_data.size());

  graph.execute();

  vTensorPtr t_out = graph.get_tensor(out_ioval.value);
  std::vector<float> output_data(t_out->staging_buffer_numel());
  graph.copy_from_staging(
      out_ioval.staging, output_data.data(), output_data.size());
  vTensorPtr t_idx = graph.get_tensor(idx_ioval.value);
  std::vector<int> index_data(t_idx->staging_buffer_numel());
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
  StagingBuffer staging_buffer_in(context(), vkapi::kFloat, in_numel);

  std::vector<float> data_in(in_numel);
  for (int i = 0; i < in_numel; i++) {
    data_in[i] = i + 1;
  }
  staging_buffer_in.copy_from(data_in.data(), sizeof(float) * in_numel);

  // Output staging buffer
  const int64_t out_numel =
      padded_sizes[0] * padded_sizes[1] * original_sizes[2] * original_sizes[3];
  StagingBuffer staging_buffer_out(context(), vkapi::kFloat, out_numel);

  // Copy data in and out of the tensor
  record_conv2d_prepack_weights_op(
      context(), staging_buffer_in.buffer(), vten, original_sizes, transposed);
  record_image_to_nchw_op(context(), vten, staging_buffer_out.buffer());

  // Execute command buffer
  submit_to_gpu();

  // Extract data from output staging buffer
  std::vector<float> data_out(out_numel);
  staging_buffer_out.copy_to(data_out.data(), sizeof(float) * out_numel);

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

void test_grid_priors(
    std::vector<int64_t> input_sizes,
    std::vector<int64_t> output_sizes,
    int stride,
    double offset,
    const std::vector<float>& data_out_expected) {
  GraphConfig config;
  ComputeGraph graph(config);

  // Build graph
  IOValueRef in = graph.add_input_tensor(
      input_sizes,
      vkapi::kFloat,
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  IOValueRef out;
  out.value = graph.add_tensor(
      output_sizes,
      vkapi::kFloat,
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);

  VK_GET_OP_FN("et_vk.grid_priors.default")
  (graph,
   {in.value,
    graph.add_scalar<int64_t>(stride),
    graph.add_scalar<double>(offset),
    out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_prepack();
  graph.prepack();
  graph.encode_execute();

  vTensorPtr t_in = graph.get_tensor(in.value);
  vTensorPtr t_out = graph.get_tensor(out.value);
  // Resize input
  graph.propagate_resize();

  // run graph
  graph.execute();

  std::vector<float> output_data(t_out->staging_buffer_numel());
  graph.copy_from_staging(out.staging, output_data.data(), output_data.size());

  // check results
  int h_out = utils::val_at(-2, t_out->sizes());
  int w_out = utils::val_at(-1, t_out->sizes());
  for (size_t i = 0; i < h_out; ++i) {
    for (size_t j = 0; j < w_out; ++j) {
      size_t idx_out = i * w_out + j;
      CHECK_VALUE(output_data, idx_out, data_out_expected[idx_out]);
    }
  }
}

TEST(VulkanComputeGraphOpsTest, grid_priors_test) {
  test_grid_priors(
      /*input size = */ {1, 5, 2, 3},
      /*output size = */ {6, 2},
      /*stride = */ 1,
      /*offset = */ 0.0,
      /*data_out_expected = */ {0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1});

  test_grid_priors(
      /*input size = */ {1, 5, 2, 3},
      /*output size = */ {6, 2},
      /*stride = */ 8,
      /*offset = */ 0.5,
      /*data_out_expected = */ {4, 4, 12, 4, 20, 4, 4, 12, 12, 12, 20, 12});
}

void test_transpose_view_mm(
    const int B,
    const int M,
    const int K,
    const int N,
    utils::StorageType storage_type) {
  GraphConfig config;
  config.set_storage_type_override(storage_type);
  ComputeGraph graph(config);

  std::vector<int64_t> mat1_size = {M, K};
  std::vector<int64_t> mat2_t_size = {N, K};
  std::vector<int64_t> out_size = {M, N};

  std::vector<int64_t> mat1_small_size = {M - 4, K - 3};
  std::vector<int64_t> mat2_t_small_size = {N - 1, K - 3};

  if (B > 1) {
    mat1_size.resize(3);
    mat1_size = {B, M, K};
    mat2_t_size.resize(3);
    mat2_t_size = {B, N, K};
    out_size.resize(3);
    out_size = {B, M, N};

    mat1_small_size.resize(3);
    mat1_small_size = {B, M - 4, K - 3};
    mat2_t_small_size.resize(3);
    mat2_t_small_size = {B, N - 1, K - 3};
  }

  // Build graph; use shared objects to test views of shared objects

  IOValueRef mat1 =
      graph.add_input_tensor(mat1_size, vkapi::kFloat, utils::kWidthPacked, 0);
  IOValueRef mat2_transpose = graph.add_input_tensor(
      mat2_t_size, vkapi::kFloat, utils::kWidthPacked, 1);

  ValueRef mat2 = graph.add_tensor_view(mat2_transpose.value);

  ValueRef dim0;
  ValueRef dim1;

  if (B > 1) {
    dim0 = graph.add_scalar<int64_t>(1);
    dim1 = graph.add_scalar<int64_t>(2);
  } else {
    dim0 = graph.add_scalar<int64_t>(0);
    dim1 = graph.add_scalar<int64_t>(1);
  }

  IOValueRef out;
  out.value = graph.add_tensor(out_size, vkapi::kFloat, utils::kWidthPacked, 2);

  VK_GET_OP_FN("aten.transpose.int")
  (graph, {mat2_transpose.value, dim0, dim1, mat2});
  VK_GET_OP_FN("aten.mm.default")(graph, {mat1.value, mat2, out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_prepack();
  graph.prepack();
  graph.encode_execute();

  for (int i = 1; i < 4; i++) {
    float val_mat1 = i;
    float val_mat2 = i + 1;
    float val_out = K * (val_mat1 * val_mat2);

    // Try at full size
    graph.resize_input(0, mat1_size);
    graph.resize_input(1, mat2_t_size);
    graph.propagate_resize();
    execute_graph_and_check_output(graph, {val_mat1, val_mat2}, {val_out});

    // Try at reduced sizes
    val_out = (K - 3) * (val_mat1 * val_mat2);
    graph.resize_input(0, mat1_small_size);
    graph.resize_input(1, mat2_t_small_size);
    graph.propagate_resize();
    execute_graph_and_check_output(graph, {val_mat1, val_mat2}, {val_out});
  }
}

TEST(VulkanComputeGraphOpsTest, test_transpose_with_mm) {
  for (auto storage_type : {utils::kBuffer, utils::kTexture3D}) {
    test_transpose_view_mm(2, 7, 17, 5, storage_type);
  }
}

void test_to_copy() {
  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);
  int M = 8;
  int N = 8;
  int K = 8;
  // Build graph
  IOValueRef in = graph.add_input_tensor(
      {1, M, N, K},
      vkapi::kFloat,
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);

  std::vector<float> data_in =
      create_random_float_buffer(M * N * K, -1024, 1024);
  graph.copy_into_staging(in.staging, data_in.data(), data_in.size());

  IOValueRef out;
  out.value = graph.add_tensor(
      {1, M, N, K},
      vkapi::kHalf,
      utils::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);

  auto op = VK_GET_OP_FN("aten._to_copy.default");
  op(graph,
     {in.value,
      graph.add_none(),
      graph.add_none(),
      graph.add_none(),
      graph.add_none(),
      graph.add_none(),
      graph.add_none(),
      out.value});

  out.staging = graph.set_output_tensor(out.value);

  graph.prepare();
  graph.encode_prepack();
  graph.prepack();
  graph.encode_execute();
  graph.propagate_resize();
  graph.execute();

  std::vector<torch::executor::Half> output_data(graph.numel_of(out.value));
  graph.copy_from_staging(out.staging, output_data.data(), output_data.size());

  EXPECT_EQ(data_in.size(), output_data.size());

  float mse_ex = 0.0f;
  float mse_vk = 0.0f;

  // check results
  for (size_t i = 0; i < output_data.size(); ++i) {
    float input = data_in[i];
    torch::executor::Half expected_output =
        static_cast<torch::executor::Half>(input);
    uint16_t* expected_bits = reinterpret_cast<uint16_t*>(&expected_output);
    torch::executor::Half output = output_data[i];
    uint16_t* output_bits = reinterpret_cast<uint16_t*>(&output);

    std::string msg;
    msg.reserve(64);
    msg = "input = " + std::to_string(input) + "(0b" +
        std::bitset<32>(*reinterpret_cast<uint32_t*>(&input)).to_string() +
        "), expected output = " + std::to_string(expected_output) + "(0b" +
        std::bitset<16>(*expected_bits).to_string() +
        "), recieved output = " + std::to_string(output) + "(0b" +
        std::bitset<16>(*output_bits).to_string() + ")";

    std::cout << msg << std::endl;

    // Note: Torch executor half "rounds up" when converting to fp16 whereas
    // most driver implementations of Vulkan's opFConvert() just truncates the
    // extra bits for performance (rounding introduces conditional).
    // Example:
    // INPUT F32 = 25.248 (sign{0b0}, exp{0b10000011},
    // mantissa{0b10010011111101111100111}),
    // TORCH HALF OUTPUT F16 = 25.25 (sign{0b0}, exp{0b10011},
    // mantissa{0b1001010000}),
    // VULKAN OUTPUT F16 = 25.2344 (sign{0b0}, exp{0b10011},
    // mantissa{0b1001001111})
    // Note:
    // The vulkan mantissa exactly matches the first 10
    // bits of the input 23 bit mantissa. But since the 11th bit is 1, the
    // torch half output is rounded up (essentially adding a 1).
    // Vulkan mantissa{0b1001001111} + 1 = Torch half mantissa{0b1001010000}

    EXPECT_TRUE(
        (*output_bits == *expected_bits) ||
        /*rounding error*/ ((*output_bits + 1u) == *expected_bits));
    mse_ex += std::pow(expected_output - input, 2);
    mse_vk += std::pow(output - input, 2);
  }

  mse_ex /= output_data.size();
  mse_vk /= output_data.size();
  std::cout << "========================================================="
            << std::endl;
  std::cout << "mse_ex = " << mse_ex << ", mse_vk = " << mse_vk << std::endl;
}

TEST(VulkanComputeGraphOpsTest, test_to_copy) {
  if (context()->adapter_ptr()->supports_16bit_storage_buffers()) {
    test_to_copy();
  }
}
