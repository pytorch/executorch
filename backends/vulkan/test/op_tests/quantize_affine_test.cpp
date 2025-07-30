/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include "test_utils.h"

#include <cassert>
#include <iostream>
#include <limits>

static inline void
_check_dims(c10::string_view name, int64_t expected, int64_t actual) {
  VK_CHECK_COND(
      expected == actual,
      name,
      " has rank ",
      actual,
      " but block_size has length ",
      expected);
}

at::Tensor quantize_affine_reference_impl(
    const at::Tensor& input_,
    const std::vector<int64_t>& block_size,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& zero_point_opt,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType out_dtype,
    c10::optional<std::string> zero_point_domain_opt = std::string("INT")) {
  constexpr float kEps = 1e-7f;

  const int64_t ndim = input_.dim();
  _check_dims("input", block_size.size(), ndim);

  VK_CHECK_COND(
      input_.scalar_type() == at::kFloat || input_.scalar_type() == at::kHalf ||
          input_.scalar_type() == at::kBFloat16,
      "Unsupported input dtype: ",
      input_.dtype());

  auto zero_point_domain =
      zero_point_domain_opt.has_value() ? *zero_point_domain_opt : "INT";

  bool has_zp = zero_point_opt.has_value();
  VK_CHECK_COND(
      has_zp || zero_point_domain == "NONE" || zero_point_domain == "",
      "zero_point must be supplied unless zero_point_domain is NONE or null");

  at::Tensor input = input_.contiguous();

  std::vector<int64_t> shape_for_reduction;
  std::vector<int64_t> reduction_dims;
  int64_t cur_dim = 0;

  auto in_sizes = input.sizes();
  for (int64_t i = 0; i < ndim; ++i) {
    const int64_t blk = block_size[i];
    const int64_t dim = in_sizes[i];

    if (blk != dim && blk > 1) {
      VK_CHECK_COND(
          dim % blk == 0,
          "Input size ",
          dim,
          " is not divisible by block_size ",
          blk,
          " at dimension ",
          i);
      shape_for_reduction.push_back(dim / blk);
      shape_for_reduction.push_back(blk);
      reduction_dims.push_back(cur_dim + 1);
      cur_dim += 2;
    } else {
      shape_for_reduction.push_back(dim);
      if (blk != 1) {
        reduction_dims.push_back(cur_dim);
      }
      cur_dim += 1;
    }
  }

  at::Tensor input_reshaped = input.view(shape_for_reduction);

  std::vector<int64_t> shape_after_reduction = shape_for_reduction;
  for (int64_t d : reduction_dims) {
    shape_after_reduction[d] = 1;
  }

  at::Tensor scale_b =
      scale.view(shape_after_reduction).to(input_reshaped.scalar_type());

  at::Tensor zp_b;
  if (has_zp) {
    zp_b = (*zero_point_opt).view(shape_after_reduction).toType(at::kFloat);
  }

  scale_b = scale_b.clamp_min(kEps);
  at::Tensor inv_scale = 1.0f / scale_b;

  at::Tensor q;
  if (zero_point_domain == "INT") {
    VK_CHECK_COND(has_zp, "INT zero_point_domain requires zero_point tensor");
    q = at::round(input_reshaped * inv_scale) + zp_b;
  } else if (zero_point_domain == "NONE" || zero_point_domain.empty()) {
    VK_CHECK_COND(
        !has_zp, "zero_point must be None when domain is NONE / null");
    q = at::round(input_reshaped * inv_scale);
  } else {
    VK_CHECK_COND(
        has_zp && zero_point_domain == "FLOAT",
        "zero_point_domain must be INT, FLOAT, NONE or null");
    const float mid_point = (quant_max + quant_min + 1) * 0.5f;
    at::Tensor min_val = zp_b - scale_b * mid_point;
    q = at::round((input_reshaped - min_val) / scale_b);
  }

  q = at::clamp(q, (double)quant_min, (double)quant_max);

  q = q.view(in_sizes).to(out_dtype);

  return q;
}

at::Tensor dequantize_affine_reference_impl(
    const at::Tensor& input_,
    const std::vector<int64_t>& block_size,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& zero_point_opt,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType out_dtype,
    c10::optional<std::string> zero_point_domain_opt = std::string("INT")) {
  const int64_t ndim = input_.dim();
  _check_dims("input", block_size.size(), ndim);

  VK_CHECK_COND(
      input_.scalar_type() == at::kByte || input_.scalar_type() == at::kChar ||
          input_.scalar_type() == at::kShort ||
          input_.scalar_type() == at::kInt,
      "Unsupported input dtype: ",
      input_.dtype());

  VK_CHECK_COND(
      out_dtype == at::kFloat || out_dtype == at::kHalf ||
          out_dtype == at::kBFloat16,
      "Unsupported output dtype: ",
      out_dtype);

  auto zero_point_domain =
      zero_point_domain_opt.has_value() ? *zero_point_domain_opt : "INT";

  bool has_zp = zero_point_opt.has_value();
  VK_CHECK_COND(
      has_zp || zero_point_domain == "NONE" || zero_point_domain == "",
      "zero_point must be supplied unless zero_point_domain is NONE or null");

  at::Tensor input = input_.contiguous();

  std::vector<int64_t> shape_for_reduction;
  std::vector<int64_t> reduction_dims;
  int64_t cur_dim = 0;

  auto in_sizes = input.sizes();
  for (int64_t i = 0; i < ndim; ++i) {
    const int64_t blk = block_size[i];
    const int64_t dim = in_sizes[i];

    if (blk != dim && blk > 1) {
      VK_CHECK_COND(
          dim % blk == 0,
          "Input size ",
          dim,
          " is not divisible by block_size ",
          blk,
          " at dimension ",
          i);
      shape_for_reduction.push_back(dim / blk);
      shape_for_reduction.push_back(blk);
      reduction_dims.push_back(cur_dim + 1);
      cur_dim += 2;
    } else {
      shape_for_reduction.push_back(dim);
      if (blk != 1) {
        reduction_dims.push_back(cur_dim);
      }
      cur_dim += 1;
    }
  }

  at::Tensor input_reshaped = input.view(shape_for_reduction);

  std::vector<int64_t> shape_after_reduction = shape_for_reduction;
  for (int64_t d : reduction_dims) {
    shape_after_reduction[d] = 1;
  }

  at::Tensor scale_b = scale.view(shape_after_reduction).to(out_dtype);

  at::Tensor zp_b;
  if (has_zp) {
    zp_b = (*zero_point_opt).view(shape_after_reduction).to(out_dtype);
  }

  at::Tensor input_fp = input_reshaped.to(out_dtype);
  at::Tensor dq;

  if (zero_point_domain == "INT") {
    VK_CHECK_COND(has_zp, "INT zero_point_domain requires zero_point tensor");
    dq = (input_fp - zp_b) * scale_b;
  } else if (zero_point_domain == "NONE" || zero_point_domain.empty()) {
    VK_CHECK_COND(
        !has_zp, "zero_point must be None when domain is NONE / null");
    dq = input_fp * scale_b;
  } else {
    VK_CHECK_COND(
        has_zp && zero_point_domain == "FLOAT",
        "zero_point_domain must be INT, FLOAT, NONE or null");
    const float mid_point = (quant_max + quant_min + 1) * 0.5f;
    at::Tensor min_val = zp_b - scale_b * mid_point;
    dq = input_fp * scale_b + min_val;
  }

  dq = dq.view(in_sizes);

  return dq;
}

// Wrapper function to maintain compatibility with existing test code (above is
// a good reference for how the python implementation works)
at::Tensor quantize_affine_reference_impl(
    const at::Tensor& input,
    const std::vector<int64_t>& block_size,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  return quantize_affine_reference_impl(
      input,
      block_size,
      scale,
      c10::optional<at::Tensor>(zero_point),
      quant_min,
      quant_max,
      dtype,
      std::string("INT"));
}

// Wrapper function for dequantize_affine
at::Tensor dequantize_affine_reference_impl(
    const at::Tensor& input,
    const std::vector<int64_t>& block_size,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType dtype) {
  return dequantize_affine_reference_impl(
      input,
      block_size,
      scale,
      c10::optional<at::Tensor>(zero_point),
      quant_min,
      quant_max,
      dtype,
      std::string("INT"));
}

void test_vulkan_quantize_affine_impl(
    const std::vector<int>& input_sizes,
    const std::vector<int64_t>& block_size,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype = at::kFloat,
    at::ScalarType dtype = at::kInt,
    const vkcompute::utils::StorageType in_storage =
        vkcompute::utils::kTexture3D,
    const vkcompute::utils::StorageType out_storage =
        vkcompute::utils::kTexture3D) {
  // Create input tensor with random values
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input =
      at::rand(input_sizes_int64, at::device(at::kCPU).dtype(in_dtype));

  // Create scale and zero_point tensors
  at::Tensor scale_tensor =
      at::tensor(scales, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor zero_point_tensor =
      at::tensor(zero_points, at::device(at::kCPU).dtype(at::kInt));

  // Get reference output
  at::Tensor reference_out = quantize_affine_reference_impl(
      input,
      block_size,
      scale_tensor,
      zero_point_tensor,
      quant_min,
      quant_max,
      dtype);

  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(in_storage);
  ComputeGraph graph(config);

  IOValueRef r_input = graph.add_input_tensor(
      input.sizes().vec(), from_at_scalartype(input.scalar_type()), in_storage);

  std::vector<int64_t> block_size_copy(block_size);
  const ValueRef r_block_size =
      graph.add_scalar_list<int64_t>(std::move(block_size_copy));

  IOValueRef r_scale = graph.add_input_tensor(
      scale_tensor.sizes().vec(),
      vkapi::kFloat,
      utils::kBuffer,
      utils::kWidthPacked);
  IOValueRef r_zero_point = graph.add_input_tensor(
      zero_point_tensor.sizes().vec(),
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked);

  const ValueRef r_output_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(dtype));
  const ValueRef r_quant_min = graph.add_scalar<int64_t>(quant_min);
  const ValueRef r_quant_max = graph.add_scalar<int64_t>(quant_max);

  const ValueRef r_out = graph.add_tensor(
      input.sizes().vec(), from_at_scalartype(dtype), out_storage);

  VK_GET_OP_FN("torchao.quantize_affine.default")
  (graph,
   {
       r_input.value,
       r_block_size,
       r_scale.value,
       r_zero_point.value,
       r_output_dtype,
       r_quant_min,
       r_quant_max,
       r_out,
   });

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.prepack();
  graph.encode_execute();

  // Copy input data to GPU
  graph.copy_into_staging(
      r_input.staging, input.const_data_ptr(), input.numel());

  // Copy scale tensor to GPU
  graph.copy_into_staging(
      r_scale.staging, scale_tensor.const_data_ptr(), scale_tensor.numel());

  // Copy zero_point tensor to GPU
  graph.copy_into_staging(
      r_zero_point.staging,
      zero_point_tensor.const_data_ptr(),
      zero_point_tensor.numel());

  // Execute the graph
  graph.execute();

  // Copy output data back to CPU
  at::Tensor vk_out = at::empty_like(reference_out).contiguous();
  graph.copy_from_staging(
      staging_out, vk_out.mutable_data_ptr(), vk_out.numel());

  // Compare outputs
  at::Tensor reference_int = reference_out.to(at::kInt);
  at::Tensor vk_int = vk_out.to(at::kInt);

  // Tolerance is 1 to address rounding errors and fp math differences between
  // CPU/GPU
  const bool output_correct =
      at::allclose(reference_int, vk_int, /*rtol=*/1, /*atol=*/1);
  if (!output_correct) {
    std::cout << "\nFailed with parameters:" << std::endl;
    std::cout << "  input_sizes: [";
    for (size_t i = 0; i < input_sizes.size(); i++) {
      std::cout << input_sizes[i] << (i < input_sizes.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    std::cout << "  block_size: [";
    for (size_t i = 0; i < block_size.size(); i++) {
      std::cout << block_size[i] << (i < block_size.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    std::cout << "  scales: [";
    for (size_t i = 0; i < scales.size(); i++) {
      std::cout << scales[i] << (i < scales.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    std::cout << "  zero_points: [";
    for (size_t i = 0; i < zero_points.size(); i++) {
      std::cout << zero_points[i] << (i < zero_points.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    std::cout << "  quant_min: " << quant_min << std::endl;
    std::cout << "  quant_max: " << quant_max << std::endl;
    std::cout << "  storage type: "
              << (in_storage == vkcompute::utils::kBuffer ? "buffer"
                                                          : "texture")
              << std::endl;

    std::cout << "input:" << std::endl << input << std::endl;
    std::cout << "reference:" << std::endl << reference_int << std::endl;
    std::cout << "vulkan:" << std::endl << vk_int << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

// Wrapper function to test both buffer and texture storage types
void test_vulkan_quantize_affine(
    const std::vector<int>& input_sizes,
    const std::vector<int64_t>& block_size,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype = at::kFloat,
    at::ScalarType dtype = at::kInt) {
  // Test with buffer storage
  test_vulkan_quantize_affine_impl(
      input_sizes,
      block_size,
      scales,
      zero_points,
      quant_min,
      quant_max,
      in_dtype,
      dtype,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  // Test with texture storage
  test_vulkan_quantize_affine_impl(
      input_sizes,
      block_size,
      scales,
      zero_points,
      quant_min,
      quant_max,
      in_dtype,
      dtype,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

TEST(VulkanQuantizeAffineTest, test_1d_quantization) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  // 1D: 1x1x1x12 Tensor, block_size is 3
  test_vulkan_quantize_affine(
      {12}, // input_sizes
      {3}, // block_size
      {0.1f, 0.2f, 0.15f, 0.25f}, // scales (4 blocks)
      {10, -20, 5, 30}, // zero_points (4 blocks)
      -128, // quant_min (char min)
      127, // quant_max (char max)
      at::kFloat, // input dtype
      at::kChar); // output dtype
}

TEST(VulkanQuantizeAffineTest, test_2d_quantization) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  // 2D: 1x1x8x6 Tensor, block_size is 1x1x2x3 (8/2=4, 6/3=2, so 4*2=8 blocks)
  test_vulkan_quantize_affine(
      {8, 6}, // input_sizes
      {2, 3}, // block_size (1/1=1, 1/1=1, 8/2=4, 6/3=2)
      {0.1f, 0.2f, 0.15f, 0.25f, 0.3f, 0.05f, 0.4f, 0.35f}, // scales (8 blocks)
      {-10, 15, 0, 25, -5, 20, 10, -15}, // zero_points (8 blocks)
      -128, // quant_min (char min)
      127, // quant_max (char max)
      at::kFloat, // input dtype
      at::kChar); // output dtype
}

TEST(VulkanQuantizeAffineTest, test_3d_quantization) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  // 3D: 1x6x4x6 Tensor, block_size is 3x2x2 (6/3=2, 4/2=2, 6/2=3, so 2*2*3=12
  // blocks)
  test_vulkan_quantize_affine(
      {6, 4, 6}, // input_sizes (changed 7->6 so divisible by 3)
      {3,
       2,
       2}, // block_size (6 divisible by 3, 4 divisible by 2, 6 divisible by 2)
      {0.1f,
       0.2f,
       0.15f,
       0.25f,
       0.3f,
       0.05f,
       0.4f,
       0.35f,
       0.12f,
       0.18f,
       0.22f,
       0.28f}, // scales (12 blocks)
      {-15, 10, 5, -25, 20, -10, 15, -5, 8, -12, 18, -8}, // zero_points (12
                                                          // blocks)
      -128, // quant_min (char min)
      127, // quant_max (char max)
      at::kFloat, // input dtype
      at::kChar); // output dtype
}

TEST(VulkanQuantizeAffineTest, test_4d_quantization) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  // 4D: 8x6x6x6 Tensor, block_size is 2x3x2x3 (8/2=4, 6/3=2, 6/2=3, 6/3=2, so
  // 4*2*3*2=48 blocks)
  test_vulkan_quantize_affine(
      {8, 6, 6, 6}, // input_sizes
      {2, 3, 2, 3}, // block_size (8/2=4, 6/3=2, 6/2=3, 6/3=2)
      {0.1f,  0.2f,  0.15f, 0.25f, 0.3f,  0.05f, 0.4f,  0.35f, 0.12f, 0.18f,
       0.22f, 0.28f, 0.32f, 0.08f, 0.45f, 0.38f, 0.14f, 0.24f, 0.16f, 0.26f,
       0.34f, 0.06f, 0.44f, 0.36f, 0.11f, 0.21f, 0.13f, 0.23f, 0.31f, 0.07f,
       0.41f, 0.37f, 0.19f, 0.29f, 0.17f, 0.27f, 0.33f, 0.09f, 0.43f, 0.39f,
       0.10f, 0.20f, 0.14f, 0.24f, 0.30f, 0.04f, 0.40f, 0.34f}, // scales (48
                                                                // blocks)
      {-20, 10,  5,   -15, 25,  -10, 15,  -5, 8,  -12, 18,  -8, 22,
       -18, 12,  -22, -25, 15,  0,   -20, 30, -5, 20,  -10, 5,  -25,
       10,  -15, 35,  -15, 25,  -35, -30, 20, -5, -25, 40,  0,  30,
       -40, 10,  -30, 15,  -10, 45,  -20, 35, -45}, // zero_points (48 blocks)
      -128, // quant_min (char min)
      127, // quant_max (char max)
      at::kFloat, // input dtype
      at::kChar); // output dtype
}

void test_vulkan_dequantize_affine_impl(
    const std::vector<int>& input_sizes,
    const std::vector<int64_t>& block_size,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype = at::kChar,
    at::ScalarType out_dtype = at::kFloat,
    const vkcompute::utils::StorageType in_storage =
        vkcompute::utils::kTexture3D,
    const vkcompute::utils::StorageType out_storage =
        vkcompute::utils::kTexture3D) {
  // Create input tensor with random integer values within quant_min and
  // quant_max
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input = at::randint(
      quant_min,
      quant_max + 1,
      input_sizes_int64,
      at::device(at::kCPU).dtype(in_dtype));

  // Create scale and zero_point tensors
  at::Tensor scale_tensor =
      at::tensor(scales, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor zero_point_tensor =
      at::tensor(zero_points, at::device(at::kCPU).dtype(at::kInt));

  // Get reference output
  at::Tensor reference_out = dequantize_affine_reference_impl(
      input,
      block_size,
      scale_tensor,
      zero_point_tensor,
      quant_min,
      quant_max,
      out_dtype);

  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(in_storage);
  ComputeGraph graph(config);

  IOValueRef r_input = graph.add_input_tensor(
      input.sizes().vec(), from_at_scalartype(input.scalar_type()), in_storage);

  // Create block_size as IntList instead of Tensor
  std::vector<int64_t> block_size_copy(block_size);
  const ValueRef r_block_size =
      graph.add_scalar_list<int64_t>(std::move(block_size_copy));

  IOValueRef r_scale = graph.add_input_tensor(
      scale_tensor.sizes().vec(),
      vkapi::kFloat,
      utils::kBuffer,
      utils::kWidthPacked);
  IOValueRef r_zero_point = graph.add_input_tensor(
      zero_point_tensor.sizes().vec(),
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked);

  // Create input_dtype scalar
  const ValueRef r_input_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(in_dtype));
  const ValueRef r_quant_min = graph.add_scalar<int64_t>(quant_min);
  const ValueRef r_quant_max = graph.add_scalar<int64_t>(quant_max);
  const ValueRef r_output_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(out_dtype));

  const ValueRef r_out = graph.add_tensor(
      input.sizes().vec(), from_at_scalartype(out_dtype), out_storage);

  // Match the argument order in dequantize_affine_impl in Dequantize.cpp:
  // input, block_size, scale, zero_point, input_dtype, quant_min, quant_max,
  // output_dtype, output
  VK_GET_OP_FN("torchao.dequantize_affine.default")
  (graph,
   {
       r_input.value,
       r_block_size,
       r_scale.value,
       r_zero_point.value,
       r_input_dtype,
       r_quant_min,
       r_quant_max,
       r_output_dtype,
       r_out,
   });

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.prepack();
  graph.encode_execute();

  // Copy input data to GPU
  graph.copy_into_staging(
      r_input.staging, input.const_data_ptr(), input.numel());

  // Copy scale tensor to GPU
  graph.copy_into_staging(
      r_scale.staging, scale_tensor.const_data_ptr(), scale_tensor.numel());

  // Copy zero_point tensor to GPU
  graph.copy_into_staging(
      r_zero_point.staging,
      zero_point_tensor.const_data_ptr(),
      zero_point_tensor.numel());

  // Execute the graph
  graph.execute();

  // Copy output data back to CPU
  at::Tensor vk_out = at::empty_like(reference_out).contiguous();
  graph.copy_from_staging(
      staging_out, vk_out.mutable_data_ptr(), vk_out.numel());

  // Compare outputs
  const bool output_correct =
      at::allclose(reference_out, vk_out, /*rtol=*/1e-5, /*atol=*/1e-5);
  if (!output_correct) {
    std::cout << "\nFailed with parameters:" << std::endl;
    std::cout << "  input_sizes: [";
    for (size_t i = 0; i < input_sizes.size(); i++) {
      std::cout << input_sizes[i] << (i < input_sizes.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    std::cout << "  block_size: [";
    for (size_t i = 0; i < block_size.size(); i++) {
      std::cout << block_size[i] << (i < block_size.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    std::cout << "  scales: [";
    for (size_t i = 0; i < scales.size(); i++) {
      std::cout << scales[i] << (i < scales.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    std::cout << "  zero_points: [";
    for (size_t i = 0; i < zero_points.size(); i++) {
      std::cout << zero_points[i] << (i < zero_points.size() - 1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
    std::cout << "  quant_min: " << quant_min << std::endl;
    std::cout << "  quant_max: " << quant_max << std::endl;
    std::cout << "  storage type: "
              << (in_storage == vkcompute::utils::kBuffer ? "buffer"
                                                          : "texture")
              << std::endl;

    std::cout << "input:" << std::endl << input << std::endl;
    std::cout << "reference:" << std::endl << reference_out << std::endl;
    std::cout << "vulkan:" << std::endl << vk_out << std::endl;
  }

  ASSERT_TRUE(output_correct);
}

// Wrapper function to test both buffer and texture storage types
void test_vulkan_dequantize_affine(
    const std::vector<int>& input_sizes,
    const std::vector<int64_t>& block_size,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int64_t quant_min,
    int64_t quant_max,
    at::ScalarType in_dtype = at::kChar,
    at::ScalarType out_dtype = at::kFloat) {
  // Test with buffer storage
  test_vulkan_dequantize_affine_impl(
      input_sizes,
      block_size,
      scales,
      zero_points,
      quant_min,
      quant_max,
      in_dtype,
      out_dtype,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  // Test with texture storage
  test_vulkan_dequantize_affine_impl(
      input_sizes,
      block_size,
      scales,
      zero_points,
      quant_min,
      quant_max,
      in_dtype,
      out_dtype,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

TEST(VulkanDequantizeAffineTest, test_1d_dequantization) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  // 1D: 1x1x1x12 Tensor, block_size is 3
  test_vulkan_dequantize_affine(
      {12}, // input_sizes
      {3}, // block_size
      {0.1f, 0.2f, 0.15f, 0.25f}, // scales (4 blocks)
      {10, -20, 5, 30}, // zero_points (4 blocks)
      -128, // quant_min (char min)
      127, // quant_max (char max)
      at::kChar, // input dtype
      at::kFloat); // output dtype
}

TEST(VulkanDequantizeAffineTest, test_2d_dequantization) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  // 2D: 1x1x8x6 Tensor, block_size is 1x1x2x3 (8/2=4, 6/3=2, so 4*2=8 blocks)
  test_vulkan_dequantize_affine(
      {8, 6}, // input_sizes
      {2, 3}, // block_size (1/1=1, 1/1=1, 8/2=4, 6/3=2)
      {0.1f, 0.2f, 0.15f, 0.25f, 0.3f, 0.05f, 0.4f, 0.35f}, // scales (8 blocks)
      {-10, 15, 0, 25, -5, 20, 10, -15}, // zero_points (8 blocks)
      -128, // quant_min (char min)
      127, // quant_max (char max)
      at::kChar, // input dtype
      at::kFloat); // output dtype
}

TEST(VulkanDequantizeAffineTest, test_3d_dequantization) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  // 3D: 1x6x4x6 Tensor, block_size is 3x2x2 (6/3=2, 4/2=2, 6/2=3, so 2*2*3=12
  // blocks)
  test_vulkan_dequantize_affine(
      {6, 4, 6}, // input_sizes (changed 7->6 so divisible by 3)
      {3,
       2,
       2}, // block_size (6 divisible by 3, 4 divisible by 2, 6 divisible by 2)
      {0.1f,
       0.2f,
       0.15f,
       0.25f,
       0.3f,
       0.05f,
       0.4f,
       0.35f,
       0.12f,
       0.18f,
       0.22f,
       0.28f}, // scales (12 blocks)
      {-15, 10, 5, -25, 20, -10, 15, -5, 8, -12, 18, -8}, // zero_points (12
                                                          // blocks)
      -128, // quant_min (char min)
      127, // quant_max (char max)
      at::kChar, // input dtype
      at::kFloat); // output dtype
}

TEST(VulkanDequantizeAffineTest, test_4d_dequantization) {
  if (!vkcompute::api::context()
           ->adapter_ptr()
           ->has_full_int8_buffers_support()) {
    GTEST_SKIP();
  }
  // 4D: 8x6x6x6 Tensor, block_size is 2x3x2x3 (8/2=4, 6/3=2, 6/2=3, 6/3=2, so
  // 4*2*3*2=48 blocks)
  test_vulkan_dequantize_affine(
      {8, 6, 6, 6}, // input_sizes
      {2, 3, 2, 3}, // block_size (8/2=4, 6/3=2, 6/2=3, 6/3=2)
      {0.1f,  0.2f,  0.15f, 0.25f, 0.3f,  0.05f, 0.4f,  0.35f, 0.12f, 0.18f,
       0.22f, 0.28f, 0.32f, 0.08f, 0.45f, 0.38f, 0.14f, 0.24f, 0.16f, 0.26f,
       0.34f, 0.06f, 0.44f, 0.36f, 0.11f, 0.21f, 0.13f, 0.23f, 0.31f, 0.07f,
       0.41f, 0.37f, 0.19f, 0.29f, 0.17f, 0.27f, 0.33f, 0.09f, 0.43f, 0.39f,
       0.10f, 0.20f, 0.14f, 0.24f, 0.30f, 0.04f, 0.40f, 0.34f}, // scales (48
                                                                // blocks)
      {-20, 10,  5,   -15, 25,  -10, 15,  -5, 8,  -12, 18,  -8, 22,
       -18, 12,  -22, -25, 15,  0,   -20, 30, -5, 20,  -10, 5,  -25,
       10,  -15, 35,  -15, 25,  -35, -30, 20, -5, -25, 40,  0,  30,
       -40, 10,  -30, 15,  -10, 45,  -20, 35, -45}, // zero_points (48 blocks)
      -128, // quant_min (char min)
      127, // quant_max (char max)
      at::kChar, // input dtype
      at::kFloat); // output dtype
}
