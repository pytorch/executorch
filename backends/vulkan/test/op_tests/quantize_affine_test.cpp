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

std::tuple<at::Tensor, at::Tensor> choose_qparams_affine_reference_impl(
    const at::Tensor& input_,
    const std::string& mapping_type,
    const std::vector<int64_t>& block_size,
    int64_t quant_min,
    int64_t quant_max,
    double eps) {
  const int64_t ndim = input_.dim();
  _check_dims("input", block_size.size(), ndim);

  VK_CHECK_COND(
      input_.scalar_type() == at::kFloat || input_.scalar_type() == at::kHalf ||
          input_.scalar_type() == at::kBFloat16,
      "Unsupported input dtype: ",
      input_.dtype());

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

  at::Tensor min_val = input_reshaped.amin(reduction_dims, /*keepdim=*/true);
  at::Tensor max_val = input_reshaped.amax(reduction_dims, /*keepdim=*/true);

  at::Tensor scale, zero_point;

  if (mapping_type == "ASYMMETRIC") {
    // Include zero in the range
    min_val = at::minimum(min_val, at::zeros_like(min_val));
    max_val = at::maximum(max_val, at::zeros_like(max_val));

    // Calculate scale
    scale = (max_val - min_val) / (quant_max - quant_min);
    scale = at::maximum(scale, at::full_like(scale, eps));

    // Calculate zero_point
    zero_point = at::round(quant_min - min_val / scale);
    zero_point = at::clamp(zero_point, quant_min, quant_max);
  } else if (mapping_type == "SYMMETRIC") {
    // Include zero in the range
    min_val = at::minimum(min_val, at::zeros_like(min_val));
    max_val = at::maximum(max_val, at::zeros_like(max_val));

    // Calculate max absolute value
    at::Tensor abs_min = at::abs(min_val);
    at::Tensor abs_max = at::abs(max_val);
    at::Tensor M = at::maximum(abs_min, abs_max);

    // Calculate scale
    scale = M / ((quant_max - quant_min) * 0.5);
    scale = at::maximum(scale, at::full_like(scale, eps));

    // Calculate zero_point (mid-point)
    zero_point =
        at::full_like(scale, (quant_max + quant_min + 1) / 2, at::kInt);
  } else if (mapping_type == "SYMMETRIC_NO_CLIPPING_ERR") {
    // Include zero in the range
    min_val = at::minimum(min_val, at::zeros_like(min_val));
    max_val = at::maximum(max_val, at::zeros_like(max_val));

    // Calculate scale based on min/max values
    at::Tensor s_min = at::abs(min_val) / std::abs(quant_min);
    at::Tensor s_max = max_val / quant_max;
    scale = at::maximum(s_min, s_max);
    scale = at::maximum(scale, at::full_like(scale, eps));

    // Calculate zero_point (mid-point)
    zero_point =
        at::full_like(scale, (quant_max + quant_min + 1) / 2, at::kInt);
  } else {
    VK_CHECK_COND(
        false,
        "Unsupported mapping_type: ",
        mapping_type,
        ". Expected ASYMMETRIC, SYMMETRIC, or SYMMETRIC_NO_CLIPPING_ERR");
  }

  std::vector<int64_t> output_shape;
  for (size_t i = 0; i < shape_after_reduction.size(); ++i) {
    if (shape_after_reduction[i] != 1 ||
        std::find(reduction_dims.begin(), reduction_dims.end(), i) ==
            reduction_dims.end()) {
      output_shape.push_back(shape_after_reduction[i]);
    }
  }

  // Reshape scale and zero_point to final output shape
  scale = scale.view(output_shape);
  zero_point = zero_point.view(output_shape);

  return std::make_tuple(scale, zero_point);
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

void test_vulkan_choose_qparams_affine_impl(
    const std::vector<int>& input_sizes,
    const std::vector<int64_t>& block_size,
    const std::string& mapping_type,
    int64_t quant_min,
    int64_t quant_max,
    double eps,
    at::ScalarType in_dtype = at::kFloat,
    const vkcompute::utils::StorageType in_storage =
        vkcompute::utils::kTexture3D,
    const vkcompute::utils::StorageType out_storage =
        vkcompute::utils::kBuffer) {
  // Create input tensor with random values
  std::vector<int64_t> input_sizes_int64(
      input_sizes.begin(), input_sizes.end());
  at::Tensor input =
      at::rand(input_sizes_int64, at::device(at::kCPU).dtype(in_dtype));

  // Get reference output
  auto reference_out = choose_qparams_affine_reference_impl(
      input, mapping_type, block_size, quant_min, quant_max, eps);

  at::Tensor reference_scale = std::get<0>(reference_out);
  at::Tensor reference_zero_point = std::get<1>(reference_out);

  reference_zero_point = reference_zero_point.to(at::kInt);

  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(in_storage);
  ComputeGraph graph(config);

  IOValueRef r_input = graph.add_input_tensor(
      input.sizes().vec(), from_at_scalartype(input.scalar_type()), in_storage);

  // Create mapping_type as string
  std::string mapping_type_copy = mapping_type;
  const ValueRef r_mapping_type =
      graph.add_string(std::move(mapping_type_copy));

  // Create block_size as IntList
  std::vector<int64_t> block_size_copy(block_size);
  const ValueRef r_block_size =
      graph.add_scalar_list<int64_t>(std::move(block_size_copy));

  // Create target_dtype, quant_min, quant_max, eps
  const ValueRef r_target_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(at::kChar));
  const ValueRef r_quant_min = graph.add_scalar<int64_t>(quant_min);
  const ValueRef r_quant_max = graph.add_scalar<int64_t>(quant_max);
  const ValueRef r_eps = graph.add_scalar<double>(eps);

  // Create scale_dtype and zero_point_dtype
  const ValueRef r_scale_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(at::kFloat));
  const ValueRef r_zero_point_dtype =
      graph.add_scalar<int64_t>(static_cast<int64_t>(at::kInt));

  // Create output tuple
  std::vector<ValueRef> out_tuple;

  // Create scale and zero_point output tensors
  const ValueRef r_scale_out = graph.add_tensor(
      reference_scale.sizes().vec(), vkapi::kFloat, out_storage);
  const ValueRef r_zero_point_out = graph.add_tensor(
      reference_zero_point.sizes().vec(), vkapi::kInt, out_storage);

  out_tuple.push_back(r_scale_out);
  out_tuple.push_back(r_zero_point_out);

  const ValueRef r_out_tuple = graph.add_value_list(std::move(out_tuple));

  VK_GET_OP_FN("torchao.choose_qparams_affine.default")
  (graph,
   {
       r_input.value,
       r_mapping_type,
       r_block_size,
       r_target_dtype,
       r_quant_min,
       r_quant_max,
       r_eps,
       r_scale_dtype,
       r_zero_point_dtype,
       r_out_tuple,
   });

  ValueRef staging_scale = graph.set_output_tensor(r_scale_out);
  ValueRef staging_zero_point = graph.set_output_tensor(r_zero_point_out);

  graph.prepare();
  graph.prepack();

  // Copy input data to GPU
  graph.copy_into_staging(
      r_input.staging, input.const_data_ptr(), input.numel());

  // Execute the graph
  graph.execute();

  // Copy output data back to CPU
  at::Tensor vk_scale = at::empty_like(reference_scale).contiguous();
  at::Tensor vk_zero_point = at::empty_like(reference_zero_point).contiguous();

  graph.copy_from_staging(
      staging_scale, vk_scale.mutable_data_ptr(), vk_scale.numel());
  graph.copy_from_staging(
      staging_zero_point,
      vk_zero_point.mutable_data_ptr(),
      vk_zero_point.numel());

  // Compare outputs
  const bool scale_correct =
      at::allclose(reference_scale, vk_scale, /*rtol=*/1e-3, /*atol=*/1e-3);

  // For zero point, we need to compare as integers since zero point should be
  // an integer First convert both tensors to int if they aren't already
  at::Tensor ref_zp_int = reference_zero_point.to(at::kInt);
  at::Tensor vk_zp_int = vk_zero_point.to(at::kInt);
  const bool zero_point_correct = at::equal(ref_zp_int, vk_zp_int);

  if (!scale_correct || !zero_point_correct) {
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
    std::cout << "  mapping_type: " << mapping_type << std::endl;
    std::cout << "  quant_min: " << quant_min << std::endl;
    std::cout << "  quant_max: " << quant_max << std::endl;
    std::cout << "  eps: " << eps << std::endl;
    std::cout << "  storage type: "
              << (in_storage == vkcompute::utils::kBuffer ? "buffer"
                                                          : "texture")
              << std::endl;

    if (!scale_correct || !zero_point_correct) {
      std::cout << "input:" << std::endl;
      std::cout << input << std::endl;

      std::cout << "reference_scale:" << std::endl
                << reference_scale << std::endl;
      std::cout << "vulkan_scale:" << std::endl << vk_scale << std::endl;

      std::cout << "reference_zero_point:" << std::endl
                << reference_zero_point << std::endl;
      std::cout << "vulkan_zero_point:" << std::endl
                << vk_zero_point << std::endl;
    }
  }

  ASSERT_TRUE(scale_correct);
  ASSERT_TRUE(zero_point_correct);
}

// Wrapper function to test both buffer and texture storage types
void test_vulkan_choose_qparams_affine(
    const std::vector<int>& input_sizes,
    const std::vector<int64_t>& block_size,
    const std::string& mapping_type,
    int64_t quant_min,
    int64_t quant_max,
    double eps,
    at::ScalarType in_dtype = at::kFloat) {
  // Test with buffer storage for both input and output
  test_vulkan_choose_qparams_affine_impl(
      input_sizes,
      block_size,
      mapping_type,
      quant_min,
      quant_max,
      eps,
      in_dtype,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  // Test with texture storage for input and buffer storage for output
  // (shader always uses buffer storage for outputs)
  test_vulkan_choose_qparams_affine_impl(
      input_sizes,
      block_size,
      mapping_type,
      quant_min,
      quant_max,
      eps,
      in_dtype,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kBuffer);
}

TEST(VulkanChooseQParamsAffineTest, test_1d_asymmetric) {
  // 1D: 12 Tensor, block_size is 3
  test_vulkan_choose_qparams_affine(
      {12}, // input_sizes
      {3}, // block_size
      "ASYMMETRIC", // mapping_type
      -128, // quant_min (char min)
      127, // quant_max (char max)
      1e-5, // eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_2d_symmetric) {
  // 2D: 8x6 Tensor, block_size is 2x3
  test_vulkan_choose_qparams_affine(
      {8, 6}, // input_sizes
      {2, 3}, // block_size
      "SYMMETRIC", // mapping_type
      -128, // quant_min (char min)
      127, // quant_max (char max)
      1e-5, // eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_3d_symmetric_no_clipping) {
  // 3D: 6x4x6 Tensor, block_size is 3x2x2
  test_vulkan_choose_qparams_affine(
      {6, 4, 6}, // input_sizes
      {3, 2, 2}, // block_size
      "SYMMETRIC_NO_CLIPPING_ERR", // mapping_type
      -128, // quant_min (char min)
      127, // quant_max (char max)
      1e-5, // eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_4d_asymmetric) {
  // 4D: 4x6x6x6 Tensor, block_size is 2x3x2x3
  test_vulkan_choose_qparams_affine(
      {4, 6, 6, 6}, // input_sizes (reduced from 8 to 4 to make test faster)
      {2, 3, 2, 3}, // block_size
      "ASYMMETRIC", // mapping_type
      -128, // quant_min (char min)
      127, // quant_max (char max)
      1e-5, // eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_per_tensor) {
  // Per-tensor: block_size equals tensor size
  test_vulkan_choose_qparams_affine(
      {4, 6, 8}, // input_sizes
      {4, 6, 8}, // block_size equals tensor size
      "ASYMMETRIC", // mapping_type
      -128, // quant_min (char min)
      127, // quant_max (char max)
      1e-5, // eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_per_token) {
  // Per-token: block_size is all 1s except last dimension
  test_vulkan_choose_qparams_affine(
      {4, 6, 8}, // input_sizes
      {1, 1, 8}, // block_size is all 1s except last dimension
      "ASYMMETRIC", // mapping_type
      -128, // quant_min (char min)
      127, // quant_max (char max)
      1e-5, // eps
      at::kFloat); // input dtype
}

// Additional tests for choose_qparams_affine

TEST(VulkanChooseQParamsAffineTest, test_uint8_range) {
  // Test with uint8 range (0-255)
  test_vulkan_choose_qparams_affine(
      {6, 8}, // input_sizes
      {2, 4}, // block_size
      "ASYMMETRIC", // mapping_type
      0, // quant_min (uint8 min)
      255, // quant_max (uint8 max)
      1e-5, // eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_int16_range) {
  // Test with int16 range (-32768 to 32767)
  test_vulkan_choose_qparams_affine(
      {6, 8}, // input_sizes
      {2, 4}, // block_size
      "SYMMETRIC", // mapping_type
      -32768, // quant_min (int16 min)
      32767, // quant_max (int16 max)
      1e-5, // eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_larger_eps) {
  // Test with larger epsilon value
  test_vulkan_choose_qparams_affine(
      {6, 8}, // input_sizes
      {2, 4}, // block_size
      "ASYMMETRIC", // mapping_type
      -128, // quant_min
      127, // quant_max
      1e-2, // larger eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_per_channel_first_dim) {
  // Per-channel quantization on first dimension
  test_vulkan_choose_qparams_affine(
      {8, 6, 4}, // input_sizes
      {1, 6, 4}, // block_size (per-channel on dim 0)
      "SYMMETRIC", // mapping_type
      -128, // quant_min
      127, // quant_max
      1e-5, // eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_per_channel_middle_dim) {
  // Per-channel quantization on middle dimension
  test_vulkan_choose_qparams_affine(
      {4, 8, 6}, // input_sizes
      {4, 1, 6}, // block_size (per-channel on dim 1)
      "SYMMETRIC", // mapping_type
      -128, // quant_min
      127, // quant_max
      1e-5, // eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_mixed_block_sizes) {
  // Mixed block sizes (some dimensions fully quantized, some partially)
  test_vulkan_choose_qparams_affine(
      {8, 6, 10}, // input_sizes
      {4, 6, 2}, // block_size (mixed: partial, full, partial)
      "ASYMMETRIC", // mapping_type
      -128, // quant_min
      127, // quant_max
      1e-5, // eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_small_tensor) {
  // Test with a small tensor
  test_vulkan_choose_qparams_affine(
      {2, 3}, // small input_sizes
      {2, 3}, // block_size (full tensor)
      "ASYMMETRIC", // mapping_type
      -128, // quant_min
      127, // quant_max
      1e-5, // eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_asymmetric_narrow_range) {
  // Test with a narrow quantization range
  test_vulkan_choose_qparams_affine(
      {6, 8}, // input_sizes
      {2, 4}, // block_size
      "ASYMMETRIC", // mapping_type
      -10, // quant_min (narrow range)
      10, // quant_max (narrow range)
      1e-5, // eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_symmetric_narrow_range) {
  // Test with a narrow quantization range with symmetric mapping
  test_vulkan_choose_qparams_affine(
      {6, 8}, // input_sizes
      {2, 4}, // block_size
      "SYMMETRIC", // mapping_type
      -10, // quant_min (narrow range)
      10, // quant_max (narrow range)
      1e-5, // eps
      at::kFloat); // input dtype
}

TEST(VulkanChooseQParamsAffineTest, test_symmetric_no_clipping_narrow_range) {
  // Test with a narrow quantization range with symmetric no clipping mapping
  test_vulkan_choose_qparams_affine(
      {6, 8}, // input_sizes
      {2, 4}, // block_size
      "SYMMETRIC_NO_CLIPPING_ERR", // mapping_type
      -10, // quant_min (narrow range)
      10, // quant_max (narrow range)
      1e-5, // eps
      at::kFloat); // input dtype
}