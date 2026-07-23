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

//
// Reference Implementation
//

// Pack unpacked [N, K] codes (0..15) into the flat [N, K/2] uint8 weight the
// forward's prepack consumes: even-K in the low nibble, odd-K in the high.
at::Tensor pack_codes_flat(const at::Tensor& codes) {
  const int64_t N = codes.size(0);
  const int64_t K = codes.size(1);
  at::Tensor packed =
      at::empty({N, K / 2}, at::device(at::kCPU).dtype(at::kByte));
  auto ca = codes.accessor<int, 2>();
  auto pa = packed.accessor<uint8_t, 2>();
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t kb = 0; kb < K / 2; ++kb) {
      const int lo = ca[n][2 * kb] & 0xF;
      const int hi = ca[n][2 * kb + 1] & 0xF;
      pa[n][kb] = static_cast<uint8_t>(lo | (hi << 4));
    }
  }
  return packed;
}

// Golden d_x[M, K] = d_out[M, N] @ dequant(W)[N, K], with
// dequant(W[n, k]) = (code(n, k) - 8) * scales[k / group_size, n].
// Mirrors the CPU-eager linear_q4gsw_backward_impl in custom_ops_lib.py.
at::Tensor linear_q4gsw_backward_reference_impl(
    const at::Tensor& d_out,
    const at::Tensor& codes,
    const at::Tensor& scales,
    const int64_t group_size) {
  const int64_t N = codes.size(0);
  const int64_t K = codes.size(1);
  const at::Tensor group_idx =
      at::arange(K, at::device(at::kCPU).dtype(at::kLong))
          .div(group_size, "floor");
  const at::Tensor scale_full =
      scales.t().contiguous().index_select(1, group_idx); // [N, K]
  const at::Tensor dequant_w =
      (codes.to(at::kFloat) - 8.0) * scale_full; // [N, K]
  const at::Tensor d_x_flat = d_out.reshape({-1, N}).matmul(dequant_w);
  std::vector<int64_t> out_shape = d_out.sizes().vec();
  out_shape.back() = K;
  return d_x_flat.reshape(out_shape).contiguous(); // d_out[..., :-1] + [K]
}

//
// Test function
//

void test_vulkan_linear_q4gsw_backward_impl(
    const std::vector<int64_t>& d_out_sizes,
    const int64_t K,
    const int64_t group_size) {
  const int64_t N = d_out_sizes.back();
  const int64_t num_groups = K / group_size;

  at::Tensor codes =
      at::randint(0, 16, {N, K}, at::device(at::kCPU).dtype(at::kInt));
  at::Tensor scales =
      at::rand({num_groups, N}, at::device(at::kCPU).dtype(at::kFloat)) + 0.5;
  at::Tensor packed = pack_codes_flat(codes);
  at::Tensor d_out =
      at::rand(d_out_sizes, at::device(at::kCPU).dtype(at::kFloat));

  at::Tensor d_x_ref =
      linear_q4gsw_backward_reference_impl(d_out, codes, scales, group_size);

  using namespace vkcompute;

  GraphConfig config;
  ComputeGraph graph(config);

  ValueRef r_weights = graph.add_tensorref(
      packed.sizes().vec(),
      from_at_scalartype(packed.scalar_type()),
      packed.const_data_ptr());
  ValueRef r_scales = graph.add_tensorref(
      scales.sizes().vec(),
      from_at_scalartype(scales.scalar_type()),
      scales.const_data_ptr());

  IOValueRef r_d_out = graph.add_input_tensor(
      d_out.sizes().vec(),
      from_at_scalartype(d_out.scalar_type()),
      utils::kBuffer);
  const ValueRef r_group_size = graph.add_scalar<int64_t>(group_size);
  const ValueRef r_d_x = graph.add_tensor(
      d_x_ref.sizes().vec(),
      from_at_scalartype(d_x_ref.scalar_type()),
      utils::kBuffer);

  VK_GET_OP_FN("et_vk.linear_q4gsw_backward.default")
  (graph, {r_d_out.value, r_weights, r_scales, r_group_size, r_d_x});

  ValueRef staging_out = graph.set_output_tensor(r_d_x);

  graph.prepare();
  graph.prepack();
  graph.propagate_resize();

  graph.maybe_cast_and_copy_into_staging(
      r_d_out.staging,
      d_out.const_data_ptr(),
      d_out.numel(),
      from_at_scalartype(d_out.scalar_type()));

  graph.execute();

  at::Tensor vk_d_x = at::empty_like(d_x_ref);
  graph.maybe_cast_and_copy_from_staging(
      staging_out,
      vk_d_x.mutable_data_ptr(),
      vk_d_x.numel(),
      from_at_scalartype(vk_d_x.scalar_type()));

  ASSERT_TRUE(at::allclose(vk_d_x, d_x_ref, 1e-3, 1e-3));
}

// Tile-aligned single-group shapes.
TEST(VulkanLinearQ4gswBackwardTest, test_tile_aligned) {
  test_vulkan_linear_q4gsw_backward_impl(
      /*d_out_sizes=*/{8, 16}, /*K=*/32, /*group_size=*/32);
}

// Multiple quantization groups along K.
TEST(VulkanLinearQ4gswBackwardTest, test_grouped) {
  test_vulkan_linear_q4gsw_backward_impl(
      /*d_out_sizes=*/{8, 32}, /*K=*/64, /*group_size=*/32);
}

// N not a multiple of 8 (odd N4 -> padded W_4X8 stride) plus partial-M tile.
TEST(VulkanLinearQ4gswBackwardTest, test_odd_n4_partial_m) {
  test_vulkan_linear_q4gsw_backward_impl(
      /*d_out_sizes=*/{5, 12}, /*K=*/16, /*group_size=*/16);
}

// Leading dims > 2D: M is the flattened product of all leading dims.
TEST(VulkanLinearQ4gswBackwardTest, test_leading_dims_flatten) {
  test_vulkan_linear_q4gsw_backward_impl(
      /*d_out_sizes=*/{2, 3, 16}, /*K=*/32, /*group_size=*/32);
}
