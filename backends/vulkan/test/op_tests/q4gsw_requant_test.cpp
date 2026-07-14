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

// Pack [N, K] codes (0..15) into the W_4X8 block-packed int buffer the forward
// reads. Mirrors pack_q4_linear_weight__w_4x8.glsl: one ivec4 per (k4, n8),
// byte b holds an (even-N low nibble, odd-N high nibble) pair at K = k4*4 + b.
std::vector<int32_t> pack_codes_w4x8(const at::Tensor& codes) {
  const int64_t N = codes.size(0);
  const int64_t K = codes.size(1);
  const int64_t K4 = K / 4;
  const int64_t N4 = N / 4;
  const int64_t N4_padded = (N4 + 1) & ~int64_t{1};
  const int64_t N8 = N4_padded / 2;
  std::vector<int32_t> buf(K4 * N4_padded * 2, 0);
  auto ca = codes.accessor<int, 2>();

  auto pack_tile = [&](int64_t k4, int64_t n4, uint32_t& px, uint32_t& py) {
    px = 0u;
    py = 0u;
    for (int ni = 0; ni < 4; ++ni) {
      const int64_t n = n4 * 4 + ni;
      for (int b = 0; b < 4; ++b) {
        const uint32_t code = static_cast<uint32_t>(ca[n][k4 * 4 + b] & 0xF);
        const int shift = 8 * b + (ni & 1) * 4;
        if (ni < 2) {
          px |= code << shift;
        } else {
          py |= code << shift;
        }
      }
    }
  };

  for (int64_t k4 = 0; k4 < K4; ++k4) {
    for (int64_t n8 = 0; n8 < N8; ++n8) {
      const int64_t n4_a = 2 * n8;
      const int64_t n4_b = n4_a + 1;
      uint32_t px_a, py_a, px_b = 0x88888888u, py_b = 0x88888888u;
      pack_tile(k4, n4_a, px_a, py_a);
      if (n4_b < N4) {
        pack_tile(k4, n4_b, px_b, py_b);
      }
      const int64_t base = (k4 * N8 + n8) * 4;
      buf[base + 0] = static_cast<int32_t>(px_a);
      buf[base + 1] = static_cast<int32_t>(py_a);
      buf[base + 2] = static_cast<int32_t>(px_b);
      buf[base + 3] = static_cast<int32_t>(py_b);
    }
  }
  return buf;
}

//
// Test function
//

void test_vulkan_q4gsw_requant_impl(
    const int64_t N,
    const int64_t K,
    const int64_t group_size,
    const bool with_zero_scale) {
  const int64_t num_groups = K / group_size;

  at::Tensor scales =
      at::rand({num_groups, N}, at::device(at::kCPU).dtype(at::kFloat)) + 0.5;
  if (with_zero_scale) {
    scales.index_put_({0, 0}, 0.0);
  }

  const at::Tensor group_idx =
      at::arange(K, at::device(at::kCPU).dtype(at::kLong))
          .div(group_size, "floor");
  const at::Tensor scale_full =
      scales.t().contiguous().index_select(1, group_idx); // [N, K]

  // Deterministic quotient targets, each >=0.2 from any .5 tie, so GPU fp32
  // division (~2.5 ULP, not correctly rounded) and the CPU golden round
  // identically. Covers round both directions and clamp past [-8, 7].
  const std::vector<float> pattern = {
      0.3f, -0.4f, 2.7f, -3.3f, 6.4f, -6.4f, 13.2f, -21.7f};
  const at::Tensor pat =
      at::tensor(pattern, at::device(at::kCPU).dtype(at::kFloat));
  const at::Tensor q_idx =
      at::arange(N * K, at::device(at::kCPU).dtype(at::kLong))
          .remainder(static_cast<int64_t>(pattern.size()));
  const at::Tensor target_q = pat.index_select(0, q_idx).reshape({N, K});
  at::Tensor latent = target_q * scale_full;

  // Golden codes, mirroring quant_nibble: q=0 where scale==0, else roundEven
  // (matches at::round half-to-even); clamp to [-8, 7]; code = (q + 8) & 0xF.
  const at::Tensor nonzero = scale_full != 0;
  const at::Tensor safe =
      at::where(nonzero, scale_full, at::ones_like(scale_full));
  at::Tensor q = at::round(latent / safe);
  q = at::where(nonzero, q, at::zeros_like(q));
  q = at::clamp(q, -8, 7);
  const at::Tensor golden_codes =
      (q.to(at::kInt) + 8).bitwise_and(0xF); // [N, K] in 0..15

  const std::vector<int32_t> expected = pack_codes_w4x8(golden_codes);

  using namespace vkcompute;

  GraphConfig config;
  ComputeGraph graph(config);

  IOValueRef r_latent = graph.add_input_tensor(
      latent.sizes().vec(),
      from_at_scalartype(latent.scalar_type()),
      utils::kBuffer);
  ValueRef r_scales = graph.add_tensorref(
      scales.sizes().vec(),
      from_at_scalartype(scales.scalar_type()),
      scales.const_data_ptr());
  const ValueRef r_group_size = graph.add_scalar<int64_t>(group_size);

  const int64_t N4 = N / 4;
  const int64_t N4_padded = (N4 + 1) & ~int64_t{1};
  const ValueRef r_packed = graph.add_tensor(
      {(K / 4) * N4_padded * 2}, vkapi::kInt, utils::kBuffer);

  VK_GET_OP_FN("et_vk.q4gsw_requant.default")
  (graph, {r_latent.value, r_scales, r_group_size, r_packed});

  ValueRef staging_out = graph.set_output_tensor(r_packed);

  graph.prepare();
  graph.prepack();
  graph.propagate_resize();

  graph.maybe_cast_and_copy_into_staging(
      r_latent.staging,
      latent.const_data_ptr(),
      latent.numel(),
      from_at_scalartype(latent.scalar_type()));

  graph.execute();

  at::Tensor vk_packed = at::empty(
      {static_cast<int64_t>(expected.size())},
      at::device(at::kCPU).dtype(at::kInt));
  graph.maybe_cast_and_copy_from_staging(
      staging_out,
      vk_packed.mutable_data_ptr(),
      vk_packed.numel(),
      from_at_scalartype(vk_packed.scalar_type()));

  auto va = vk_packed.accessor<int, 1>();
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(va[i], expected[i]) << "mismatch at packed int " << i;
  }
}

// Tile-aligned single-group.
TEST(VulkanQ4gswRequantTest, test_tile_aligned) {
  test_vulkan_q4gsw_requant_impl(
      /*N=*/16, /*K=*/32, /*group_size=*/32, /*with_zero_scale=*/false);
}

// Multiple quantization groups along K.
TEST(VulkanQ4gswRequantTest, test_grouped) {
  test_vulkan_q4gsw_requant_impl(
      /*N=*/32, /*K=*/64, /*group_size=*/32, /*with_zero_scale=*/false);
}

// N not a multiple of 8 (odd N4 -> padded stride + bias-zero OOB tile).
TEST(VulkanQ4gswRequantTest, test_odd_n4) {
  test_vulkan_q4gsw_requant_impl(
      /*N=*/12, /*K=*/16, /*group_size=*/16, /*with_zero_scale=*/false);
}

// A zero scale must produce the bias-zero code (8), not a divide-by-zero.
TEST(VulkanQ4gswRequantTest, test_zero_scale) {
  test_vulkan_q4gsw_requant_impl(
      /*N=*/16, /*K=*/32, /*group_size=*/32, /*with_zero_scale=*/true);
}
