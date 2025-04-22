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

#include <cassert>

//
// Reference Implementations
//

at::Tensor linear_weight_int4_reference_impl(
    const at::Tensor& x,
    const at::Tensor& weights_4x2,
    const int64_t groupsize,
    const at::Tensor& scales_and_zeros,
    const int64_t inner_k_tiles) {
  const std::vector<int64_t> original_x_size(x.sizes().vec());
  const size_t ndim = original_x_size.size();
  const int64_t out_features = weights_4x2.size(0);
  const at::Tensor x_flattened = x.reshape({-1, original_x_size[ndim - 1]});
  at::Tensor out = at::_weight_int4pack_mm_for_cpu(
      x_flattened, weights_4x2, groupsize, scales_and_zeros);
  std::vector<int64_t> out_shape(
      original_x_size.begin(), original_x_size.end());
  out_shape.at(ndim - 1) = out_features;
  return out.reshape(out_shape);
}

at::Tensor unpack_weights_4x2(const at::Tensor& weights_4x2) {
  std::vector<int64_t> weights_shape(weights_4x2.sizes().vec());
  weights_shape[1] *= 2;

  at::Tensor weights_unpacked =
      at::empty(weights_shape, at::device(at::kCPU).dtype(at::kInt));

  const int64_t N = weights_unpacked.size(0);
  const int64_t K = weights_unpacked.size(1);

  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k += 2) {
      const uint8_t packed_val = weights_4x2[n][k / 2].item().to<uint8_t>();
      const uint8_t second_val = packed_val & 0x0F;
      const uint8_t first_val = (packed_val & 0xF0) >> 4;

      weights_unpacked[n][k] = int(first_val);
      weights_unpacked[n][k + 1] = int(second_val);
    }
  }

  return weights_unpacked;
}

at::Tensor dequantize_and_linear(
    const at::Tensor& x,
    const at::Tensor& weights_4x2,
    const int64_t groupsize,
    const at::Tensor& scales_and_zeros,
    const int64_t inner_k_tiles) {
  std::vector<int64_t> weights_shape(weights_4x2.sizes().vec());
  weights_shape[1] *= 2;

  at::Tensor weights_dequantized =
      at::empty(weights_shape, at::device(at::kCPU).dtype(at::kFloat));

  const int64_t N = weights_dequantized.size(0);
  const int64_t K = weights_dequantized.size(1);

  const int k_groups = K / groupsize;
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k += 2) {
      const int group_idx = k / groupsize;
      // const int scale_idx = k_groups * n + group_idx;
      const uint8_t packed_val = weights_4x2[n][k / 2].item().to<uint8_t>();
      const uint8_t second_val = packed_val & 0x0F;
      const uint8_t first_val = (packed_val & 0xF0) >> 4;

      const float scale = scales_and_zeros[group_idx][n][0].item().to<float>();
      const float zero = scales_and_zeros[group_idx][n][1].item().to<float>();

      weights_dequantized[n][k] = (float(first_val) - 8.0) * scale + zero;
      weights_dequantized[n][k + 1] = (float(second_val) - 8.0) * scale + zero;
    }
  }

  return at::linear(x, weights_dequantized);
}

//
// Test functions
//

void test_reference_linear_int4(
    const int B,
    const int M,
    const int K,
    const int N,
    const int group_size = 32,
    const int inner_k_tiles = 8) {
  assert(K % group_size == 0);

  at::Tensor x = at::rand({B, M, K}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weights_4x2 =
      at::randint(0, 256, {N, K / 2}, at::device(at::kCPU).dtype(at::kByte));
  at::Tensor weights_int = unpack_weights_4x2(weights_4x2);

  const int k_groups = K / group_size;
  at::Tensor scales_and_zeros =
      at::rand({k_groups, N, 2}, at::device(at::kCPU).dtype(at::kFloat));

  at::Tensor out = linear_weight_int4_reference_impl(
      x,
      at::_convert_weight_to_int4pack_for_cpu(weights_int, group_size),
      group_size,
      scales_and_zeros,
      inner_k_tiles);

  at::Tensor out_ref = dequantize_and_linear(
      x, weights_4x2, group_size, scales_and_zeros, inner_k_tiles);

  ASSERT_TRUE(at::allclose(out, out_ref));
}

vkcompute::vkapi::ScalarType from_at_scalartype(c10::ScalarType at_scalartype) {
  using namespace vkcompute;
  switch (at_scalartype) {
    case c10::kFloat:
      return vkapi::kFloat;
    case c10::kHalf:
      return vkapi::kHalf;
    case c10::kInt:
      return vkapi::kInt;
    case c10::kLong:
      return vkapi::kInt;
    case c10::kChar:
      return vkapi::kChar;
    case c10::kByte:
      return vkapi::kByte;
    default:
      VK_THROW("Unsupported at::ScalarType!");
  }
}

void test_vulkan_linear_int4_impl(
    const int B,
    const int M,
    const int K,
    const int N,
    const int group_size = 32,
    const int inner_k_tiles = 8,
    const vkcompute::utils::StorageType in_storage =
        vkcompute::utils::kTexture3D,
    const vkcompute::utils::StorageType out_storage =
        vkcompute::utils::kTexture3D) {
  assert(K % group_size == 0);

  at::Tensor x = at::rand({B, M, K}, at::device(at::kCPU).dtype(at::kFloat));
  at::Tensor weights_4x2 =
      at::randint(0, 256, {N, K / 2}, at::device(at::kCPU).dtype(at::kByte));

  const int k_groups = K / group_size;
  at::Tensor scales_and_zeros =
      at::rand({k_groups, N, 2}, at::device(at::kCPU).dtype(at::kFloat));

  at::Tensor weights_int = unpack_weights_4x2(weights_4x2);
  at::Tensor out_ref = linear_weight_int4_reference_impl(
      x,
      at::_convert_weight_to_int4pack_for_cpu(weights_int, group_size),
      group_size,
      scales_and_zeros,
      inner_k_tiles);

  // Build Vulkan graph
  using namespace vkcompute;

  GraphConfig config;
  config.set_storage_type_override(utils::kTexture3D);
  ComputeGraph graph(config);

#define MAKE_TENSORREF_FOR(x)              \
  ValueRef r_##x = graph.add_tensorref(    \
      x.sizes().vec(),                     \
      from_at_scalartype(x.scalar_type()), \
      x.const_data_ptr());

  MAKE_TENSORREF_FOR(weights_4x2);
  MAKE_TENSORREF_FOR(scales_and_zeros);

  IOValueRef r_x = graph.add_input_tensor(
      x.sizes().vec(), from_at_scalartype(x.scalar_type()), in_storage);

  const ValueRef r_out = graph.add_tensor(
      out_ref.sizes().vec(),
      from_at_scalartype(out_ref.scalar_type()),
      out_storage);

  VK_GET_OP_FN("et_vk.linear_weight_int4.default")
  (graph,
   {r_x.value,
    r_weights_4x2,
    graph.add_scalar<int64_t>(group_size),
    r_scales_and_zeros,
    kDummyValueRef,
    r_out});

  ValueRef staging_out = graph.set_output_tensor(r_out);

  graph.prepare();
  graph.encode_prepack();
  graph.prepack();
  graph.encode_execute();

  //
  // Run model
  //

  graph.propagate_resize();
  graph.copy_into_staging(r_x.staging, x.const_data_ptr(), x.numel());

  graph.execute();

  at::Tensor vk_out = at::empty_like(out_ref);
  graph.copy_from_staging(
      staging_out, vk_out.mutable_data_ptr(), vk_out.numel());

  ASSERT_TRUE(at::allclose(vk_out, out_ref, 1e-4, 1e-4));
}

void test_vulkan_linear_int4(
    const int B,
    const int M,
    const int K,
    const int N,
    const int group_size = 32,
    const int inner_k_tiles = 8) {
  test_vulkan_linear_int4_impl(
      B,
      M,
      K,
      N,
      group_size,
      inner_k_tiles,
      vkcompute::utils::kBuffer,
      vkcompute::utils::kBuffer);

  test_vulkan_linear_int4_impl(
      B,
      M,
      K,
      N,
      group_size,
      inner_k_tiles,
      vkcompute::utils::kTexture3D,
      vkcompute::utils::kTexture3D);
}

TEST(VulkanInt4LinearTest, test_reference_impl) {
  test_reference_linear_int4(
      /*B = */ 1,
      /*M = */ 4,
      /*K = */ 128,
      /*N = */ 32);
}

TEST(VulkanInt4LinearTest, test_vulkan_impl_small_m) {
  test_vulkan_linear_int4(
      /*B = */ 1,
      /*M = */ 4,
      /*K = */ 128,
      /*N = */ 32);

  test_vulkan_linear_int4(
      /*B = */ 1,
      /*M = */ 1,
      /*K = */ 256,
      /*N = */ 256);
}

TEST(VulkanInt4LinearTest, test_vulkan_impl_gemm) {
  test_vulkan_linear_int4(
      /*B = */ 1,
      /*M = */ 256,
      /*K = */ 256,
      /*N = */ 256);
}
