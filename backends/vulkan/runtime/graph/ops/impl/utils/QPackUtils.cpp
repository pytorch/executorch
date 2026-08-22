/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/QPackUtils.h>

namespace vkcompute {

void pack4(const uint8_t* w_ptr, uint8_t* b_ptr, uint32_t N, uint32_t K) {
  for (int32_t n = 0; n < N; n++) {
    for (int32_t k2 = 0; k2 < K / 2; k2++) {
      uint8_t src_val0 = w_ptr[n * K + k2 * 2];
      uint8_t src_val1 = w_ptr[n * K + k2 * 2 + 1];
      b_ptr[n * (K / 2) + k2] = (uint8_t(src_val1) << 4) | uint8_t(src_val0);
    }
  }
}

std::vector<uint8_t> int4mm_pack_weights(
    const std::vector<int64_t>& W_sizes,
    const uint8_t* w_ptr) {
  const int32_t N = utils::val_at(-1, W_sizes);
  const int32_t K = utils::val_at(-2, W_sizes);

  const auto numel = K * N;
  std::vector<uint8_t> w_ptr_T(numel);
  std::vector<uint8_t> b_ptr(utils::div_up(numel, 2));

  // Transpose the weights
  for (int32_t k = 0; k < K; k++) {
    for (int32_t n = 0; n < N; n++) {
      w_ptr_T[n * K + k] = w_ptr[k * N + n];
    }
  }

  // Pack two int4s into each int8
  pack4(w_ptr_T.data(), b_ptr.data(), N, K);

  return b_ptr;
}

std::vector<float> int4mm_dequantize_weights(
    const std::vector<int64_t>& W_sizes,
    const uint8_t* w_ptr,
    const uint32_t group_size,
    const float* scales_and_zeros) {
  const int64_t N = utils::val_at(-1, W_sizes);
  const int64_t K = utils::val_at(-2, W_sizes);

  std::vector<float> w_ptr_deq(K * N);
  const int k_groups = K / group_size;
  const int zeros_stride = k_groups * N;

  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      const int kb = k / group_size;
      const int scale_idx = k_groups * n + kb;
      const float scale = scales_and_zeros[scale_idx];
      const float zero =
          scales_and_zeros[scale_idx + zeros_stride] - scale * 8.0;
      w_ptr_deq[k * N + n] = w_ptr[k * N + n] * scale + zero;
    }
  }

  return w_ptr_deq;
}

} // namespace vkcompute
