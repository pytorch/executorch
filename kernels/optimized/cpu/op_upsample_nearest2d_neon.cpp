/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__aarch64__)

#include <executorch/kernels/optimized/cpu/op_upsample_nearest2d_neon.h>

#include <arm_neon.h>

#include <cstring>

namespace torch {
namespace executor {
namespace native {
namespace opt_upsample_nearest2d_internal {

void upsample_nearest2d_2x_nchw_u16(
    const uint16_t* input,
    uint16_t* output,
    const int64_t begin,
    const int64_t end,
    const int64_t input_height,
    const int64_t input_width) {
  const int64_t output_height = input_height * 2;
  const int64_t output_width = input_width * 2;
  const int64_t input_plane_size = input_height * input_width;
  const int64_t output_plane_size = output_height * output_width;

  for (int64_t plane = begin; plane < end; ++plane) {
    const uint16_t* input_plane = input + plane * input_plane_size;
    uint16_t* output_plane = output + plane * output_plane_size;

    for (int64_t h = 0; h < input_height; ++h) {
      const uint16_t* input_row = input_plane + h * input_width;
      uint16_t* output_row = output_plane + (2 * h) * output_width;
      int64_t w = 0;
      for (; w + 8 <= input_width; w += 8) {
        const uint16x8_t values = vld1q_u16(input_row + w);
        uint16x8x2_t duplicated;
        duplicated.val[0] = values;
        duplicated.val[1] = values;
        vst2q_u16(output_row + 2 * w, duplicated);
      }
      for (; w < input_width; ++w) {
        const uint16_t value = input_row[w];
        output_row[2 * w] = value;
        output_row[2 * w + 1] = value;
      }
      std::memcpy(
          output_row + output_width,
          output_row,
          output_width * sizeof(uint16_t));
    }
  }
}

} // namespace opt_upsample_nearest2d_internal
} // namespace native
} // namespace executor
} // namespace torch

#endif
