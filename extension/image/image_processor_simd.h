/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <executorch/extension/image/image_processor_config.h>
#include <executorch/runtime/core/error.h>

namespace executorch {
namespace extension {
namespace image {

// SIMD-accelerated image-processing kernels (NEON on ARM, scalar fallback
// elsewhere), shared by the Apple and portable ImageProcessor backends.

// Deinterleave an 8-bit interleaved image into planar CHW float with a fused
// per-channel affine normalize:
//   out = pixel * (scale_factor / std_dev[c]) + (-mean[c] / std_dev[c]).
// Uses NEON (vld4q_u8 / vld3q_u8 + FMA) on ARM, scalar elsewhere.
//
// in_channels is 3 (RGB) or 4 (BGRA/RGBA; the alpha byte is ignored).
// r_off/g_off/b_off are the byte offsets of R, G, B within a pixel
// (BGRA -> {2, 1, 0}, RGB/RGBA -> {0, 1, 2}); they also index the deinterleaved
// channels, so each must be < in_channels. norm.{mean,std_dev} are in RGB
// order.
//
// Writes a src_w x src_h region at (offset_x, offset_y) into the final_w x
// final_h planes; pixels outside that region are left untouched, so callers
// that letterbox must pre-fill the padding. src_stride is in bytes.
runtime::Error deinterleave_to_chw(
    const uint8_t* src,
    int32_t src_w,
    int32_t src_h,
    int32_t src_stride,
    int32_t in_channels,
    int32_t r_off,
    int32_t g_off,
    int32_t b_off,
    float* output,
    int32_t final_w,
    int32_t final_h,
    int32_t offset_x,
    int32_t offset_y,
    const Normalization& norm);

} // namespace image
} // namespace extension
} // namespace executorch
