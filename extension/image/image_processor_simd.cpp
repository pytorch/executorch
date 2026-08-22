/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/image/image_processor_simd.h>

#include <cstddef>

#include <executorch/runtime/platform/assert.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define ET_IMAGE_USE_NEON 1
#else
#define ET_IMAGE_USE_NEON 0
#endif

namespace executorch {
namespace extension {
namespace image {

using runtime::Error;

namespace {

#if ET_IMAGE_USE_NEON
// Widen 16 uint8 -> 4x float32x4, apply out = in * a + b (single-rounding FMA),
// and store the 16 resulting floats.
__attribute__((always_inline)) inline void
widen_fma_store(uint8x16_t ch, float* dst, float32x4_t a, float32x4_t b) {
  uint16x8_t lo = vmovl_u8(vget_low_u8(ch));
  uint16x8_t hi = vmovl_u8(vget_high_u8(ch));
  vst1q_f32(
      dst + 0, vfmaq_f32(b, vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))), a));
  vst1q_f32(
      dst + 4, vfmaq_f32(b, vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))), a));
  vst1q_f32(
      dst + 8, vfmaq_f32(b, vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))), a));
  vst1q_f32(
      dst + 12, vfmaq_f32(b, vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))), a));
}
#endif // ET_IMAGE_USE_NEON

// Deinterleave + normalize one contiguous run of `n` pixels (stride
// in_channels bytes/pixel) into the r/g/b float planes. NEON when available,
// scalar otherwise; the scalar tail also finishes the final (<16) pixels.
void deinterleave_run(
    const uint8_t* __restrict src,
    size_t n,
    int32_t in_channels,
    int32_t r_off,
    int32_t g_off,
    int32_t b_off,
    float* __restrict r_out,
    float* __restrict g_out,
    float* __restrict b_out,
    float a_r,
    float b_r,
    float a_g,
    float b_g,
    float a_b,
    float b_b) {
  size_t i = 0;
#if ET_IMAGE_USE_NEON
  const float32x4_t va_r = vdupq_n_f32(a_r);
  const float32x4_t vb_r = vdupq_n_f32(b_r);
  const float32x4_t va_g = vdupq_n_f32(a_g);
  const float32x4_t vb_g = vdupq_n_f32(b_g);
  const float32x4_t va_b = vdupq_n_f32(a_b);
  const float32x4_t vb_b = vdupq_n_f32(b_b);
  if (in_channels == 4) {
    for (; i + 16 <= n; i += 16) {
      uint8x16x4_t px = vld4q_u8(src + i * 4);
      widen_fma_store(px.val[r_off], r_out + i, va_r, vb_r);
      widen_fma_store(px.val[g_off], g_out + i, va_g, vb_g);
      widen_fma_store(px.val[b_off], b_out + i, va_b, vb_b);
    }
  } else { // in_channels == 3
    for (; i + 16 <= n; i += 16) {
      uint8x16x3_t px = vld3q_u8(src + i * 3);
      widen_fma_store(px.val[r_off], r_out + i, va_r, vb_r);
      widen_fma_store(px.val[g_off], g_out + i, va_g, vb_g);
      widen_fma_store(px.val[b_off], b_out + i, va_b, vb_b);
    }
  }
#endif // ET_IMAGE_USE_NEON
  for (; i < n; ++i) {
    const uint8_t* p = src + i * in_channels;
    r_out[i] = static_cast<float>(p[r_off]) * a_r + b_r;
    g_out[i] = static_cast<float>(p[g_off]) * a_g + b_g;
    b_out[i] = static_cast<float>(p[b_off]) * a_b + b_b;
  }
}

} // namespace

Error deinterleave_to_chw(
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
    const Normalization& norm) {
  ET_DCHECK_MSG(
      in_channels == 3 || in_channels == 4, "in_channels must be 3 or 4");
  ET_DCHECK_MSG(
      r_off < in_channels && g_off < in_channels && b_off < in_channels,
      "channel offsets must be < in_channels");
  const size_t spatial = static_cast<size_t>(final_w) * final_h;

  // Per-channel affine coefficients for `out = in * a + b`, in RGB order.
  const float a_r = norm.scale_factor / norm.std_dev[0];
  const float a_g = norm.scale_factor / norm.std_dev[1];
  const float a_b = norm.scale_factor / norm.std_dev[2];
  const float b_r = -norm.mean[0] / norm.std_dev[0];
  const float b_g = -norm.mean[1] / norm.std_dev[1];
  const float b_b = -norm.mean[2] / norm.std_dev[2];

  // Output planes in CHW order: R, G, B.
  float* r_plane = output + 0 * spatial;
  float* g_plane = output + 1 * spatial;
  float* b_plane = output + 2 * spatial;

  // Fast path: contiguous source covering the entire plane (no stride padding,
  // no letterbox offset, src dims == final dims) -> one run over all pixels.
  if (src_stride == src_w * in_channels && offset_x == 0 && offset_y == 0 &&
      src_w == final_w && src_h == final_h) {
    deinterleave_run(
        src,
        static_cast<size_t>(src_w) * src_h,
        in_channels,
        r_off,
        g_off,
        b_off,
        r_plane,
        g_plane,
        b_plane,
        a_r,
        b_r,
        a_g,
        b_g,
        a_b,
        b_b);
    return Error::Ok;
  }

  // Slow path: row by row to honor stride padding and/or a letterbox offset.
  for (int32_t y = 0; y < src_h; ++y) {
    const uint8_t* src_row = src + static_cast<size_t>(y) * src_stride;
    const size_t dst_off =
        static_cast<size_t>(y + offset_y) * final_w + offset_x;
    deinterleave_run(
        src_row,
        src_w,
        in_channels,
        r_off,
        g_off,
        b_off,
        r_plane + dst_off,
        g_plane + dst_off,
        b_plane + dst_off,
        a_r,
        b_r,
        a_g,
        b_g,
        a_b,
        b_b);
  }
  return Error::Ok;
}

} // namespace image
} // namespace extension
} // namespace executorch
