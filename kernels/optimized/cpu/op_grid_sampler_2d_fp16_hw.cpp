/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Hardware-fp16 variant of the NEON grid_sampler_2d.out bilinear + zeros-
// padding fast path. This translation unit is compiled with
// `-march=armv8.2-a+fp16`, which lets the compiler emit hardware fp16
// load/store/convert intrinsics (vld1_f16 / vcvt_f32_f16 / vst1_f16 /
// vcvt_f16_f32). Those instructions are undefined on ARMv8.0 and ARMv8.1
// chips without the fp16 extension, so this entry point must only be
// invoked after a runtime CPU-feature check — see the dispatcher in
// op_grid_sampler_2d.cpp (cpuinfo_has_arm_neon_fp16).
//
// Math happens in fp32 regardless: we load fp16 from memory, convert to
// fp32 via the hardware instruction, do the weighted-sum FMA chain in
// fp32, convert back to fp16 on store. This matches the precision of
// the portable kernel once #19117 lands.

#ifdef __aarch64__

#include <executorch/kernels/optimized/cpu/op_grid_sampler_2d_fp16_hw.h>

#include <arm_neon.h>
#include <cmath>

namespace torch {
namespace executor {
namespace native {
namespace opt_grid_sampler_2d_internal {

namespace {

// One output spatial location, all channels.
inline void bilinear_all_channels_fp16_hw_sample(
    const __fp16* input_n,
    __fp16* output_n,
    int C,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    int h_out,
    int w_out,
    float gx,
    float gy) {
  const int x0 = static_cast<int>(std::floor(gx));
  const int y0 = static_cast<int>(std::floor(gy));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const float fx = gx - static_cast<float>(x0);
  const float fy = gy - static_cast<float>(y0);

  const bool tl_v = static_cast<unsigned>(x0) < static_cast<unsigned>(W_in) &&
      static_cast<unsigned>(y0) < static_cast<unsigned>(H_in);
  const bool tr_v = static_cast<unsigned>(x1) < static_cast<unsigned>(W_in) &&
      static_cast<unsigned>(y0) < static_cast<unsigned>(H_in);
  const bool bl_v = static_cast<unsigned>(x0) < static_cast<unsigned>(W_in) &&
      static_cast<unsigned>(y1) < static_cast<unsigned>(H_in);
  const bool br_v = static_cast<unsigned>(x1) < static_cast<unsigned>(W_in) &&
      static_cast<unsigned>(y1) < static_cast<unsigned>(H_in);

  const int off_tl = y0 * W_in + x0;
  const int off_tr = y0 * W_in + x1;
  const int off_bl = y1 * W_in + x0;
  const int off_br = y1 * W_in + x1;
  const int spatial_in = H_in * W_in;
  const int spatial_out = H_out * W_out;
  const int out_off = h_out * W_out + w_out;

  const float32x4_t vw_tl = vdupq_n_f32((1.0f - fx) * (1.0f - fy));
  const float32x4_t vw_tr = vdupq_n_f32(fx * (1.0f - fy));
  const float32x4_t vw_bl = vdupq_n_f32((1.0f - fx) * fy);
  const float32x4_t vw_br = vdupq_n_f32(fx * fy);

  int c = 0;
  for (; c + 3 < C; c += 4) {
    const __fp16* p0 = input_n + (c + 0) * spatial_in;
    const __fp16* p1 = input_n + (c + 1) * spatial_in;
    const __fp16* p2 = input_n + (c + 2) * spatial_in;
    const __fp16* p3 = input_n + (c + 3) * spatial_in;

    __fp16 tl[4] = {0}, tr[4] = {0}, bl[4] = {0}, br[4] = {0};
    if (tl_v) {
      tl[0] = p0[off_tl];
      tl[1] = p1[off_tl];
      tl[2] = p2[off_tl];
      tl[3] = p3[off_tl];
    }
    if (tr_v) {
      tr[0] = p0[off_tr];
      tr[1] = p1[off_tr];
      tr[2] = p2[off_tr];
      tr[3] = p3[off_tr];
    }
    if (bl_v) {
      bl[0] = p0[off_bl];
      bl[1] = p1[off_bl];
      bl[2] = p2[off_bl];
      bl[3] = p3[off_bl];
    }
    if (br_v) {
      br[0] = p0[off_br];
      br[1] = p1[off_br];
      br[2] = p2[off_br];
      br[3] = p3[off_br];
    }

    // Hardware fp16 -> fp32 conversion (requires +fp16 extension).
    const float32x4_t v_tl = vcvt_f32_f16(vld1_f16(tl));
    const float32x4_t v_tr = vcvt_f32_f16(vld1_f16(tr));
    const float32x4_t v_bl = vcvt_f32_f16(vld1_f16(bl));
    const float32x4_t v_br = vcvt_f32_f16(vld1_f16(br));

    float32x4_t result = vmulq_f32(vw_tl, v_tl);
    result = vfmaq_f32(result, vw_tr, v_tr);
    result = vfmaq_f32(result, vw_bl, v_bl);
    result = vfmaq_f32(result, vw_br, v_br);

    __fp16 res[4];
    vst1_f16(res, vcvt_f16_f32(result));
    output_n[(c + 0) * spatial_out + out_off] = res[0];
    output_n[(c + 1) * spatial_out + out_off] = res[1];
    output_n[(c + 2) * spatial_out + out_off] = res[2];
    output_n[(c + 3) * spatial_out + out_off] = res[3];
  }

  // Scalar tail.
  const float w_tl = (1.0f - fx) * (1.0f - fy);
  const float w_tr = fx * (1.0f - fy);
  const float w_bl = (1.0f - fx) * fy;
  const float w_br = fx * fy;
  for (; c < C; ++c) {
    const __fp16* p = input_n + c * spatial_in;
    float v = 0.0f;
    if (tl_v)
      v += w_tl * static_cast<float>(p[off_tl]);
    if (tr_v)
      v += w_tr * static_cast<float>(p[off_tr]);
    if (bl_v)
      v += w_bl * static_cast<float>(p[off_bl]);
    if (br_v)
      v += w_br * static_cast<float>(p[off_br]);
    output_n[c * spatial_out + out_off] = static_cast<__fp16>(v);
  }
}

} // namespace

// Exposed entry point. Called by op_grid_sampler_2d.cpp's dispatcher only
// when cpuinfo_has_arm_neon_fp16() reports true. Input/output data are
// raw uint16_t buffers interpreted as __fp16; N/C/H/W/grid come pre-
// computed from the dispatcher.
void grid_sampler_2d_bilinear_fp16_hw(
    const void* input,
    const void* grid,
    void* output,
    int N,
    int C,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    bool align_corners) {
  const __fp16* in = reinterpret_cast<const __fp16*>(input);
  const __fp16* gd = reinterpret_cast<const __fp16*>(grid);
  __fp16* out = reinterpret_cast<__fp16*>(output);

  const int spatial_in = H_in * W_in;
  const int spatial_out = H_out * W_out;

  for (int n = 0; n < N; ++n) {
    const __fp16* input_n = in + n * C * spatial_in;
    __fp16* output_n = out + n * C * spatial_out;
    const __fp16* grid_n = gd + n * H_out * W_out * 2;

    for (int h = 0; h < H_out; ++h) {
      if (h + 1 < H_out) {
        __builtin_prefetch(grid_n + (h + 1) * W_out * 2, 0, 1);
      }
      for (int w = 0; w < W_out; ++w) {
        const int grid_off = (h * W_out + w) * 2;
        float gx = static_cast<float>(grid_n[grid_off]);
        float gy = static_cast<float>(grid_n[grid_off + 1]);
        if (align_corners) {
          gx = (gx + 1.0f) * (W_in - 1) * 0.5f;
          gy = (gy + 1.0f) * (H_in - 1) * 0.5f;
        } else {
          gx = (gx + 1.0f) * W_in * 0.5f - 0.5f;
          gy = (gy + 1.0f) * H_in * 0.5f - 0.5f;
        }
        bilinear_all_channels_fp16_hw_sample(
            input_n, output_n, C, H_in, W_in, H_out, W_out, h, w, gx, gy);
      }
    }
  }
}

} // namespace opt_grid_sampler_2d_internal
} // namespace native
} // namespace executor
} // namespace torch

#endif // __aarch64__
