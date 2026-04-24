/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Optimized grid_sampler_2d.out for CPU. On aarch64 this is a NEON-vectorized
// implementation for the common (bilinear + zeros padding) case, processing
// 4 channels at a time. Other modes — and non-aarch64 targets — fall through
// to the portable kernel.
//
// fp16 inputs: all interior math (interpolation weights and corner
// accumulation) is done in fp32. Loads/stores stay in the tensor's dtype.
// Avoids catastrophic cancellation on `ix_se - ix`-style subtractions that
// would otherwise make fp16 weights meaningless.

#include <executorch/runtime/kernel/kernel_includes.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include <cmath>

namespace torch {
namespace executor {
namespace native {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;

// Portable kernel (same-op fallback). Both libs link into the same binary.
Tensor& grid_sampler_2d_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    Tensor& out);

#ifdef __aarch64__
namespace {

// One output spatial location, all channels. fp32 path.
inline void bilinear_all_channels_f32(
    const float* input_n,
    float* output_n,
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
    const float* p0 = input_n + (c + 0) * spatial_in;
    const float* p1 = input_n + (c + 1) * spatial_in;
    const float* p2 = input_n + (c + 2) * spatial_in;
    const float* p3 = input_n + (c + 3) * spatial_in;

    float tl[4] = {0}, tr[4] = {0}, bl[4] = {0}, br[4] = {0};
    if (tl_v) {
      tl[0] = p0[off_tl]; tl[1] = p1[off_tl];
      tl[2] = p2[off_tl]; tl[3] = p3[off_tl];
    }
    if (tr_v) {
      tr[0] = p0[off_tr]; tr[1] = p1[off_tr];
      tr[2] = p2[off_tr]; tr[3] = p3[off_tr];
    }
    if (bl_v) {
      bl[0] = p0[off_bl]; bl[1] = p1[off_bl];
      bl[2] = p2[off_bl]; bl[3] = p3[off_bl];
    }
    if (br_v) {
      br[0] = p0[off_br]; br[1] = p1[off_br];
      br[2] = p2[off_br]; br[3] = p3[off_br];
    }

    float32x4_t result = vmulq_f32(vw_tl, vld1q_f32(tl));
    result = vfmaq_f32(result, vw_tr, vld1q_f32(tr));
    result = vfmaq_f32(result, vw_bl, vld1q_f32(bl));
    result = vfmaq_f32(result, vw_br, vld1q_f32(br));

    float res[4];
    vst1q_f32(res, result);
    output_n[(c + 0) * spatial_out + out_off] = res[0];
    output_n[(c + 1) * spatial_out + out_off] = res[1];
    output_n[(c + 2) * spatial_out + out_off] = res[2];
    output_n[(c + 3) * spatial_out + out_off] = res[3];
  }

  // Scalar tail
  const float w_tl = (1.0f - fx) * (1.0f - fy);
  const float w_tr = fx * (1.0f - fy);
  const float w_bl = (1.0f - fx) * fy;
  const float w_br = fx * fy;
  for (; c < C; ++c) {
    const float* p = input_n + c * spatial_in;
    float v = 0.0f;
    if (tl_v) v += w_tl * p[off_tl];
    if (tr_v) v += w_tr * p[off_tr];
    if (bl_v) v += w_bl * p[off_bl];
    if (br_v) v += w_br * p[off_br];
    output_n[c * spatial_out + out_off] = v;
  }
}

// fp16 path: loads/stores fp16, math in fp32.
inline void bilinear_all_channels_f16(
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
      tl[0] = p0[off_tl]; tl[1] = p1[off_tl];
      tl[2] = p2[off_tl]; tl[3] = p3[off_tl];
    }
    if (tr_v) {
      tr[0] = p0[off_tr]; tr[1] = p1[off_tr];
      tr[2] = p2[off_tr]; tr[3] = p3[off_tr];
    }
    if (bl_v) {
      bl[0] = p0[off_bl]; bl[1] = p1[off_bl];
      bl[2] = p2[off_bl]; bl[3] = p3[off_bl];
    }
    if (br_v) {
      br[0] = p0[off_br]; br[1] = p1[off_br];
      br[2] = p2[off_br]; br[3] = p3[off_br];
    }

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

  const float w_tl = (1.0f - fx) * (1.0f - fy);
  const float w_tr = fx * (1.0f - fy);
  const float w_bl = (1.0f - fx) * fy;
  const float w_br = fx * fy;
  for (; c < C; ++c) {
    const __fp16* p = input_n + c * spatial_in;
    float v = 0.0f;
    if (tl_v) v += w_tl * static_cast<float>(p[off_tl]);
    if (tr_v) v += w_tr * static_cast<float>(p[off_tr]);
    if (bl_v) v += w_bl * static_cast<float>(p[off_bl]);
    if (br_v) v += w_br * static_cast<float>(p[off_br]);
    output_n[c * spatial_out + out_off] = static_cast<__fp16>(v);
  }
}

template <typename SCALAR, typename SampleFn>
void grid_sampler_2d_neon(
    const SCALAR* input,
    const SCALAR* grid,
    SCALAR* output,
    int N,
    int C,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    bool align_corners,
    SampleFn sample_fn) {
  const int spatial_in = H_in * W_in;
  const int spatial_out = H_out * W_out;

  for (int n = 0; n < N; ++n) {
    const SCALAR* input_n = input + n * C * spatial_in;
    SCALAR* output_n = output + n * C * spatial_out;
    const SCALAR* grid_n = grid + n * H_out * W_out * 2;

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
        sample_fn(
            input_n, output_n, C, H_in, W_in, H_out, W_out, h, w, gx, gy);
      }
    }
  }
}

} // namespace
#endif // __aarch64__

Tensor& opt_grid_sampler_2d_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    Tensor& out) {
  // The NEON path indexes input/grid/out directly assuming a contiguous NCHW
  // default-dim-order layout — no use of .strides() or .dim_order(). If the
  // caller passes anything else, fall back to portable (which does handle
  // arbitrary strides and dim orders correctly). These are cheap checks.
  const bool fast_eligible = tensor_is_default_dim_order(input) &&
      tensor_is_default_dim_order(grid) &&
      tensor_is_default_dim_order(out) &&
      tensor_is_contiguous(input) &&
      tensor_is_contiguous(grid) &&
      tensor_is_contiguous(out);

  // Only the bilinear + zeros-padding combination is accelerated. Everything
  // else — non-default layout, any non-aarch64 target — delegates to portable.
  if (interpolation_mode != 0 || padding_mode != 0 || !fast_eligible) {
    return grid_sampler_2d_out(
        ctx, input, grid, interpolation_mode, padding_mode, align_corners, out);
  }
#ifndef __aarch64__
  return grid_sampler_2d_out(
      ctx, input, grid, interpolation_mode, padding_mode, align_corners, out);
#else
  const int N = static_cast<int>(input.size(0));
  const int C = static_cast<int>(input.size(1));
  const int H_in = static_cast<int>(input.size(2));
  const int W_in = static_cast<int>(input.size(3));
  const int H_out = static_cast<int>(grid.size(1));
  const int W_out = static_cast<int>(grid.size(2));

  if (input.scalar_type() == ScalarType::Float) {
    grid_sampler_2d_neon<float>(
        input.const_data_ptr<float>(),
        grid.const_data_ptr<float>(),
        out.mutable_data_ptr<float>(),
        N, C, H_in, W_in, H_out, W_out,
        align_corners,
        bilinear_all_channels_f32);
    return out;
  }
  if (input.scalar_type() == ScalarType::Half) {
    static_assert(sizeof(__fp16) == 2, "expected __fp16 == 2 bytes");
    grid_sampler_2d_neon<__fp16>(
        reinterpret_cast<const __fp16*>(input.const_data_ptr<uint16_t>()),
        reinterpret_cast<const __fp16*>(grid.const_data_ptr<uint16_t>()),
        reinterpret_cast<__fp16*>(out.mutable_data_ptr<uint16_t>()),
        N, C, H_in, W_in, H_out, W_out,
        align_corners,
        bilinear_all_channels_f16);
    return out;
  }
  // Any other dtype (e.g. Double, BFloat16): let portable handle it.
  return grid_sampler_2d_out(
      ctx, input, grid, interpolation_mode, padding_mode, align_corners, out);
#endif
}

} // namespace native
} // namespace executor
} // namespace torch
