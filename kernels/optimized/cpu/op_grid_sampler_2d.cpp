/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Optimized grid_sampler_2d.out for CPU. On aarch64 this is a NEON-vectorized
// implementation for the common (bilinear + zeros padding) case. fp16 inputs
// are promoted to fp32 for weight computation and accumulation and cast back
// on store — this avoids fp16 catastrophic cancellation on `ix_se - ix`-style
// weight subtractions in the portable kernel.
//
// fp16 comes in two flavors to avoid SIGILL on ARMv8 chips without the
// +fp16 extension:
//
//   * Hardware path (op_grid_sampler_2d_fp16_hw.cpp) — compiled with
//     `-march=armv8.2-a+fp16`. Uses hardware fp16 NEON instructions
//     (vld1_f16 / vcvt_f32_f16 / ...). Fast on capable chips; illegal
//     instructions on older ones.
//
//   * Software path (below) — plain ARMv8 NEON. Converts fp16<->fp32 in
//     software via `c10::Half`'s portable conversion. Slower per
//     conversion but safe on any ARMv8 CPU.
//
// A runtime cpuinfo_has_arm_neon_fp16() check picks the right one. Non-aarch64
// targets, and any unsupported interpolation/padding/layout combination,
// delegate to the portable kernel.

#include <executorch/runtime/kernel/kernel_includes.h>

#ifdef __aarch64__
#include <arm_neon.h>
#include <cpuinfo.h>
#include <executorch/kernels/optimized/cpu/op_grid_sampler_2d_fp16_hw.h>
#endif

#include <c10/util/Half.h>

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

// -------------------- fp32 (plain ARMv8 NEON) --------------------

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

  const float w_tl = (1.0f - fx) * (1.0f - fy);
  const float w_tr = fx * (1.0f - fy);
  const float w_bl = (1.0f - fx) * fy;
  const float w_br = fx * fy;
  for (; c < C; ++c) {
    const float* p = input_n + c * spatial_in;
    float v = 0.0f;
    if (tl_v)
      v += w_tl * p[off_tl];
    if (tr_v)
      v += w_tr * p[off_tr];
    if (bl_v)
      v += w_bl * p[off_bl];
    if (br_v)
      v += w_br * p[off_br];
    output_n[c * spatial_out + out_off] = v;
  }
}

// -------------------- fp16 software-convert path --------------------
//
// Uses only plain ARMv8 NEON. fp16 <-> fp32 conversion goes through
// c10::Half's portable `operator float()` / constructor, which is a
// software conversion on chips that lack the +fp16 extension.

inline void bilinear_all_channels_f16_sw(
    const c10::Half* input_n,
    c10::Half* output_n,
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
    const c10::Half* p0 = input_n + (c + 0) * spatial_in;
    const c10::Half* p1 = input_n + (c + 1) * spatial_in;
    const c10::Half* p2 = input_n + (c + 2) * spatial_in;
    const c10::Half* p3 = input_n + (c + 3) * spatial_in;

    // SW fp16 -> fp32: use c10::Half's portable conversion on each lane.
    float tl[4] = {0}, tr[4] = {0}, bl[4] = {0}, br[4] = {0};
    if (tl_v) {
      tl[0] = static_cast<float>(p0[off_tl]);
      tl[1] = static_cast<float>(p1[off_tl]);
      tl[2] = static_cast<float>(p2[off_tl]);
      tl[3] = static_cast<float>(p3[off_tl]);
    }
    if (tr_v) {
      tr[0] = static_cast<float>(p0[off_tr]);
      tr[1] = static_cast<float>(p1[off_tr]);
      tr[2] = static_cast<float>(p2[off_tr]);
      tr[3] = static_cast<float>(p3[off_tr]);
    }
    if (bl_v) {
      bl[0] = static_cast<float>(p0[off_bl]);
      bl[1] = static_cast<float>(p1[off_bl]);
      bl[2] = static_cast<float>(p2[off_bl]);
      bl[3] = static_cast<float>(p3[off_bl]);
    }
    if (br_v) {
      br[0] = static_cast<float>(p0[off_br]);
      br[1] = static_cast<float>(p1[off_br]);
      br[2] = static_cast<float>(p2[off_br]);
      br[3] = static_cast<float>(p3[off_br]);
    }

    float32x4_t result = vmulq_f32(vw_tl, vld1q_f32(tl));
    result = vfmaq_f32(result, vw_tr, vld1q_f32(tr));
    result = vfmaq_f32(result, vw_bl, vld1q_f32(bl));
    result = vfmaq_f32(result, vw_br, vld1q_f32(br));

    float res[4];
    vst1q_f32(res, result);
    // SW fp32 -> fp16 on store.
    output_n[(c + 0) * spatial_out + out_off] = c10::Half(res[0]);
    output_n[(c + 1) * spatial_out + out_off] = c10::Half(res[1]);
    output_n[(c + 2) * spatial_out + out_off] = c10::Half(res[2]);
    output_n[(c + 3) * spatial_out + out_off] = c10::Half(res[3]);
  }

  const float w_tl = (1.0f - fx) * (1.0f - fy);
  const float w_tr = fx * (1.0f - fy);
  const float w_bl = (1.0f - fx) * fy;
  const float w_br = fx * fy;
  for (; c < C; ++c) {
    const c10::Half* p = input_n + c * spatial_in;
    float v = 0.0f;
    if (tl_v)
      v += w_tl * static_cast<float>(p[off_tl]);
    if (tr_v)
      v += w_tr * static_cast<float>(p[off_tr]);
    if (bl_v)
      v += w_bl * static_cast<float>(p[off_bl]);
    if (br_v)
      v += w_br * static_cast<float>(p[off_br]);
    output_n[c * spatial_out + out_off] = c10::Half(v);
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
        sample_fn(input_n, output_n, C, H_in, W_in, H_out, W_out, h, w, gx, gy);
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
  // The NEON paths index input/grid/out directly assuming a contiguous NCHW
  // default-dim-order layout — no use of .strides() or .dim_order(). Fall
  // back to portable for anything else.
  const bool fast_eligible = tensor_is_default_dim_order(input) &&
      tensor_is_default_dim_order(grid) && tensor_is_default_dim_order(out) &&
      tensor_is_contiguous(input) && tensor_is_contiguous(grid) &&
      tensor_is_contiguous(out);

  // The fast paths read input/grid and write out as a single dtype: float for
  // the fp32 NEON path, fp16 for both the fp16 HW path (which raw-casts the
  // void* pointers to __fp16*) and the SW fp16 NEON path (which uses
  // data_ptr<c10::Half>(), whose runtime dtype check is not guaranteed in
  // release builds). Reject any mixed-dtype call up front so none of those
  // unchecked casts can be reached with a mismatched buffer.
  const bool dtypes_match = input.scalar_type() == grid.scalar_type() &&
      input.scalar_type() == out.scalar_type();

  if (interpolation_mode != 0 || padding_mode != 0 || !fast_eligible ||
      !dtypes_match) {
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
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        align_corners,
        bilinear_all_channels_f32);
    return out;
  }
  if (input.scalar_type() == ScalarType::Half) {
    if (cpuinfo_initialize() && cpuinfo_has_arm_neon_fp16()) {
      // Hardware fp16 path — safe because the CPU supports the +fp16
      // extension. Declared in op_grid_sampler_2d_fp16_hw.cpp.
      opt_grid_sampler_2d_internal::grid_sampler_2d_bilinear_fp16_hw(
          input.const_data_ptr(),
          grid.const_data_ptr(),
          out.mutable_data_ptr(),
          N,
          C,
          H_in,
          W_in,
          H_out,
          W_out,
          align_corners);
      return out;
    }
    // Software fp16<->fp32 conversion path. Works on any ARMv8.
    grid_sampler_2d_neon<c10::Half>(
        input.const_data_ptr<c10::Half>(),
        grid.const_data_ptr<c10::Half>(),
        out.mutable_data_ptr<c10::Half>(),
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        align_corners,
        bilinear_all_channels_f16_sw);
    return out;
  }
  // Any other dtype: let portable handle it.
  return grid_sampler_2d_out(
      ctx, input, grid, interpolation_mode, padding_mode, align_corners, out);
#endif
}

} // namespace native
} // namespace executor
} // namespace torch
