/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * conv2d_gemm: GEMM step of im2col-backed conv2d.
 *
 * Reads one tile of the im2col'd input produced by conv2d_im2col.glsl — a 2D
 * matrix of shape [M_TILE, K_total] holding OH_TILE output-height rows
 * (M_TILE = OH_TILE * W_out, K_total = Kh*Kw*Cin_padded) starting at output
 * row OH_OFFSET — and writes the corresponding output rows as texture3D
 * channels-packed, logical shape [1, C_out, H_out, W_out]. The full im2col
 * matrix is processed OH_TILE rows per dispatch so the scratch tensor is bounded
 * to a fixed byte budget regardless of resolution; with tiling disabled the
 * caller passes OH_OFFSET = 0 and OH_TILE = H_out (one dispatch covers all M).
 *
 * The im2col input tile can be any of:
 *   - texture2d, width-packed: texel at (k4, r) holds 4 K values for tile-local
 *     row r.  IN_STORAGE=texture2d codegen.
 *   - texture3d, channels-packed: texel at (ow, oh_local, k4) holds 4 K values
 *     for tile-local row r = oh_local * W_out + ow.  Used when the per-tile 2D
 *     extent (M_TILE = OH_TILE * W_out) would exceed max_texture2d_dim — rare,
 *     since OH_TILE is capped by the scratch byte budget.  IN_STORAGE=texture3d
 *     codegen.
 *   - buffer: vec4 at offset r*K4 + k4, same K packing.
 *     IN_STORAGE=buffer codegen.
 *
 * The matmul interpretation (over this tile's rows) is:
 *   out[r, n] = sum_k im2col[r, k] * weight[n, k] + bias[n]
 * with K = K_total, N = C_out, and r the tile-local row whose global output
 * spatial position is (OH_OFFSET + r / W_out, r % W_out).
 */

#version 450 core

#define PRECISION ${PRECISION}

$if IN_STORAGE == "buffer" and DTYPE == "half":
  ${define_explicit_type_extensions(DTYPE)}

// VEC4_T is the input storage's natural texel type, which is also the tile type
// (the linear_fp_*_tile headers default the tile vec4 type to VEC4_T). For the
// buffer/half path this resolves to f16vec4, so the GEMM inner loop accumulates
// in true FP16 — the fma emits mad.f16 and the accumulators live in half-width
// registers. Texture-sampled half always returns vec4, so FP16 accumulation is
// naturally confined to the buffer (Mali) path; the texture variants (Adreno),
// where FP16 accumulation regresses, stay vec4 / FP32 with no extra gating.
#define VEC4_T ${texel_load_type(DTYPE, IN_STORAGE)}

// OUT_VEC4_T is the output surface type. t_out is always texture3d, whose
// imageStore ABI takes vec4 (fp32) regardless of DTYPE, so the accumulator tile
// is cast from VEC4_T to OUT_VEC4_T at store time.
#define OUT_VEC4_T ${texel_load_type(DTYPE, "texture3d")}

#define TILE_M4 ${TILE_M4}
#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M ${TILE_M}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

$if IN_STORAGE == "buffer":
  #define INPUT_BUFFER
$elif IN_STORAGE == "texture3d":
  #define INPUT_TEXTURE3D

${define_required_extensions("texture3d", DTYPE)}
$if IN_STORAGE == "buffer":
  ${define_required_extensions("buffer", DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
$if IN_STORAGE == "buffer":
  ${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer", is_scalar_array=False)}
$else:
  ${layout_declare_tensor(B, "r", "t_in", DTYPE, IN_STORAGE)}
${layout_declare_tensor(B, "r", "t_weight_packed", DTYPE, "texture2d")}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "texture2d")}

${layout_declare_ubo(B, "ivec4", "out_sizes")}

// Push constants are uploaded in 16-byte chunks (one ivec4 each).
// K4_total is shape-independent (it depends only on C_in and the conv kernel
// dims), so it is safe to bake at build time even under dynamic shapes.
// M = H_out * W_out IS shape-dependent, so it is derived at runtime from the
// refreshed out_sizes UBO in main() rather than read from here.
//
// This dispatch consumes one tile of the im2col matrix: OH_TILE output-height
// rows starting at output-height row OH_OFFSET. The im2col scratch (t_in) holds
// OH_TILE * W_out tile-local rows; the GEMM reads tile-local rows and writes the
// output at the corresponding global spatial position (OH_OFFSET + oh_local,
// ow). OH_OFFSET / OH_TILE are shape-independent (fixed at build time); the
// global W_out / H_out come from the refreshed out_sizes UBO.
layout(push_constant) uniform restrict Block {
  ivec4 gemm_dims;   // (K4_total, OH_OFFSET, OH_TILE, _unused)
  vec4  clamp_vals;  // (out_min, out_max, _unused, _unused)
};

#define K4_TOTAL  gemm_dims.x
#define OH_OFFSET gemm_dims.y
#define OH_TILE   gemm_dims.z
#define OUT_MIN   clamp_vals.x
#define OUT_MAX   clamp_vals.y

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "activation_type", "0")}

#include "linear_fp_input_tile.glslh"
#include "linear_fp_packed_weight_tile_load.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"

/*
 * Load TILE_M rows × TILE_K4 K-tiles of the im2col'd input.
 * The im2col scratch holds M_TILE tile-local rows in a contiguous
 * (M_TILE, K_total/4) matrix of vec4s; row here is the tile-local index, so the
 * load is a plain 2D fetch — no spatial decomposition. (The output store, not
 * this load, maps the tile-local row to its global spatial position.)
 */
void load_input_tile_with_checks(
    out FPInputTile tile,
    const int k4_start,
    const int m_start,
    const int K4,
    const int M_TILE,
    const int W_out) {
  // W_out is only consumed by the texture3d variant below.
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int k4 = 0; k4 < TILE_K4; ++k4) {
      if (k4_start + k4 < K4 && m_start + m < M_TILE) {
        const int row = m_start + m;
        const int col = k4_start + k4;
#if defined(INPUT_BUFFER)
        // Cast SSBO texel into the input tile type (f16vec4 for half, vec4 for
        // float).
        tile.data[m][k4] = LINEAR_FP_INPUT_TILE_VEC4_T(t_in[row * K4 + col]);
#elif defined(INPUT_TEXTURE3D)
        // texture3d scratch [1, K_total, OH_TILE, W_out]: the tile-local row
        // decomposes into (ow, oh_local) and K4 is along the Z axis. texelFetch
        // returns vec4 (fp32); cast to the input tile type.
        tile.data[m][k4] = LINEAR_FP_INPUT_TILE_VEC4_T(
            texelFetch(t_in, ivec3(row % W_out, row / W_out, col), 0));
#else
        tile.data[m][k4] =
            LINEAR_FP_INPUT_TILE_VEC4_T(texelFetch(t_in, ivec2(col, row), 0));
#endif
      } else {
        tile.data[m][k4] = LINEAR_FP_INPUT_TILE_VEC4_T(0.0);
      }
    }
  }
}

// m_start is a tile-local row offset; the scratch read uses it directly, but
// the output store maps it to the GLOBAL spatial position via oh_global =
// OH_OFFSET + (m_local / W_out). Rows whose global oh lands past H_out (the
// partial trailing tile, or a dynamic shape that shrinks H_out) are skipped by
// the `oh < H_out` guard. The companion `m_local < M_TILE` guard enforces the
// tile's UPPER oh bound: since M_TILE = OH_TILE * W_out, m_local < M_TILE means
// oh_local < OH_TILE, i.e. oh < OH_OFFSET + OH_TILE — so no row this dispatch
// writes can leak into a neighboring tile's output-row range.
void store_output_tile_with_checks(
    const FPOutTile out_tile,
    const int n4_start,
    const int m_start,
    const int N4,
    const int M_TILE,
    const int H_out,
    const int W_out) {
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      const int m_local = m_start + m;
      const int ow = m_local % W_out;
      const int oh = OH_OFFSET + m_local / W_out;
      if (m_local < M_TILE && oh < H_out && n4_start + n4 < N4) {
        // Cast the accumulator (f16vec4 for the buffer/half path) to the
        // texture3d output surface type for the activation clamp and store.
        OUT_VEC4_T texel = OUT_VEC4_T(out_tile.data[m][n4]);
        if (activation_type == 1) {
          texel = max(texel, OUT_VEC4_T(0.0));
        } else if (activation_type == 2) {
          texel = clamp(texel, OUT_VEC4_T(OUT_MIN), OUT_VEC4_T(OUT_MAX));
        }
        imageStore(t_out, ivec3(ow, oh, n4_start + n4), texel);
      }
    }
  }
}

void main() {
  const int tile_idx_n = int(gl_GlobalInvocationID.x);
  const int tile_idx_m = int(gl_GlobalInvocationID.y);

  const int n4_start = tile_idx_n * TILE_N4;
  const int m_start = tile_idx_m * TILE_M;

  const int W_out = out_sizes.x;
  const int H_out = out_sizes.y;
  // W_out / H_out are derived from the refreshed out_sizes UBO so they track
  // dynamic output shapes (out_sizes is virtual_resize'd on trigger_resize).
  // M_TILE = OH_TILE * W_out is the tile-local row count materialized in the
  // im2col scratch (t_in); the GEMM reads scratch rows in [0, M_TILE).
  const int M_TILE = OH_TILE * W_out;
  const int K4 = K4_TOTAL;
  const int N = out_sizes.z;
  const int N4 = div_up_4(N);

  if (n4_start >= N4 || m_start >= M_TILE) {
    return;
  }

  FPOutTile out_tile;
  initialize(out_tile);

  FPInputTile in_tile;
  FPWeightTile w_tile;

  for (int k4 = 0; k4 < K4; k4 += TILE_K4) {
    load_input_tile_with_checks(in_tile, k4, m_start, K4, M_TILE, W_out);
    load_packed_weight_tile_with_checks(w_tile, n4_start, k4, 0, N4, K4);
    fp_accumulate_with_fp_weight(out_tile, in_tile, w_tile);
  }

  // Apply bias. The bias texel depends only on n4, so fetch it once per n4 and
  // add it to every m row rather than re-fetching inside the M loop.
  [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
    if (n4_start + n4 < N4) {
      // t_bias is an fp32 texture2d; cast its texel to the accumulator type.
      const LINEAR_FP_OUTPUT_TILE_VEC4_T bias_texel =
          LINEAR_FP_OUTPUT_TILE_VEC4_T(
              texelFetch(t_bias, ivec2(n4_start + n4, 0), 0));
      [[unroll]] for (int m = 0; m < TILE_M; ++m) {
        out_tile.data[m][n4] += bias_texel;
      }
    }
  }

  store_output_tile_with_checks(
      out_tile, n4_start, m_start, N4, M_TILE, H_out, W_out);
}
