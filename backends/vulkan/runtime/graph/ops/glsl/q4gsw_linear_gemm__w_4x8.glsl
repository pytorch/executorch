/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// q4gsw linear GEMM kernel with 4M x 8N per-thread output tiles.
//
// Shader naming convention:
//   q4gsw_linear_gemm__w_4x8_<arrangement>
//   ^^^^^^^^^^^^^^^^^  ^^^^^ ^^^^^^^^^^^^^
//   op base            tile  weight binding form (nc=ivec4 buffer, kc=ivec4 image2D)
//
// The absence of an input layout tag (e.g. `tin`) indicates that the
// activation is consumed directly from the logical [M, K] row-major layout —
// no preprocess-time transpose is required. This path is used for fp32 I/O
// where preserving the contiguous input and using fp32 accumulation yields
// better performance than the pre-transposed path used by the fp16 variant.
//
// Weight block layout (4K x 8N), interleaved (dp4a-style) byte pairs:
//   Each block packs 4 ints. The 4 ints carry byte-pair nibble lanes for two
//   consecutive n4 tiles (n4_a = 2*n8, n4_b = 2*n8+1) at the same k4:
//     int 0 byte b = (N=4*n4_a+0, K=k4*4+b) | (N=4*n4_a+1, K=k4*4+b) << 4
//     int 1 byte b = (N=4*n4_a+2, K=k4*4+b) | (N=4*n4_a+3, K=k4*4+b) << 4
//     int 2 byte b = (N=4*n4_b+0, K=k4*4+b) | (N=4*n4_b+1, K=k4*4+b) << 4
//     int 3 byte b = (N=4*n4_b+2, K=k4*4+b) | (N=4*n4_b+3, K=k4*4+b) << 4
//   The low nibble per byte is the even-N row; the high nibble is the odd-N
//   row. This is the natural memory split for the per-mi FMA chain (no repack)
//   and is shared with the GEMV coop kc shader.
//
// Weight storage variants (selected by WEIGHT_STORAGE):
//   "buffer"    (nc) — ivec4 buffer; one ivec4 per (k4, n8) at flat index
//                      `k4 * (N4_padded / 2) + n8`. Row stride padded to
//                      N4_padded (next-even N4) so the 16B load never
//                      straddles a k4 row even when N % 8 != 0.
//   "texture2d" (kc) — ivec4 image2D; texelFetch at ivec2(k4, n8) returns the
//                      same 4-int payload. K is the texture-fetch contiguous
//                      axis, routing weight reads through the texture cache
//                      (shared with the kc GEMV variant).
//
// Thread mapping:
//   gl_GlobalInvocationID.x -> N tile index (n4 = TILE_N4 tiles wide)
//   gl_GlobalInvocationID.y -> M tile index (4 M rows per tile)
//
// Tile shape: 4M x (4 * TILE_N4)N per thread, accumulated as
// VEC4_T out_tile[TILE_M][TILE_N4]. Scales are loaded once per quantization
// group and reused across K4_per_group inner K steps.
//
// IO_STORAGE applies to both input activation and output; tests always keep
// them matching. Scales and bias are always buffers.

#version 450 core

${define_required_extensions(IO_STORAGE, DTYPE)}
${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, IO_STORAGE)}
#define T ${texel_load_component_type(DTYPE, IO_STORAGE)}

$if IO_STORAGE == "buffer":
  #define OUTPUT_BUFFER
  #define INPUT_BUFFER

$if WEIGHT_STORAGE == "texture2d":
  #define WEIGHT_TEX2D

$if WEIGHT_KC == 1:
  #define WEIGHT_KC

#define TILE_M4 ${TILE_M4}
#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M (TILE_M4 * 4)
#define TILE_K (TILE_K4 * 4)
#define TILE_N (TILE_N4 * 4)

#extension GL_EXT_control_flow_attributes : require

#define div_up_4(x) (((x) + 3) >> 2)

layout(std430) buffer;

// Unified 6-binding layout shared across q4gsw_linear shaders so a single
// DynamicDispatchNode with pick_shader_fn can switch between GEMM and GEMV
// kernels. This shader reads t_fp_input (the raw activation). The
// t_transposed_input binding is declared to preserve slot order but is never
// referenced here — the driver compiles it out to zero runtime cost; only
// the descriptor slot is allocated.
${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_fp_input", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_transposed_input", DTYPE, "buffer", is_scalar_array=False)}
// W_4X8 block-packed weight binding. Two variants share a 4K x 8N block payload:
//
//   WEIGHT_STORAGE == "buffer" (nc): ivec4 buffer view of the uint stream
//     produced by pack_q4_linear_weight__w_4x8_nc. Two consecutive 4Kx4N
//     ivec2 tiles along N are packed into a single ivec4 to issue one 16B
//     LSU transaction instead of two 8B ones — measurably cheaper on Adreno.
//     The ivec4 at index `k4 * (N4_padded / 2) + (n4 / 2)` covers both ivec2
//     blocks at (k4, n4) and (k4, n4 + 1). w_block.xy = packed_weight[0];
//     w_block.zw = packed_weight[1]. The prepack pads the buffer's row stride to
//     N4_padded (next-even N4), so this load never straddles k4 rows even
//     for N % 8 != 0 inputs — the OOB tile is populated with the bias-zero
//     nibble pattern (0x88888888u) by the prepack shader's (n < N) branch.
//
//   WEIGHT_STORAGE == "texture2d" (kc): ivec4 image2D produced by
//     pack_q4_linear_weight__w_4x8_kc_texture2d. texelFetch at ivec2(k4, n8)
//     returns the same 4-int payload covering 4K x 8N. Routing weight reads
//     through the texture cache (shared with the kc coop GEMV) recovers
//     measurable perf on Adreno when the GEMV is also dispatched against the
//     same prepack output. K is the inner-contiguous axis.
${layout_declare_tensor(B, "r", "t_q4_weights", "int", WEIGHT_STORAGE, is_scalar_array=False, vec_size=4)}
${layout_declare_tensor(B, "r", "t_scales", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer")}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "0")}
${layout_declare_spec_const(C, "int", "K", "1024")}
${layout_declare_spec_const(C, "int", "group_size", "32")}

void main() {
  const int out_tile_x = int(gl_GlobalInvocationID.x);
  const int out_tile_y = int(gl_GlobalInvocationID.y);

  const int n = out_tile_x * TILE_N;
  const int m = out_tile_y * TILE_M;

  const int n4 = n / 4;

  if (n >= output_sizes.x || m >= output_sizes.y) {
    return;
  }

  const int M = input_sizes.y;
  const int K4 = div_up_4(input_sizes.x);
  const int N4 = div_up_4(output_sizes.x);
  // Padded N4 row stride for the W_4X8 block-packed weight buffer (next-even N4). The
  // prepack pads the buffer to this stride so the ivec4 weight load never
  // straddles a k4 row. For N % 8 == 0 this is identical to N4.
  const int N4_padded = (N4 + 1) & ~1;
  const int K4_per_group = group_size / 4;

  // Output accumulator tile: [TILE_M][TILE_N4] VEC4_T
  VEC4_T out_tile[TILE_M][TILE_N4];
  [[unroll]] for (int i = 0; i < TILE_M; ++i) {
    [[unroll]] for (int j = 0; j < TILE_N4; ++j) {
      out_tile[i][j] = VEC4_T(0);
    }
  }

  // Input tile: [TILE_M][TILE_K4] VEC4_T
  VEC4_T in_tile[TILE_M][TILE_K4];

  // n8 = (n / 8) — pair index used by both the buffer ivec4 stride math and
  // the Tex2D texelFetch coordinate. With TILE_N4 = 2 and TILE_N = 8, each
  // thread covers exactly one n8 worth of N rows.
  const int n8 = n4 >> 1;

  // W_4X8 block-packed weight payload: one ivec4 per (k4, n8) covers TILE_N4=2 N4 tiles
  // (= 8 N-rows) at once. Same int layout for buffer (nc) and texture2d (kc).
  ivec4 w_block;

  // Scales: [TILE_N4] VEC4_T
  VEC4_T scales[TILE_N4];

  const int num_groups = K4 / K4_per_group;

  for (int group_i = 0; group_i < num_groups; ++group_i) {
    // Load scales for this quantization group. The scales buffer holds
    // (K/gs) * N4 vec4 elements (no padding — only the weight buffer is
    // padded to N4_padded). For odd N4 the boundary thread's i=1 read at
    // (n4 + 1 == N4) would read OOB at the very last group; clamp the index
    // to N4 - 1 to keep the read in-bounds. The output store gates n4 + ni
    // < N4, so the (n4 + 1 == N4) accumulation is never persisted — only
    // memory-safety matters here, not correctness of the discarded value.
    [[unroll]] for (int i = 0; i < TILE_N4; ++i) {
      const int n4_clamped = min(n4 + i, N4 - 1);
      scales[i] = VEC4_T(t_scales[group_i * N4 + n4_clamped]);
    }

    for (int k4_inner = 0; k4_inner < K4_per_group; ++k4_inner) {
      const int k4 = group_i * K4_per_group + k4_inner;

      // Load input tile. Tail rows may be read but are discarded by the output
      // store guard below.
      [[unroll]] for (int mi = 0; mi < TILE_M; ++mi) {
#ifdef INPUT_BUFFER
        in_tile[mi][0] = t_fp_input[((m + mi) * K4) + k4];
#else
        in_tile[mi][0] = texelFetch(t_fp_input, ivec3(k4, m + mi, 0), 0);
#endif
      }

      // Load both W_4X8 weight blocks for (k4, n4) and (k4, n4+1) as a single
      // ivec4 covering the 4K x 8N block. Buffer (nc) path: ivec4 view of
      // the uint stream; index `k4 * (N4_padded / 2) + n8` lands on the
      // 2-tile pair. N4_padded is even by construction (prepack rounds up
      // the row stride to the next even value), so the load is well-formed
      // for any N satisfying N % 4 == 0 — the OOB tile when N4 is odd is
      // populated with bias-zero nibbles by the prepack. Texture2d (kc)
      // path: same payload returned by texelFetch at (k4, n8); routes the
      // weight read through the texture cache.
#if defined(WEIGHT_TEX2D) && defined(WEIGHT_KC)
      // kc dense Tex2D form: image position (k4, n8); texelFetch returns the
      // 4K x 8N byte-pair payload routed through the texture cache.
      w_block = texelFetch(t_q4_weights, ivec2(k4, n8), 0);
#elif defined(WEIGHT_TEX2D)
      // nc Tex2D form: image position (n8, k4); same byte-pair payload as the
      // nc-buffer variant but routed through the texture cache. Adjacent
      // texels along x are adjacent n8 (nc-contiguous).
      w_block = texelFetch(t_q4_weights, ivec2(n8, k4), 0);
#elif defined(WEIGHT_KC)
      // kc dense buffer form: SSBO ivec4 indexed at `n8 * K4 + k4`. Same
      // 4K x 8N byte-pair payload as the Tex2D variant; only the cache path
      // changes (SSBO vs texture cache). Stride along k4 is 1 ivec4; stride
      // along n8 is K4 ivec4s.
      w_block = t_q4_weights[n8 * K4 + k4];
#else
      // nc buffer form. Index `k4 * (N4_padded / 2) + n8`. N4_padded is even
      // by construction (prepack rounds up the row stride to next even).
      w_block = t_q4_weights[k4 * (N4_padded >> 1) + n8];
#endif

      // Dequantize and accumulate. Loop nesting: k4i outer, both ni's paired
      // adjacently inside. This pairing lets the Adreno compiler fold the
      // ni=1 FMA chain into multi-shot mads with the ni=0 chain across the 4
      // mi's of TILE_M (measured: (rpt2) on the FMA pass), and coalesces both
      // halves of the dequant register block (drops 5 GPRs vs the ni-outer
      // form, doubles occupancy 37% -> 50% on Adreno 750).
      //
      // weight_texels declared as a 2-element local array (instead of two
      // separate VEC4_T scalars) gives the compiler freedom to allocate both
      // halves in a contiguous register block; the live region of the second
      // half stays adjacent to the first across the FMA sweep.
      VEC4_T weight_texels[2];
      [[unroll]] for (int k4i = 0; k4i < 4; ++k4i) {
        const int shift_lo = 8 * k4i;       // even-N rows (low nibble)
        const int shift_hi = 8 * k4i + 4;   // odd-N rows  (high nibble)

        // Adjacent-pairs layout: w_block.x covers (N0,N1), .y covers (N2,N3),
        // .z covers (N4,N5), .w covers (N6,N7). One VEC4_T output (4 N rows)
        // packs 2 adjacent component pairs with alternating low/high shifts.
        weight_texels[0] = VEC4_T(
            T(int((uint(w_block.x) >> shift_lo) & 0xFu) - 8),
            T(int((uint(w_block.x) >> shift_hi) & 0xFu) - 8),
            T(int((uint(w_block.y) >> shift_lo) & 0xFu) - 8),
            T(int((uint(w_block.y) >> shift_hi) & 0xFu) - 8));
        weight_texels[1] = VEC4_T(
            T(int((uint(w_block.z) >> shift_lo) & 0xFu) - 8),
            T(int((uint(w_block.z) >> shift_hi) & 0xFu) - 8),
            T(int((uint(w_block.w) >> shift_lo) & 0xFu) - 8),
            T(int((uint(w_block.w) >> shift_hi) & 0xFu) - 8));

        // Scale both halves before the FMA chain. fma(w, scale, 0) folds to a
        // mul; the FMA shape matches the SSA produced by helper-driven LEGACY
        // builds and keeps instruction selection identical.
        weight_texels[0] = fma(weight_texels[0], scales[0], VEC4_T(0));
        weight_texels[1] = fma(weight_texels[1], scales[1], VEC4_T(0));

        // FMA both halves into accum, paired per m. The ni=1 FMA right after
        // ni=0 lets the compiler fold the second mad into (rpt2) with the
        // first across the 4 mi's.
        [[unroll]] for (int mi = 0; mi < TILE_M; ++mi) {
          out_tile[mi][0] =
              fma(VEC4_T(in_tile[mi][0][k4i]), weight_texels[0], out_tile[mi][0]);
          out_tile[mi][1] =
              fma(VEC4_T(in_tile[mi][0][k4i]), weight_texels[1], out_tile[mi][1]);
        }
      }
    }
  }

  // Apply bias. The bias tensor is exactly N elements wide (no padding), so
  // for the OOB-N4 thread (n4 + i == N4 when N4 is odd) the load at base = n
  // + i*4 would read past the end. Clamp base to the largest in-bounds 4-N
  // group (n4 = N4 - 1 -> base = (N4 - 1) * 4). The corresponding bias values
  // feed an accumulator slot whose output store is gated by n4 + ni < N4, so
  // the clamped (incorrect) value never reaches memory — only memory safety
  // matters here.
  if (apply_bias > 0) {
    VEC4_T bias[TILE_N4];
    [[unroll]] for (int i = 0; i < TILE_N4; ++i) {
      const int base = min(n + i * 4, (N4 - 1) * 4);
      bias[i] = VEC4_T(
          T(t_bias[base + 0]),
          T(t_bias[base + 1]),
          T(t_bias[base + 2]),
          T(t_bias[base + 3]));
    }
    [[unroll]] for (int mi = 0; mi < TILE_M; ++mi) {
      [[unroll]] for (int ni = 0; ni < TILE_N4; ++ni) {
        out_tile[mi][ni] = out_tile[mi][ni] + bias[ni];
      }
    }
  }

  // Store output tile with bounds checks
  [[unroll]] for (int mi = 0; mi < TILE_M; ++mi) {
    [[unroll]] for (int ni = 0; ni < TILE_N4; ++ni) {
      if (m + mi < M && n4 + ni < N4) {
#ifdef OUTPUT_BUFFER
        t_output[(m + mi) * N4 + n4 + ni] = out_tile[mi][ni];
#else
        imageStore(t_output, ivec3(n4 + ni, m + mi, 0), out_tile[mi][ni]);
#endif
      }
    }
  }
}
