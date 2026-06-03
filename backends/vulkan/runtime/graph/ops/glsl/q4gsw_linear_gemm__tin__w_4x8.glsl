/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Adreno-optimized GEMM kernel for q4gsw weights with vectorized SSBO
// activations.
//
// The input activation buffer is [K * ceil(M/4)] vec4 elements, transposed
// from [M, K] row-major. Element at index [k * M4 + m4] holds 4 consecutive
// activations at K=k, M=m4*4..m4*4+3. The element type matches ACC_DTYPE:
// f16vec4 for half, vec4 for float.
//
// Output can be buffer or texture3D. Weights and activations are always
// buffers.
//
// Tile shape: TILE_M x TILE_N per thread. Weights in W_4X8 block-packed uvec2 format.
//
// Weight tile layout (4K x 4N uvec2), interleaved (dp4a-style) byte pairs:
//   Each byte of the 32-bit lane holds one (N_even, N_odd) nibble pair at a
//   fixed K. Byte b of .x = (N0, K=b) | (N1, K=b) << 4;
//   byte b of .y = (N2, K=b) | (N3, K=b) << 4. The low nibble per byte is the
//   even-N row; the high nibble is the odd-N row. This is the natural memory
//   split for the unified u16vec4 hoist below (no repack) and lets the same
//   shader body be repurposed later for int8/int4 integer matmul that
//   operates directly on byte-interleaved nibble pairs.
//
// Weight storage variants (selected by WEIGHT_STORAGE):
//   "buffer"    (nc) — ivec2 buffer view of pack_q4_linear_weight__w_4x8_nc.
//                      One ivec2 per (k4, n_tile) at flat index
//                      `k4 * N4_padded + n_tile`. Per-thread tile is 8M x 4N
//                      (= one n4 tile), so each thread consumes the full
//                      ivec2 it loads.
//   "texture2d" (kc) — ivec4 image2D from
//                      pack_q4_linear_weight__w_4x8_kc_texture2d. Each texel
//                      covers 4K x 8N. Per-thread tile is still 8M x 4N, so
//                      the thread fetches one ivec4 at (k4, n_tile/2) and
//                      uses only its half (.xy when n_tile is even, .zw when
//                      odd). The adjacent-N thread fetching the SAME texel
//                      coord hits the texture cache, so the unused-half
//                      "waste" is mostly absorbed at the cache layer. The
//                      primary benefit is sharing the prepack tensor with
//                      the kc coop GEMV and routing weight reads through
//                      the texture cache on Adreno.
//
// codegen-nosub

#version 450 core

${define_required_extensions(OUT_STORAGE, DTYPE)}
${define_required_extensions(IN_STORAGE, DTYPE)}
${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}

#define TILE_M ${TILE_M}
#define TILE_N ${TILE_N}
#define TILE_M4 (TILE_M / 4)

$if OUT_STORAGE == "buffer":
  #define OUTPUT_BUFFER

$if WEIGHT_STORAGE == "texture2d":
  #define WEIGHT_TEX2D

$if WEIGHT_KC == 1:
  #define WEIGHT_KC

// 16-bit integer types (int16_t / uint16_t / u16vec4) are used directly in
// the nibble bit-manipulation path regardless of DTYPE — the extract is
// orthogonal to FP precision and int16 saves a register per value on Adreno.
// The unified u16vec4 hoist splits nib_pack into {lo16(.x), hi16(.x),
// lo16(.y), hi16(.y)}; this is the natural memory split for the interleaved
// byte-pair packing (no repack required).
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require

// Accumulation dtype is derived from DTYPE: fp16 IO -> f16 accum (2× ALU
// throughput on Adreno), fp32 IO -> f32 accum. Output binding (OUT_VEC4_T)
// collapses to the same type since t_output also uses DTYPE.
$if DTYPE == "half":
  #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
  #define ACC_VEC4_T f16vec4
  #define ACC_SCALAR_T float16_t
  #define ACC_ZERO ACC_VEC4_T(0.0hf)
  #define ACC_LITERAL(x) float16_t(x)
  #define OUT_VEC4_T f16vec4
$else:
  #define ACC_VEC4_T vec4
  #define ACC_SCALAR_T float
  #define ACC_ZERO ACC_VEC4_T(0.0)
  #define ACC_LITERAL(x) float(x)
  #define OUT_VEC4_T vec4

layout(std430) buffer;

// Unified 6-binding layout shared across q4gsw_linear shaders so a single
// DynamicDispatchNode with pick_shader_fn can switch between GEMM and GEMV
// kernels. This shader reads t_transposed_input (pre-transposed activation
// vectorized along M). The t_fp_input binding is declared to preserve slot
// order but is never referenced here — the driver compiles it out to zero
// runtime cost; only the descriptor slot is allocated.
//
// Output: [M, N] tensor, buffer or texture3D
${layout_declare_tensor(B, "w", "t_output", DTYPE, OUT_STORAGE, is_scalar_array=False)}

// Unused fp_input — declared only so this shader shares the descriptor set
// layout with the fp32 GEMM and GEMV shaders. IN_STORAGE is passed in from
// the YAML so texture3d / buffer variants pick the right image type.
${layout_declare_tensor(B, "r", "t_fp_input", DTYPE, IN_STORAGE, is_scalar_array=False)}

// Input activations: vec4 buffer [K * ceil(M/4)] in the source dtype; cast to
// ACC_VEC4_T on load so the transposed tensor preserves the input dtype and
// avoids a preprocess-time cast.
${layout_declare_tensor(B, "r", "t_transposed_input", DTYPE, "buffer", is_scalar_array=False)}

// Nibble weight binding (unified ivec4 form across all three storage paths):
//   nc        = ivec4 buffer (one ivec4 per (k4, n8) covering 4K x 8N, flat
//               index `k4 * (N4_padded / 2) + n8`).
//   kc Tex2D  = ivec4 image2D (one ivec4 per (k4, n8) covering 4K x 8N).
//   kc Buffer = ivec4 SSBO    (one ivec4 per (k4, n8) covering 4K x 8N, flat
//                              index `n8 * K4 + k4`).
// Per-thread tile is 8M x 4N (TILE_N = 4 = half of 4K x 8N). Each thread
// fetches one ivec4 and uses only its half (.xy when n_tile is even, .zw when
// odd). The adjacent-n_tile thread fetches the same coordinate, so the
// unused-half "waste" is absorbed at the cache layer.
${layout_declare_tensor(B, "r", "t_q4_weights", "int", WEIGHT_STORAGE, is_scalar_array=False, vec_size=4)}

// Scales: vec4 buffer [(K/gs) * (N/4)] in the source dtype; cast to ACC_VEC4_T
// on load.
${layout_declare_tensor(B, "r", "t_scales", DTYPE, "buffer", is_scalar_array=False)}

// Bias: float buffer [N]
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer")}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
// Unused input_sizes — declared only so this shader's descriptor set layout
// matches the dispatch's 2-UBO ParamsBindList (output + input sizes), which is
// shared with the fp32 GEMM and GEMV shaders so a single DynamicDispatchNode
// can switch shader at run time. Mali drivers (Tensor G4 / Immortalis G715)
// SIGSEGV in vkUpdateDescriptorSets when the pool writes a UBO descriptor at
// a binding that does not exist in the layout; Adreno tolerates it. The
// shader body does not reference input_sizes — the driver compiles the
// binding out to zero runtime cost; only the descriptor slot is allocated.
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "0")}
${layout_declare_spec_const(C, "int", "K", "1024")}
${layout_declare_spec_const(C, "int", "group_size", "32")}

void store_output(const int row, const int n_tile, const int N4, const vec4 result) {
#ifdef OUTPUT_BUFFER
  t_output[row * N4 + n_tile] = OUT_VEC4_T(result);
#else
  imageStore(t_output, ivec3(n_tile, row, 0), result);
#endif
}

void store_row(
    const int row,
    const int n_tile,
    const int N4,
    const ACC_VEC4_T acc_col,
    const ACC_VEC4_T bias_val) {
  store_output(row, n_tile, N4, vec4(
      float(acc_col.x + bias_val.x),
      float(acc_col.y + bias_val.y),
      float(acc_col.z + bias_val.z),
      float(acc_col.w + bias_val.w)));
}

void main() {
  const int m_tile = int(gl_GlobalInvocationID.x); // token group index
  const int n_tile = int(gl_GlobalInvocationID.y); // output feature group index

  const int M = output_sizes.y;
  const int N = output_sizes.x;

  const int n = n_tile * TILE_N;
  const int N4 = N / 4;
  // Padded N4 row stride for the W_4X8 block-packed weight buffer (next-even N4). Must
  // match the prepack's allocation (see prepack_q4_w_4x8_nc_buffer in
  // Q4gswLinear.cpp) — for N % 8 != 0 the buffer's row stride differs from
  // unpadded N4 and the weight read below must use the padded stride to land
  // on the correct (k4, n_tile) ivec2 slot. No-op when N % 8 == 0.
  const int N4_padded = (N4 + 1) & ~1;
  const int m = m_tile * TILE_M;
  const int M4 = (M + 3) / 4;
  const bool full_m_tile = (m + TILE_M <= M);

  if (n >= N || m >= M) {
    return;
  }

  // acc_T[mi]: mi=0..TILE_M-1 (M positions). Each vec4 holds 4 N-channel values
  // for one M position — accumulators stored along N (transposed vs the prior
  // [TILE_N][TILE_M4] layout). This lets the inner-loop MAC
  //   acc_T[mi] += B[m4][m_in_m4] * dw_vec
  // fuse 4 N-channel MACs into a single mad.f16 (rpt3) packed instruction
  // because the 4 N components live in adjacent half-reg slots.
  ACC_VEC4_T acc_T[TILE_M];
  [[unroll]] for (int mi = 0; mi < TILE_M; ++mi) {
    acc_T[mi] = ACC_ZERO;
  }

  const int K4 = K / 4;
  // n8 = n_tile / 2 — every two adjacent n_tiles share one ivec4 weight
  // entry at coord (k4, n8). The .xy half corresponds to even n_tile (n4_a),
  // .zw to odd n_tile (n4_b).
  const int n8 = n_tile >> 1;
  const bool n_tile_is_odd = (n_tile & 1) != 0;
  for (int k = 0; k < K; k += 4) {
#if defined(WEIGHT_TEX2D) && defined(WEIGHT_KC)
    // Tex2D (kc) path. Fetch the full 4K x 8N texel and pick the ivec2 half
    // that owns this thread's n_tile. The adjacent-n_tile thread fetches
    // the SAME texel coord and hits the texture cache.
    const ivec4 w_texel = texelFetch(t_q4_weights, ivec2(k >> 2, n8), 0);
#elif defined(WEIGHT_TEX2D)
    // Tex2D (nc) path. Image position (n8, k4); same 4K x 8N byte-pair
    // payload as the nc-buffer variant routed through the texture cache.
    // Adjacent-n_tile thread fetching the SAME coord absorbs the unused-half
    // "waste" via the texture cache.
    const ivec4 w_texel = texelFetch(t_q4_weights, ivec2(n8, k >> 2), 0);
#elif defined(WEIGHT_KC)
    // kc dense Buffer path. Same 4K x 8N payload as the Tex2D variant, indexed
    // at flat index `n8 * K4 + k4`. Pick .xy or .zw based on n_tile parity.
    const ivec4 w_texel = t_q4_weights[n8 * K4 + (k >> 2)];
#else
    // Buffer (nc) path. Same 4K x 8N ivec4 payload, indexed at flat index
    // `k4 * (N4_padded / 2) + n8`. N4_padded is even by construction, so
    // (N4_padded / 2) gives the row stride in ivec4 units. Byte-identical to
    // the prior ivec2 layout: the prior ivec2 at (k4, 2*n8) lives at scalar
    // index 2*(k4 * N4_padded + 2*n8) = 4*(k4 * N8 + n8), which is the .xy
    // half of this ivec4; the ivec2 at (k4, 2*n8 + 1) is the .zw half.
    const ivec4 w_texel = t_q4_weights[(k >> 2) * (N4_padded >> 1) + n8];
#endif
    const ivec2 nib_pack = n_tile_is_odd ? w_texel.zw : w_texel.xy;

    // Unified hoist — zero-ALU memory split that matches the interleaved
    // byte-pair layout directly. bits[0..3] = {(N0,N1)@K0,1; (N0,N1)@K2,3;
    // (N2,N3)@K0,1; (N2,N3)@K2,3}.
    u16vec4 bits = u16vec4(
        uint16_t(uint(nib_pack.x) & 0xFFFFu),
        uint16_t(uint(nib_pack.x) >> 16),
        uint16_t(uint(nib_pack.y) & 0xFFFFu),
        uint16_t(uint(nib_pack.y) >> 16));

    const ACC_VEC4_T scale = t_scales[(k / group_size) * N4 + n_tile];

    [[unroll]] for (int k_inner = 0; k_inner < 4; ++k_inner) {
      // Load activations once per K sub-step, reuse across all N-channels.
      // Cast from the source input dtype into the accumulation dtype here.
      ACC_VEC4_T B[TILE_M4];
      [[unroll]] for (int m4_inner = 0; m4_inner < TILE_M4; ++m4_inner) {
        const int m4 = m_tile * TILE_M4 + m4_inner;
        B[m4_inner] = t_transposed_input[(k + k_inner) * M4 + m4];
      }

      // Build dw_vec packed across the 4 n_inner channels at this k_inner.
      // Interleaved byte layout: byte b of nib_pack.x = (N0,K=b)|(N1,K=b)<<4
      // and byte b of nib_pack.y = (N2,K=b)|(N3,K=b)<<4. The u16 split above
      // therefore gives bits[0..3] = {(N0,N1)@K0,1; (N0,N1)@K2,3;
      // (N2,N3)@K0,1; (N2,N3)@K2,3}. All indices compile-time constants
      // under [[unroll]].
      //
      // Nibble extract stays in int16 the whole way: shift+mask in u16,
      // subtract 8 in i16, convert directly i16 -> ACC_SCALAR_T. Avoids
      // an intermediate int (32-bit) that the compiler would have to keep
      // live alongside the f16 accumulators, costing a fp32 register and
      // pushing AOC occupancy below the 50% threshold on Adreno 750.
      ACC_VEC4_T dw_vec;
      [[unroll]] for (int n_inner = 0; n_inner < TILE_N; ++n_inner) {
        const int lane = 2 * (n_inner >> 1) + (k_inner >> 1);
        const int shift = 8 * (k_inner & 1) + 4 * (n_inner & 1);
        int16_t nibble = int16_t((bits[lane] >> int16_t(shift)) & uint16_t(0xFu)) - int16_t(8);
        dw_vec[n_inner] = ACC_SCALAR_T(nibble) * scale[n_inner];
      }

      // FMA all TILE_M positions against the packed dw_vec. The (rpt3) packing
      // happens here: acc_T[mi] += B_scalar * dw_vec is a single
      // mad.f16 (rpt3) over the 4 adjacent N-channel half-reg slots.
      [[unroll]] for (int m4_inner = 0; m4_inner < TILE_M4; ++m4_inner) {
        [[unroll]] for (int m_in_m4 = 0; m_in_m4 < 4; ++m_in_m4) {
          acc_T[m4_inner * 4 + m_in_m4] += B[m4_inner][m_in_m4] * dw_vec;
        }
      }
    }
  }

  // Bias values (loaded once, reused for all stores)
  ACC_VEC4_T bias_val = ACC_ZERO;
  if (apply_bias > 0) {
    bias_val = ACC_VEC4_T(
        ACC_LITERAL(t_bias[n + 0]),
        ACC_LITERAL(t_bias[n + 1]),
        ACC_LITERAL(t_bias[n + 2]),
        ACC_LITERAL(t_bias[n + 3]));
  }

  // Output store. With acc_T transposed (each vec4 = 4 N-channels at one M
  // position), each row stores directly without re-shuffling — the compiler
  // can issue the bias-add as another mad.f16 (rpt3) over the same N-lane
  // register block.
  if (n + TILE_N - 1 < N) {
    for (int h = 0; h < TILE_M4; ++h) {
      if (h > 0 && m + h * 4 >= M) {
        break;
      }
      if (full_m_tile) {
        store_row(m + h * 4 + 0, n_tile, N4, acc_T[h * 4 + 0], bias_val);
        store_row(m + h * 4 + 1, n_tile, N4, acc_T[h * 4 + 1], bias_val);
        store_row(m + h * 4 + 2, n_tile, N4, acc_T[h * 4 + 2], bias_val);
        store_row(m + h * 4 + 3, n_tile, N4, acc_T[h * 4 + 3], bias_val);
      } else {
        [[unroll]] for (int m_in_m4 = 0; m_in_m4 < 4; ++m_in_m4) {
          const int row = m + h * 4 + m_in_m4;
          if (row < M) {
            store_row(
                row, n_tile, N4, acc_T[h * 4 + m_in_m4], bias_val);
          }
        }
      }
    }
  }
}
