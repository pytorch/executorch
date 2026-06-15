/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// q4gsw linear GEMV kernel — row-pair broadcast dequant-accumulate over the
// shared w_4x8 weight prepack.
//
// Shader naming convention:
//   q4gsw_linear_gemv__w_4x8_<contig>[_nosg]
//   ^^^^^^^^^^^^^^^^^  ^^^^^ ^^^^^^^
//   op base (gemv)     tile  tile arrangement (nc or kc)
//
// Weight binding:
//   The shared pack_q4_linear_weight__w_4x8 shader writes a W_4X8 block-packed uvec2 buffer
//   where the uvec2 at logical tile index [k4, n4] lives at the 2 consecutive
//   uint slots:
//       t_q4_weights[2 * tile_idx + 0] = .x   (row pair {N0, N1})
//       t_q4_weights[2 * tile_idx + 1] = .y   (row pair {N2, N3})
//   Read as a scalar uint buffer, the uint at
//       word_idx = 2 * tile_idx + half   (half in {0, 1})
//   is one N-row-pair's 4 K-step payload. With n2 = 2 * n4 + half and
//   k_slot = k4, under WEIGHT_TILE_CONTIG_DIM=0:
//       word_idx = k_slot * (N/2) + n2
//   which is the index formula used by the per-row-pair loop here.
//
//   Interleaved (dp4a-style) byte-pair layout: each uvec2 lane's 4 bytes hold
//   4 K-consecutive positions for a pair of N rows. As a scalar uint, byte b
//   of w_pack = (N_even, K=b) | (N_odd, K=b) << 4 — the low nibble per byte
//   is the even-N row, the high nibble is odd-N. This byte packing is the
//   natural memory split for the paired-row dequant below and lets the same
//   shader body be repurposed later for int8/int4 integer matmul that
//   operates directly on byte-interleaved nibble pairs.
//
// Scale binding:
//   Uses the production dtype-matched scale bytes but reinterprets them as a
//   gvec2 (vec2 / f16vec2) array. The scale prepack emits vec4 bytes indexed
//   as t_scales[group_i * N4 + n4] where each vec4 holds 4 N-row scales. The
//   same byte layout is addressable as vec2 with index
//       vec2_idx = 2 * (group_i * N4 + n4) + half = group_i * N2 + n2
//   since each vec4 = 2 consecutive vec2 slots (low half = rows {2*n4, 2*n4+1},
//   high half = rows {2*n4+2, 2*n4+3}). Binding as vec2 halves the scale load
//   byte volume and eliminates the 2 wasted components per load.
//
// WG layout: SUBGROUP_SIZE=64, NUM_SUBGROUPS=4. y-dim splits K-blocks across waves.
// Each thread owns one row-pair (n2) and writes two output floats.

#version 450 core

${define_required_extensions(IO_STORAGE, DTYPE)}
${define_required_extensions("buffer", DTYPE)}
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
$if USE_SUBGROUP_BROADCAST:
  #extension GL_KHR_shader_subgroup_basic : require
  #extension GL_KHR_shader_subgroup_ballot : require
  #extension GL_KHR_shader_subgroup_shuffle : require

#define PRECISION ${PRECISION}

#define T ${texel_load_component_type(DTYPE, "buffer")}

$if IO_STORAGE == "buffer":
  #define IO_BUFFER

#define NUM_SUBGROUPS ${NUM_SUBGROUPS}
#define SUBGROUP_SIZE ${SUBGROUP_SIZE}
// Workgroup x-dim size — used for shared-mem indexing in the inter-wave
// reduction. Chosen to match LWG.x set by the host (kGemvSubgroupSize=64).
// In the sg variant, lanes happen to align 1:1 with x-threads when
// subgroupSize >= LWG_X_SIZE; in the nosg variant, x-thread index alone
// addresses shared-mem slots so any subgroup width is safe.
#define LWG_X_SIZE ${LWG_X_SIZE}
// Number of K elements processed per outer k-loop iteration. Format-level
// constant — each iteration loads 8 vec4 of activations (= 32 K-vals) and 8
// uint32 weight packs (4 hi + 4 lo, each holding 4 K-vals × 2 N-rows). Distinct
// from `group_size` (the quantization group): blocks_per_group = group_size /
// K_PER_STEP tells how many consecutive K-blocks share one scale pair.
#define K_PER_STEP 32

#define WEIGHT_TILE_CONTIG_DIM ${WEIGHT_TILE_CONTIG_DIM}

layout(std430) buffer;

// Unified 6-binding layout shared across q4gsw_linear shaders so a single
// DynamicDispatchNode with pick_shader_fn can switch between GEMM and GEMV
// kernels. This shader reads t_fp_input (the raw activation). The
// t_transposed_input binding is declared to preserve slot order but is never
// referenced here — the driver compiles it out to zero runtime cost; only
// the descriptor slot is allocated.
//
// Output: [1, N] scalar DTYPE buffer OR 1x1xN/4 texture3d.
// is_scalar_array is only meaningful for buffer storage; ignored for texture.
${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=True)}
// Activations: [1, K] vec4-packed.
${layout_declare_tensor(B, "r", "t_fp_input", DTYPE, IO_STORAGE, is_scalar_array=False)}
// Unused transposed input — declared only so this shader shares the
// descriptor set layout with the tin GEMM shader.
${layout_declare_tensor(B, "r", "t_transposed_input", DTYPE, "buffer", is_scalar_array=False)}
// Weight: same uvec2 W_4X8 block-packed buffer produced by pack_q4_linear_weight__w_4x8,
// bound here as a scalar uint array so the per-row-pair index math can address
// individual uint slots directly. See header comment for byte-layout proof.
${layout_declare_tensor(B, "r", "t_q4_weights", "int", "buffer")}
// Scales: dtype-matched gvec2 reinterpret of the GEMM vec4 scale prepack.
// Indexed as t_scales[group_idx * N2 + n2].
${layout_declare_tensor(B, "r", "t_scales", DTYPE, "buffer", is_scalar_array=False, vec_size=2)}
// Bias: [N] DTYPE buffer.
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "0")}
// `K` is declared only to keep the spec-constant layout aligned with the GEMM
// shaders so both variants can share a single DynamicDispatchNode with a
// runtime shader picker. It is not referenced in the GEMV body — the local
// `K` (derived from `input_sizes.x`) shadows it inside main().
${layout_declare_spec_const(C, "int", "K", "1024")}
// Quantization group size in elements. `blocks_per_group` (the original GEMV
// spec constant) is recomputed from `group_size` below since K_PER_STEP = 32.
${layout_declare_spec_const(C, "int", "group_size", "32")}

// Inter-wave reduction buffer (NUM_SUBGROUPS - 1 slabs of LWG_X_SIZE vec2).
// Slots are addressed by x-thread index, not subgroup lane index — sized to
// LWG.x so the shader is portable across subgroup widths.
shared vec2 partial_sums[LWG_X_SIZE * (NUM_SUBGROUPS - 1)];

$if not USE_SUBGROUP_BROADCAST:
  // Used by the texture-storage write path to swap acc with the n2-XOR-1
  // partner thread. Replaces subgroupShuffleXor in the nosg variant.
  shared vec2 nosg_n2_partner[LWG_X_SIZE];

// Load a vec4 of activations from input at (vec4 index) idx.
vec4 load_input_vec4(const int idx) {
#ifdef IO_BUFFER
  return vec4(t_fp_input[idx]);
#else
  return vec4(texelFetch(t_fp_input, ivec3(idx, 0, 0), 0));
#endif
}

// Load 2 scales for (n2, group) directly as a gvec2.
// The scale prepack bytes are reinterpreted as gvec2[group_idx * N2 + n2]
// where gvec2 is vec2 (fp32) or f16vec2 (fp16). The vec2(...) cast is a no-op
// for fp32 and an f16 -> f32 widening for fp16.
vec2 load_scale_pair(const int n2, const int group_idx, const int N2) {
  return vec2(t_scales[group_idx * N2 + n2]);
}

void main() {
  $if USE_SUBGROUP_BROADCAST:
    // sg path: lane_id == subgroup invocation; relies on subgroupSize == LWG.x
    // (=64) so subgroup lane and x-thread coincide for shared-mem indexing.
    const uint lane_id = gl_SubgroupInvocationID;
  $else:
    // nosg path: lane_id == x-thread within workgroup; portable across any
    // subgroup width since shared-mem slots are addressed purely by LWG.x.
    const uint lane_id = gl_LocalInvocationID.x;
  const int k_wave_id = int(gl_LocalInvocationID.y);
  const int n2 = int(gl_GlobalInvocationID.x);

  const int N = output_sizes.x;
  const int K = input_sizes.x;
  const int N2 = N / 2;
  const int num_steps = K / K_PER_STEP;
  // Derived from the shared `group_size` spec constant. Each K_PER_STEP (=32)
  // K-block is one "block"; blocks_per_group tells how many consecutive blocks
  // share a single scale pair along K.
  const int blocks_per_group = group_size / K_PER_STEP;

  // Words per K-block in the weight buffer. One K-block covers K_PER_STEP K-vals
  // = K_PER_STEP/4 k4 slices, and each k4 slice is N2 word-pairs. So
  // words_per_k_block = (K_PER_STEP / 4) * N2.
  // `k * K_BLOCK_STRIDE_W` gives the absolute word offset to the start of
  // K-block `k`.
  const int K_BLOCK_STRIDE_W = (K_PER_STEP / 4) * N2;

  if (n2 >= N2) {
    return;
  }

  vec2 acc = vec2(0.0);

  // Loop over k-blocks, waves split k-blocks (k_wave_id, k_wave_id+NUM_SUBGROUPS, ...).
  for (int k = k_wave_id; k < num_steps; k += NUM_SUBGROUPS) {
    // --- Load scale pair for this (n2, group) ---
    const int group_idx = k / blocks_per_group;
    vec2 scale_pair = load_scale_pair(n2, group_idx, N2);
    float scale0 = scale_pair.x;
    float scale1 = scale_pair.y;

    $if USE_SUBGROUP_BROADCAST:
      // --- Load 8 activations per participating lane (only lanes 0..3) ---
      // Lanes 0..3 each load 2 vec4s; other lanes receive via subgroupBroadcast
      // in the dequant loops below.
      vec4 in_vecs[2] = vec4[2](vec4(0.0), vec4(0.0));
      if (lane_id < 4u) {
        const int vec4_base = k * 8 + int(lane_id) * 2;
        in_vecs[0] = load_input_vec4(vec4_base);
        in_vecs[1] = load_input_vec4(vec4_base + 1);
      }
    $else:
      // --- Load all 8 activation vec4s per thread (no subgroup broadcast) ---
      // Each thread independently reads the 32 activations (8 vec4) for this
      // k-block. All lanes hit the same addresses, so L1 serves ~1 load from
      // DRAM per unique address across the wave.
      vec4 in_vecs[8];
      const int vec4_base = k * 8;
      [[unroll]] for (int i = 0; i < 8; ++i) {
        in_vecs[i] = load_input_vec4(vec4_base + i);
      }

    // --- Load 4 int32s for the "hi" half (K positions 0..15) ---
    int w_pack0 = t_q4_weights[n2 + k * K_BLOCK_STRIDE_W + N2 * 0];
    int w_pack1 = t_q4_weights[n2 + k * K_BLOCK_STRIDE_W + N2 * 1];
    int w_pack2 = t_q4_weights[n2 + k * K_BLOCK_STRIDE_W + N2 * 2];
    int w_pack3 = t_q4_weights[n2 + k * K_BLOCK_STRIDE_W + N2 * 3];

    // --- Dequant + accumulate for the "hi" block (K = 0..15 of this K_PER_STEP). ---
    // Each regA word contains interleaved byte pairs: byte b = (N_even, K=b)
    // in the low nibble, (N_odd, K=b) in the high nibble.
    float in_val;

    [[unroll]] for (int k4i = 0; k4i < 4; ++k4i) {
      $if USE_SUBGROUP_BROADCAST:
        in_val = subgroupBroadcast(in_vecs[0][k4i], 0u);
      $else:
        in_val = in_vecs[0][k4i];
      acc.x += (float(int((uint(w_pack0) >> (8 * k4i))     & 0xFu)) - 8.0) * scale0 * in_val;
      acc.y += (float(int((uint(w_pack0) >> (8 * k4i + 4)) & 0xFu)) - 8.0) * scale1 * in_val;
    }

    [[unroll]] for (int k4i = 0; k4i < 4; ++k4i) {
      $if USE_SUBGROUP_BROADCAST:
        in_val = subgroupBroadcast(in_vecs[1][k4i], 0u);
      $else:
        in_val = in_vecs[1][k4i];
      acc.x += (float(int((uint(w_pack1) >> (8 * k4i))     & 0xFu)) - 8.0) * scale0 * in_val;
      acc.y += (float(int((uint(w_pack1) >> (8 * k4i + 4)) & 0xFu)) - 8.0) * scale1 * in_val;
    }

    [[unroll]] for (int k4i = 0; k4i < 4; ++k4i) {
      $if USE_SUBGROUP_BROADCAST:
        in_val = subgroupBroadcast(in_vecs[0][k4i], 1u);
      $else:
        in_val = in_vecs[2][k4i];
      acc.x += (float(int((uint(w_pack2) >> (8 * k4i))     & 0xFu)) - 8.0) * scale0 * in_val;
      acc.y += (float(int((uint(w_pack2) >> (8 * k4i + 4)) & 0xFu)) - 8.0) * scale1 * in_val;
    }

    [[unroll]] for (int k4i = 0; k4i < 4; ++k4i) {
      $if USE_SUBGROUP_BROADCAST:
        in_val = subgroupBroadcast(in_vecs[1][k4i], 1u);
      $else:
        in_val = in_vecs[3][k4i];
      acc.x += (float(int((uint(w_pack3) >> (8 * k4i))     & 0xFu)) - 8.0) * scale0 * in_val;
      acc.y += (float(int((uint(w_pack3) >> (8 * k4i + 4)) & 0xFu)) - 8.0) * scale1 * in_val;
    }

    // --- Load 4 int32s for the "lo" half (K positions 16..31). ---
    w_pack0 = t_q4_weights[n2 + k * K_BLOCK_STRIDE_W + N2 * 4];
    w_pack1 = t_q4_weights[n2 + k * K_BLOCK_STRIDE_W + N2 * 5];
    w_pack2 = t_q4_weights[n2 + k * K_BLOCK_STRIDE_W + N2 * 6];
    w_pack3 = t_q4_weights[n2 + k * K_BLOCK_STRIDE_W + N2 * 7];

    // --- Dequant + accumulate for the "lo" block (K = 16..31). ---
    [[unroll]] for (int k4i = 0; k4i < 4; ++k4i) {
      $if USE_SUBGROUP_BROADCAST:
        in_val = subgroupBroadcast(in_vecs[0][k4i], 2u);
      $else:
        in_val = in_vecs[4][k4i];
      acc.x += (float(int((uint(w_pack0) >> (8 * k4i))     & 0xFu)) - 8.0) * scale0 * in_val;
      acc.y += (float(int((uint(w_pack0) >> (8 * k4i + 4)) & 0xFu)) - 8.0) * scale1 * in_val;
    }

    [[unroll]] for (int k4i = 0; k4i < 4; ++k4i) {
      $if USE_SUBGROUP_BROADCAST:
        in_val = subgroupBroadcast(in_vecs[1][k4i], 2u);
      $else:
        in_val = in_vecs[5][k4i];
      acc.x += (float(int((uint(w_pack1) >> (8 * k4i))     & 0xFu)) - 8.0) * scale0 * in_val;
      acc.y += (float(int((uint(w_pack1) >> (8 * k4i + 4)) & 0xFu)) - 8.0) * scale1 * in_val;
    }

    [[unroll]] for (int k4i = 0; k4i < 4; ++k4i) {
      $if USE_SUBGROUP_BROADCAST:
        in_val = subgroupBroadcast(in_vecs[0][k4i], 3u);
      $else:
        in_val = in_vecs[6][k4i];
      acc.x += (float(int((uint(w_pack2) >> (8 * k4i))     & 0xFu)) - 8.0) * scale0 * in_val;
      acc.y += (float(int((uint(w_pack2) >> (8 * k4i + 4)) & 0xFu)) - 8.0) * scale1 * in_val;
    }

    [[unroll]] for (int k4i = 0; k4i < 4; ++k4i) {
      $if USE_SUBGROUP_BROADCAST:
        in_val = subgroupBroadcast(in_vecs[1][k4i], 3u);
      $else:
        in_val = in_vecs[7][k4i];
      acc.x += (float(int((uint(w_pack3) >> (8 * k4i))     & 0xFu)) - 8.0) * scale0 * in_val;
      acc.y += (float(int((uint(w_pack3) >> (8 * k4i + 4)) & 0xFu)) - 8.0) * scale1 * in_val;
    }
  }

  // --- Inter-wave reduction via flat shared memory (matches OpenCL) ---
  if (k_wave_id >= 1) {
    partial_sums[(k_wave_id - 1) * LWG_X_SIZE + int(lane_id)] = acc;
  }
  barrier();
  if (k_wave_id == 0) {
    [[unroll]] for (int w = 0; w < NUM_SUBGROUPS - 1; ++w) {
      acc += partial_sums[w * LWG_X_SIZE + int(lane_id)];
    }

    // Apply bias if present
    if (apply_bias > 0) {
      acc.x += float(t_bias[n2 * 2]);
      acc.y += float(t_bias[n2 * 2 + 1]);
    }
  }

  // --- Write 2 outputs ---
#ifdef IO_BUFFER
  if (k_wave_id == 0) {
    t_output[n2 * 2] = T(acc.x);
    t_output[n2 * 2 + 1] = T(acc.y);
  }
#else
  // texture3d: output stored as width-packed vec4 at (n4, 0, 0).
  // Each thread owns 2 outputs (n2*2, n2*2+1). Two consecutive n2s share
  // one vec4; only the even-n2 thread assembles and writes the full vec4.
  $if USE_SUBGROUP_BROADCAST:
    vec2 partner = vec2(
        subgroupShuffleXor(acc.x, 1u),
        subgroupShuffleXor(acc.y, 1u));
    if (k_wave_id == 0 && (n2 & 1) == 0) {
      vec4 out_vec;
      out_vec.xy = acc;
      out_vec.zw = partner;
      const int n4 = n2 / 2;
      imageStore(t_output, ivec3(n4, 0, 0), out_vec);
    }
  $else:
    // Subgroup-free partner exchange via shared memory. Only k_wave_id==0
    // threads have a valid reduced `acc`, so only those threads write the
    // partner slot; all threads must reach the barrier (uniform control
    // flow). Then the even-n2 k_wave_id==0 threads read the n2-XOR-1
    // partner slot and assemble the output vec4. A barrier before the
    // write resynchronizes after the inter-wave reduction read of
    // partial_sums (which conflicts with the partner-exchange shared
    // memory only conceptually — they are separate arrays — but the
    // pre-barrier matches the OpenCL reference style and is cheap).
    barrier();
    if (k_wave_id == 0) {
      nosg_n2_partner[lane_id] = acc;
    }
    barrier();
    if (k_wave_id == 0 && (n2 & 1) == 0) {
      vec2 partner = nosg_n2_partner[lane_id ^ 1u];
      vec4 out_vec;
      out_vec.xy = acc;
      out_vec.zw = partner;
      const int n4 = n2 / 2;
      imageStore(t_output, ivec3(n4, 0, 0), out_vec);
    }
#endif
}
