/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// q4gsw linear GEMV — cooperative reduction variant consuming the W_4X8
// byte-pair weight packing. Switches between kc-contiguous and nc-contiguous
// weight orderings via WEIGHT_KC, and between Tex2D image and SSBO buffer
// weight bindings via WEIGHT_STORAGE.
//
// Naming: q4gsw_linear_gemv_coop__w_4x8_<kc|nc>_<weight_storage>_<io_storage>_<dtype>
//
// Mirrors LEGACY linear_q4gsw_coop's dispatch shape (LWG=(1,1,64), one WG
// per n8 tile = 8 N-outputs, lanes cooperate along K) but reads the W_4X8
// byte-pair nibble layout produced by pack_q4_linear_weight__w_4x8_kc_texture2d.
//
// Block structure: each weight texel is 4K x 8N. The 4 ints of the ivec4
// hold byte-pair nibbles for two consecutive n4 tiles at the SAME k4:
//   texel.x byte b = (N=4*n4_a+0, K=k4*4+b) | (N=4*n4_a+1, K=k4*4+b) << 4
//   texel.y byte b = (N=4*n4_a+2, K=k4*4+b) | (N=4*n4_a+3, K=k4*4+b) << 4
//   texel.z byte b = (N=4*n4_b+0, K=k4*4+b) | (N=4*n4_b+1, K=k4*4+b) << 4
//   texel.w byte b = (N=4*n4_b+2, K=k4*4+b) | (N=4*n4_b+3, K=k4*4+b) << 4
// where n4_a = 2*n8, n4_b = 2*n8 + 1, b in {0,1,2,3}. The low nibble of each
// byte is the "lower" N row of the pair; the high nibble is the "upper".
//
// Lanes split K4 = K/4 texels round-robin across the WORKERS_PER_GROUP lanes of
// a worker group; each lane fetches one texel per K-step (4 K-vals * 8 N-rows
// = 32 FMAs). A shared-mem tree reduction collapses the per-lane partial sums
// (8 N values each) into the final 8 outputs for that group.
//
// Generalized layout: each WG hosts NUM_GROUPS independent worker groups along
// the y-axis; each group cooperates over K with WORKERS_PER_GROUP workers
// along the z-axis. One WG produces NUM_GROUPS * 8 output values (NUM_GROUPS
// consecutive n8 tiles). LWG = (1, NUM_GROUPS, WORKERS_PER_GROUP). For
// NUM_GROUPS == 1, WORKERS_PER_GROUP == 64 the dispatch is identical to the
// pre-generalization shape (LWG=(1,1,64), one WG per n8 tile).

#version 450 core

${define_required_extensions(IO_STORAGE, DTYPE)}
${define_required_extensions("buffer", DTYPE)}
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, IO_STORAGE)}
#define T ${texel_load_component_type(DTYPE, IO_STORAGE)}

$if IO_STORAGE == "buffer":
  #define IO_BUFFER

$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

$if WEIGHT_KC == 1:
  #define WEIGHT_KC

#define NUM_GROUPS ${NUM_GROUPS}
#define WORKERS_PER_GROUP ${WORKERS_PER_GROUP}
// Backwards-compatible alias — historical name for the per-group worker count.
// The K-loop strides by WGS, the tree reduction halves WGS, and the partial-sum
// shared memory slabs are sized WGS deep per group.
#define WGS WORKERS_PER_GROUP

layout(std430) buffer;

// Unified 6-binding layout shared across q4gsw_linear shaders so a single
// DynamicDispatchNode with pick_shader_fn can switch between GEMM and GEMV
// kernels. This shader reads:
//   - t_fp_input          (raw activation)
//   - t_q4_weights_tex2d  (ivec4 image, kc dense form, 4K x 8N per texel)
//   - t_scales            (gvec2 scales)
//   - t_bias              (optional bias)
//
// t_transposed_input is declared to keep the descriptor slot order in sync
// with the tin GEMM shader; never referenced (compiles out).

// Output: [1, N] scalar DTYPE buffer OR 1x1xN/4 texture3d.
${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=True)}
// Activations: [1, K] vec4-packed.
${layout_declare_tensor(B, "r", "t_fp_input", DTYPE, IO_STORAGE, is_scalar_array=False)}
// Unused — kept for descriptor-set parity with tin GEMM.
${layout_declare_tensor(B, "r", "t_transposed_input", DTYPE, "buffer", is_scalar_array=False)}
// Weight: same 4K x 8N byte-pair payload across all 4 (storage, layout)
// variants; only the binding type and fetch coordinate change:
//   WEIGHT_STORAGE == "texture2d", WEIGHT_KC == 1: ivec4 image2D, texel at
//       (k4, n8) (kc-contiguous along x — texture cache path).
//   WEIGHT_STORAGE == "buffer",    WEIGHT_KC == 1: ivec4 SSBO, indexed at
//       `n8 * K4 + k4` (SSBO cache path).
//   WEIGHT_STORAGE == "texture2d", WEIGHT_KC == 0: ivec4 image2D, texel at
//       (n8, k4) (nc-contiguous along x — texture cache path).
//   WEIGHT_STORAGE == "buffer",    WEIGHT_KC == 0: ivec4 SSBO, indexed at
//       `k4 * N8 + n8` (nc-contiguous; same payload as
//       `pack_q4_linear_weight__w_4x8_nc_buffer`).
${layout_declare_tensor(B, "r", "t_q4_weights", "int", WEIGHT_STORAGE, is_scalar_array=False, vec_size=4)}
// Scales: dtype-matched gvec2 reinterpret of the GEMM vec4 scale prepack.
// Indexed as t_scales[group_idx * N2 + n2]; one gvec2 covers 2 consecutive
// N rows (the low/high pair within an n4 tile).
${layout_declare_tensor(B, "r", "t_scales", DTYPE, "buffer", is_scalar_array=False, vec_size=2)}
// Bias: [N] DTYPE buffer.
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "0")}
// Aligned with the rest of the q4gsw_linear shader family. K is unused here
// (the local one derived from input_sizes shadows it); kept to share
// descriptor + spec-constant layout.
${layout_declare_spec_const(C, "int", "K", "1024")}
${layout_declare_spec_const(C, "int", "group_size", "32")}

// Shared memory for the cooperative reduction. Each lane writes 8 partial
// floats (one per N row in the n8 tile = 2 vec4) at the end of its K loop;
// lane 0 of each group then sums all WGS slabs of that group and writes the
// 8 outputs. Stored as 2 adjacent vec4 slabs of NUM_GROUPS * WGS lanes — slot
// for (group_id, lid) is `group_id * WGS + lid`.
shared vec4 partial_sums_a[NUM_GROUPS * WGS];
shared vec4 partial_sums_b[NUM_GROUPS * WGS];

// Load a vec4 of activations from input at vec4 index `idx`.
vec4 load_input_vec4(const int idx) {
#ifdef IO_BUFFER
  return vec4(t_fp_input[idx]);
#else
  return vec4(texelFetch(t_fp_input, ivec3(idx, 0, 0), 0));
#endif
}

// Load 2 scales for (n2, group). The scale prepack stores [K/gs, N] floats
// reinterpreted as gvec2[group_idx * N2 + n2].
vec2 load_scale_pair(const int n2, const int group_idx, const int N2) {
  return vec2(t_scales[group_idx * N2 + n2]);
}

void main() {
  // Each WG hosts NUM_GROUPS independent worker groups along y; each group
  // cooperates over K with WORKERS_PER_GROUP workers along z. NUM_GROUPS == 1
  // and WORKERS_PER_GROUP == 64 reproduces the original 1-group / 64-worker
  // dispatch.
  const int wg_n8_base = int(gl_WorkGroupID.x) * NUM_GROUPS;
  const int group_id = int(gl_LocalInvocationID.y);
  const int n8 = wg_n8_base + group_id;
  const int lid = int(gl_LocalInvocationID.z);

  // Per-group base offset into the shared-mem partial-sum slabs.
  const int group_slab_base = group_id * WGS;

  const int N = output_sizes.x;
  const int K = input_sizes.x;
  const int N4 = (N + 3) / 4;
  const int N2 = N / 2;
  const int K4 = K / 4; // texels along K
  // N8 = ceil(N4/2). Only referenced by the nc-buffer weight fetch path.
  const int N8 = (N4 + 1) / 2;

  // Bound the n8 dimension. Each group owns 8 N rows = 1 n8 tile = 2 n4 tiles
  // (n4_a = 2*n8, n4_b = 2*n8 + 1). When NUM_GROUPS > 1, an individual group
  // may be OOB while peers are valid — in that case we skip the K-loop and
  // output store but still hit the shared-mem barriers below so the reduction
  // remains well-defined for the valid groups. For NUM_GROUPS == 1 every
  // thread either is valid or returns together, identical to the original.
  const bool group_valid = (n8 * 2 < N4);
  if (!group_valid && wg_n8_base * 2 >= N4) {
    // Whole WG OOB — safe to return for all threads.
    return;
  }

  const int n4_a = 2 * n8;
  const int n4_b = 2 * n8 + 1;

  // n2 indices for the two n4 tiles in this n8 (4 scale pairs per group).
  const int n2_a_lo = 2 * n4_a;     // rows n4_a*4+0, n4_a*4+1
  const int n2_a_hi = 2 * n4_a + 1; // rows n4_a*4+2, n4_a*4+3
  const int n2_b_lo = 2 * n4_b;     // rows n4_b*4+0, n4_b*4+1
  const int n2_b_hi = 2 * n4_b + 1; // rows n4_b*4+2, n4_b*4+3

  // Quantization grouping along K. Each k_step (= 4 K-vals = 1 texel) is one
  // "block"; multiple blocks may share a scale pair.
  // K_PER_TEXEL = 4 (texel covers 4 K-vals).
  const int blocks_per_group = group_size / 4;

  // Per-thread accumulators for the 8 N rows = 2 vec4 (n4_a and n4_b).
  vec4 acc_a = vec4(0.0);
  vec4 acc_b = vec4(0.0);

  int cur_group = -1;
  vec2 sc_a_lo = vec2(0.0);
  vec2 sc_a_hi = vec2(0.0);
  vec2 sc_b_lo = vec2(0.0);
  vec2 sc_b_hi = vec2(0.0);

  // Skip the K-loop for OOB groups so they don't fetch invalid weight indices,
  // but they still hit the shared-mem stores/barriers below with zero acc so
  // the per-group tree reduction stays well-defined for valid peer groups.
  const int K4_eff = group_valid ? K4 : 0;
  for (int k4 = lid; k4 < K4_eff; k4 += WGS) {
    // Update scales when crossing into a new group.
    const int group_idx = k4 / blocks_per_group;
    if (group_idx != cur_group) {
      sc_a_lo = load_scale_pair(n2_a_lo, group_idx, N2);
      sc_a_hi = load_scale_pair(n2_a_hi, group_idx, N2);
      sc_b_lo = load_scale_pair(n2_b_lo, group_idx, N2);
      sc_b_hi = load_scale_pair(n2_b_hi, group_idx, N2);
      cur_group = group_idx;
    }

    // Load 1 ivec4 weight = 4 K-vals × 8 N-rows. Same byte-pair payload across
    // all 4 (storage, layout) variants; only the binding type and fetch
    // coordinate differ.
#if defined(WEIGHT_BUFFER) && defined(WEIGHT_KC)
    // kc dense Buffer: SSBO indexed at `n8 * K4 + k4`.
    const ivec4 w_texel = t_q4_weights[n8 * K4 + k4];
#elif defined(WEIGHT_BUFFER)
    // nc Buffer: SSBO indexed at `k4 * N8 + n8`.
    const ivec4 w_texel = t_q4_weights[k4 * N8 + n8];
#elif defined(WEIGHT_KC)
    // kc dense Tex2D: image position (k4, n8).
    const ivec4 w_texel = texelFetch(t_q4_weights, ivec2(k4, n8), 0);
#else
    // nc Tex2D: image position (n8, k4).
    const ivec4 w_texel = texelFetch(t_q4_weights, ivec2(n8, k4), 0);
#endif
    const uint w_a_lo = uint(w_texel.x); // n4_a rows {0,1}, K {b=0..3}
    const uint w_a_hi = uint(w_texel.y); // n4_a rows {2,3}, K {b=0..3}
    const uint w_b_lo = uint(w_texel.z); // n4_b rows {0,1}, K {b=0..3}
    const uint w_b_hi = uint(w_texel.w); // n4_b rows {2,3}, K {b=0..3}

    // Load 4 activations (= 1 vec4) for K positions [k4*4, k4*4+4).
    const vec4 in_v = load_input_vec4(k4);

    // Dequant + accumulate. For K-byte b in {0..3}:
    //   nibble for row r is ((w >> (8*b + 4*(r&1))) & 0xF) - 8
    //   row 0 = w_*_lo low,  row 1 = w_*_lo high
    //   row 2 = w_*_hi low,  row 3 = w_*_hi high
    [[unroll]] for (int b = 0; b < 4; ++b) {
      const float a = in_v[b];
      // n4_a:
      const int a0 = int((w_a_lo >> (8 * b))     & 0xFu) - 8;
      const int a1 = int((w_a_lo >> (8 * b + 4)) & 0xFu) - 8;
      const int a2 = int((w_a_hi >> (8 * b))     & 0xFu) - 8;
      const int a3 = int((w_a_hi >> (8 * b + 4)) & 0xFu) - 8;
      acc_a.x += float(a0) * sc_a_lo.x * a;
      acc_a.y += float(a1) * sc_a_lo.y * a;
      acc_a.z += float(a2) * sc_a_hi.x * a;
      acc_a.w += float(a3) * sc_a_hi.y * a;
      // n4_b:
      const int b0 = int((w_b_lo >> (8 * b))     & 0xFu) - 8;
      const int b1 = int((w_b_lo >> (8 * b + 4)) & 0xFu) - 8;
      const int b2 = int((w_b_hi >> (8 * b))     & 0xFu) - 8;
      const int b3 = int((w_b_hi >> (8 * b + 4)) & 0xFu) - 8;
      acc_b.x += float(b0) * sc_b_lo.x * a;
      acc_b.y += float(b1) * sc_b_lo.y * a;
      acc_b.z += float(b2) * sc_b_hi.x * a;
      acc_b.w += float(b3) * sc_b_hi.y * a;
    }
  }

  // Cooperative tree reduction across the WGS lanes within each group. All
  // threads (including lanes of OOB groups) participate in the barriers; OOB
  // groups simply reduce zeros into their slab. Slot for (group_id, lid) is
  // `group_id * WGS + lid`.
  partial_sums_a[group_slab_base + lid] = acc_a;
  partial_sums_b[group_slab_base + lid] = acc_b;
  memoryBarrierShared();
  barrier();

  for (int i = WGS / 2; i > 0; i /= 2) {
    if (lid < i) {
      partial_sums_a[group_slab_base + lid] +=
          partial_sums_a[group_slab_base + lid + i];
      partial_sums_b[group_slab_base + lid] +=
          partial_sums_b[group_slab_base + lid + i];
    }
    memoryBarrierShared();
    barrier();
  }

  // Only lane 0 of each valid group writes the 8 outputs for its n8 tile.
  if (lid != 0 || !group_valid) {
    return;
  }

  vec4 out_a = partial_sums_a[group_slab_base];
  vec4 out_b = partial_sums_b[group_slab_base];

  if (apply_bias > 0) {
    const int n_base_a = n4_a * 4;
    const int n_base_b = n4_b * 4;
    out_a.x += float(t_bias[n_base_a + 0]);
    out_a.y += float(t_bias[n_base_a + 1]);
    out_a.z += float(t_bias[n_base_a + 2]);
    out_a.w += float(t_bias[n_base_a + 3]);
    out_b.x += float(t_bias[n_base_b + 0]);
    out_b.y += float(t_bias[n_base_b + 1]);
    out_b.z += float(t_bias[n_base_b + 2]);
    out_b.w += float(t_bias[n_base_b + 3]);
  }

#ifdef IO_BUFFER
  const int n_base_a = n4_a * 4;
  const int n_base_b = n4_b * 4;
  // Bounds-checked scalar writes (N may not be a multiple of 8).
  if (n_base_a + 0 < N) t_output[n_base_a + 0] = T(out_a.x);
  if (n_base_a + 1 < N) t_output[n_base_a + 1] = T(out_a.y);
  if (n_base_a + 2 < N) t_output[n_base_a + 2] = T(out_a.z);
  if (n_base_a + 3 < N) t_output[n_base_a + 3] = T(out_a.w);
  if (n_base_b + 0 < N) t_output[n_base_b + 0] = T(out_b.x);
  if (n_base_b + 1 < N) t_output[n_base_b + 1] = T(out_b.y);
  if (n_base_b + 2 < N) t_output[n_base_b + 2] = T(out_b.z);
  if (n_base_b + 3 < N) t_output[n_base_b + 3] = T(out_b.w);
#else
  // texture3d: output stored as width-packed vec4 at (n4, 0, 0).
  imageStore(t_output, ivec3(n4_a, 0, 0), out_a);
  if (n4_b < N4) {
    imageStore(t_output, ivec3(n4_b, 0, 0), out_b);
  }
#endif
}
