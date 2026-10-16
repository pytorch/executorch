/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define NUM_WORKERS_PER_WG 64

$if MODE == "llm":
  #define HAS_INPUT_POS

#define IN_DTYPE ${IN_DTYPE}
#define OUT_DTYPE ${OUT_DTYPE}
#define SOFTMAX_IN_VEC4_T ${texel_load_type(IN_DTYPE, STORAGE)}
#define SOFTMAX_ACC_T ${texel_load_component_type(IN_DTYPE, STORAGE)}
#define VEC4_T ${texel_load_type(OUT_DTYPE, STORAGE)}
#define T ${texel_load_component_type(OUT_DTYPE, STORAGE)}

${define_active_storage_type(STORAGE)}

${define_required_extensions(STORAGE, [IN_DTYPE, OUT_DTYPE])}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_attn_weights_softmax", OUT_DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_attn_weights", IN_DTYPE, STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_sizes")}
${layout_declare_ubo(B, "ivec4", "k_sizes")}
$if MODE == "llm":
  ${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

SOFTMAX_IN_VEC4_T load_attn_weights_c4(
    const int c4,
    const int s,
    const int q_h,
    const int C4,
    const int S,
    const int Q_H) {
#ifdef USING_BUFFER
  return t_attn_weights[(q_h * S * C4) + (s * C4) + c4];
#else
  return SOFTMAX_IN_VEC4_T(texelFetch(t_attn_weights, ivec3(c4, s, q_h), 0));
#endif
}

void store_attn_weights_softmax_c4(
    const VEC4_T out_texel,
    const int c4,
    const int s,
    const int q_h,
    const int C4,
    const int S,
    const int Q_H) {
#ifdef USING_BUFFER
  t_attn_weights_softmax[(q_h * S * C4) + (s * C4) + c4] = out_texel;
#else
  imageStore(t_attn_weights_softmax, ivec3(c4, s, q_h), out_texel);
#endif
}

#include "sdpa_attn_weights_softmax_row.glslh"

/*
 * 3-pass numerically stable softmax over the context_len dimension of
 * attention weights.
 *
 * LLM SDPA (HAS_INPUT_POS):
 *   reads VEC4_T (input dtype), reduces in T, writes VEC4_T.
 *   attn_weights S dim is padded to S_aligned.
 *   current context_len = input_pos + S.
 *
 * Fused SDPA (!HAS_INPUT_POS):
 *   reads vec4 (fp32 from QK), reduces in fp32, writes VEC4_T (input dtype).
 *   attn_weights S dim is not padded.
 *   context_len = k_sizes.y.
 *
 * Dispatch: (1, S, H * B) — for LLM (batch=1), H * B == H_q.
 */
void main() {
  const int worker_id = int(gl_LocalInvocationID.x);

  // Index along attention weight's sequence_len dim
  const int s = int(gl_GlobalInvocationID.y);
  // For LLM: q_head index. For fused: combined batch*H + head index.
  const int q_h = int(gl_GlobalInvocationID.z);

#ifdef HAS_INPUT_POS
  // LLM: q_sizes is WHCN {D, H_q, S, B}
  const int Q_H = q_sizes.y;
  const int S = q_sizes.z;
#else
  // Fused: q_sizes is WHCN {D, S, H, B}
  const int Q_H = q_sizes.z;
  const int S = q_sizes.y;
#endif
  const int S_aligned = align_up_4(S);

#ifdef HAS_INPUT_POS
  // manually determine size of the context_len dim of the attention weight.
  // The "actual" tensor sizes may have been aligned to a multiple of 4 to allow
  // memory loads to be aligned to texel boundaries.
  const int context_len = input_pos + S;
#else
  const int context_len = k_sizes.y;
#endif

  // LLM: attn_weights S dim is padded to S_aligned; fused: not padded.
#ifdef HAS_INPUT_POS
  const int attn_S = S_aligned;
#else
  const int attn_S = S;
#endif

  // bounds check — q_h bound is Q_H * batch_size; for LLM (batch=1) this
  // equals Q_H, for fused this equals H * B.
  if (s >= S || q_h >= Q_H * q_sizes.w) {
    return;
  }

  softmax_attn_weights_row(worker_id, s, q_h, context_len, attn_S, Q_H);
}
