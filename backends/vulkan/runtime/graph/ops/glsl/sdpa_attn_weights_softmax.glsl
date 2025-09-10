/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define T ${texel_load_component_type(DTYPE, STORAGE)}

#define NUM_WORKERS_PER_WG 64

${define_active_storage_type(STORAGE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_attn_weights_softmax", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_attn_weights", DTYPE, STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_projected_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Shared memory for cooperative exp sum finding
shared T shared_exp_sum[NUM_WORKERS_PER_WG];

VEC4_T load_attn_weights_c4(
    const int c4,
    const int s,
    const int q_h,
    const int C4,
    const int S,
    const int Q_H) {
#ifdef USING_BUFFER
  return t_attn_weights[(q_h * S * C4) + (s * C4) + c4];
#else
  return texelFetch(t_attn_weights, ivec3(c4, s, q_h), 0);
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

void main() {
  const int worker_id = int(gl_LocalInvocationID.x);

  // Index along attention weight's sequence_len dim
  const int s = int(gl_GlobalInvocationID.y);
  // idx along attention weight's num_q_heads dim
  const int q_h = int(gl_GlobalInvocationID.z);

  // number of Q heads
  const int Q_H = q_projected_sizes.y;
  // sequence length
  const int S = q_projected_sizes.z;
  // manually determine size of the context_len dim of the attention weight.
  // The "actual" tensor sizes may have been aligned to a multiple of 4 to allow
  // memory loads to be aligned to texel boundaries.
  const int context_len = input_pos + S;
  const int context_texel_len = div_up_4(context_len);

  if (s >= S || q_h >= Q_H) {
    return;
  }

  // Initialize thread-local min/max
  T local_exp_sum = 0;

  const int context_len_aligned_down = context_len - mod_4(context_len);
  const int C4_limit = div_4(context_len_aligned_down);

  // Each thread processes elements along a context_len row with a stride of the
  // number of threads in the work group.
  for (int c4 = worker_id; c4 < C4_limit; c4 += NUM_WORKERS_PER_WG) {
    VEC4_T in_texel = load_attn_weights_c4(
        c4, s, q_h, context_texel_len, S, Q_H);

    for (int comp = 0; comp < 4; comp++) {
      local_exp_sum += exp(in_texel[comp]);
    }
  }
  // First thread in the work group responsible for handling last texel if it
  // contains any padded elements
  if (worker_id == 0) {
    for (int c4 = C4_limit; c4 < context_texel_len; ++c4) {
      const int c_base = mul_4(c4);
      VEC4_T in_texel = load_attn_weights_c4(
          c4, s, q_h, context_texel_len, S, Q_H);

      [[unroll]] for (int comp = 0; comp < 4; comp++) {
        if (c_base + comp < context_len) {
          local_exp_sum += exp(in_texel[comp]);
        }
      }
    }
  }

  // Store thread-local results in shared memory
  shared_exp_sum[worker_id] = local_exp_sum;

  memoryBarrierShared();
  barrier();

  // Tree reduction to compute the overall result
  for (int i = NUM_WORKERS_PER_WG / 2; i > 0; i >>= 1) {
    if (worker_id < i) {
      shared_exp_sum[worker_id] = shared_exp_sum[worker_id] +
          shared_exp_sum[worker_id + i];
    }
    memoryBarrierShared();
    barrier();
  }

  local_exp_sum = shared_exp_sum[0];
  // Now go back through each element in the row and normalize
  for (int c4 = worker_id; c4 < C4_limit; c4 += NUM_WORKERS_PER_WG) {
    VEC4_T in_texel = load_attn_weights_c4(
        c4, s, q_h, context_texel_len, S, Q_H);

    VEC4_T out_texel = exp(in_texel) / local_exp_sum;
    store_attn_weights_softmax_c4(
        out_texel, c4, s, q_h, context_texel_len, S, Q_H);
  }
  // First thread in the work group responsible for handling last texel if it
  // contains any padded elements
  if (worker_id == 0) {
    for (int c4 = C4_limit; c4 < context_texel_len; ++c4) {
      const int c_base = mul_4(c4);
      VEC4_T in_texel = load_attn_weights_c4(
          c4, s, q_h, context_texel_len, S, Q_H);

      // Ensure that padding elements are set to 0.
      VEC4_T out_texel = VEC4_T(0);
      [[unroll]] for (int comp = 0; comp < 4; comp++) {
        if (c_base + comp < context_len) {
          out_texel[comp] = exp(in_texel[comp]) / local_exp_sum;
        }
      }
      store_attn_weights_softmax_c4(
          out_texel, c4, s, q_h, context_texel_len, S, Q_H);
    }
  }
}
