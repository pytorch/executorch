/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}

#define PRECISION ${PRECISION}
#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_loss", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_loss_partial", DTYPE, "buffer")}

${layout_declare_ubo(B, "int", "n_rows")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define NWORKERS 64

shared float red[NWORKERS];

// Self-contained [N] -> [1] tree-sum of the per-row losses in one workgroup, so
// fused_ce carries no cross-op reduce dependency.
void main() {
  const uint tid = gl_LocalInvocationID.x;

  float s = 0.0;
  for (uint j = tid; j < uint(n_rows); j += NWORKERS) {
    s += float(t_loss_partial[j]);
  }
  red[tid] = s;
  memoryBarrierShared();
  barrier();

  for (uint k = NWORKERS / 2u; k > 0u; k >>= 1u) {
    if (tid < k) {
      red[tid] += red[tid + k];
    }
    memoryBarrierShared();
    barrier();
  }

  if (tid == 0u) {
    t_loss[0] = T(red[0]);
  }
}
