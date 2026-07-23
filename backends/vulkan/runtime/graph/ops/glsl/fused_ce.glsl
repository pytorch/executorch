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

${layout_declare_tensor(B, "w", "t_dlogits", DTYPE, "buffer")}
${layout_declare_tensor(B, "w", "t_loss_partial", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_logits", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_labels", "int", "buffer")}

${layout_declare_ubo(B, "int", "vocab")}
${layout_declare_ubo(B, "int", "n_rows")}
${layout_declare_ubo(B, "float", "n_valid")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define NWORKERS 64

shared float red_m[NWORKERS];
shared float red_l[NWORKERS];

// Fused cross-entropy: one workgroup per row cooperatively reduces the vocab
// dimension with a single-pass online softmax (running max + rescaled running
// sum), then writes per-row loss and dlogits. Rows with label < 0 are masked.
void main() {
  const uint row = gl_GlobalInvocationID.y;
  if (int(row) >= n_rows) {
    return;
  }

  const uint tid = gl_LocalInvocationID.x;
  const uint V = uint(vocab);
  const uint base = row * V;
  const int lbl = t_labels[row];

  if (lbl < 0) {
    for (uint j = tid; j < V; j += NWORKERS) {
      t_dlogits[base + j] = T(0);
    }
    if (tid == 0u) {
      t_loss_partial[row] = T(0);
    }
    return;
  }

  // Single read pass: maintain a running max m and the sum l of exp(x - m),
  // rescaling l whenever a larger value updates m. Finite -3.4e38 init.
  float m = -3.4e38;
  float l = 0.0;
  for (uint j = tid; j < V; j += NWORKERS) {
    float x = float(t_logits[base + j]);
    if (x > m) {
      l = l * exp(m - x) + 1.0;
      m = x;
    } else {
      l = l + exp(x - m);
    }
  }
  red_m[tid] = m;
  red_l[tid] = l;
  memoryBarrierShared();
  barrier();

  // Tree-combine the (m, l) pairs: m becomes the max, l is rescaled to it.
  for (uint s = NWORKERS / 2u; s > 0u; s >>= 1u) {
    if (tid < s) {
      float ma = red_m[tid];
      float la = red_l[tid];
      float mb = red_m[tid + s];
      float lb = red_l[tid + s];
      float mm = max(ma, mb);
      red_m[tid] = mm;
      red_l[tid] = la * exp(ma - mm) + lb * exp(mb - mm);
    }
    memoryBarrierShared();
    barrier();
  }

  const float row_max = red_m[0];
  const float denom = red_l[0];
  const float inv = 1.0 / denom;
  const float scale = 1.0 / n_valid;

  if (tid == 0u) {
    const float lse = row_max + log(denom);
    t_loss_partial[row] = T((lse - float(t_logits[base + uint(lbl)])) * scale);
  }

  for (uint j = tid; j < V; j += NWORKERS) {
    float g = exp(float(t_logits[base + j]) - row_max) * inv * scale;
    if (j == uint(lbl)) {
      g = g - scale;
    }
    t_dlogits[base + j] = T(g);
  }
}
