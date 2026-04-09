/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("texture3d", DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}
#define T ${texel_load_component_type(DTYPE, "texture3d")}

${define_active_storage_type("texture3d")}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}

${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "texture3d")}

${layout_declare_ubo(B, "TextureMetadata", "out_meta")}
${layout_declare_ubo(B, "TextureMetadata", "in_meta")}

layout(push_constant) uniform PRECISION restrict Block {
  float epsilon;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "in_layout", "CONTIG_LAYOUT_INT")}

// This shader assumes width-packed layout.
// Dispatch: global = {1, num_rows, 1}, local = {NUM_WORKERS, 1, 1}
// Each workgroup processes one row; workers cooperatively reduce across texels.

#define NUM_WORKERS 64

shared float shared_sum[NUM_WORKERS];

void reduce_shared(const uint worker_id) {
  memoryBarrierShared();
  barrier();

  [[unroll]] for (int stride = NUM_WORKERS / 2; stride > 0; stride >>= 1) {
    if (worker_id < stride) {
      shared_sum[worker_id] += shared_sum[worker_id + stride];
    }
    memoryBarrierShared();
    barrier();
  }
}

void main() {
  const uint row_idx = gl_GlobalInvocationID.y;
  const uint worker_id = gl_LocalInvocationID.x;

  const int width = in_meta.sizes.x;
  const int num_texels = div_up_4(width);
  const int remain = width & 3;

  // Decompose row_idx into (y, z) texture coordinates. When the tensor has more
  // than one Z slice (e.g. 4D tensors), row_idx encodes both Y and Z.
  const int tex_H = in_meta.limits.y;
  const int y = int(row_idx) % tex_H;
  const int z = int(row_idx) / tex_H;

  // First pass: compute mean(x^2) via cooperative reduction in fp32
  float local_sq_sum = 0.0;
  for (int tx = int(worker_id); tx < num_texels; tx += NUM_WORKERS) {
    ivec3 pos = ivec3(tx, y, z);

    VEC4_T in_val = texelFetch(t_in, pos, 0);

    if (tx == num_texels - 1 && remain != 0) {
      const int remain_inv = 4 - remain;
      in_val.y = mix(in_val.y, T(0), remain_inv > 2);
      in_val.z = mix(in_val.z, T(0), remain_inv > 1);
      in_val.w = mix(in_val.w, T(0), remain_inv > 0);
    }
    vec4 v = vec4(in_val);
    local_sq_sum += dot(v, v);
  }

  shared_sum[worker_id] = local_sq_sum;
  reduce_shared(worker_id);

  const float mean_sq = shared_sum[0] / float(width);
  const float rstd = inversesqrt(mean_sq + epsilon);

  memoryBarrierShared();
  barrier();

  // Second pass: normalize and write output
  for (int tx = int(worker_id); tx < num_texels; tx += NUM_WORKERS) {
    ivec3 pos = ivec3(tx, y, z);

    const VEC4_T in_val = texelFetch(t_in, pos, 0);
    const VEC4_T weight = texelFetch(t_weight, ivec3(tx, 0, 0), 0);
    const VEC4_T outtex =
        VEC4_T(vec4(in_val) * rstd * vec4(weight));
    imageStore(t_out, pos, outtex);
  }
}
