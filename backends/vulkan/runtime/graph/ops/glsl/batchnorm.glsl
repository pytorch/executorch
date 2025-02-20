/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_type(DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "weight_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "bias_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "mean_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "var_in", DTYPE, STORAGE)}

${layout_declare_ubo(B, "ivec3", "out_limits")}
${layout_declare_ubo(B, "float", "eps")}
${layout_declare_ubo(B, "int", "num_texel_per_batch")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  VEC4_T v = VEC4_T(load_texel(t_in, pos));

  ivec3 param_pos = ivec3(pos.z % num_texel_per_batch, 0, 0);

  VEC4_T weight = VEC4_T(load_texel(weight_in, param_pos));
  VEC4_T bias = VEC4_T(load_texel(bias_in, param_pos));
  VEC4_T mean = VEC4_T(load_texel(mean_in, param_pos));
  VEC4_T var = VEC4_T(load_texel(var_in, param_pos));

  v = ((v - mean) / sqrt(var + eps)) * weight + bias;

  write_texel(t_out, pos, v);
}
