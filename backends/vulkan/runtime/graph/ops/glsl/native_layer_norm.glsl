/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#include "broadcasting_utils.h"
#include "indexing_utils.h"

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "w", "t_mean", DTYPE, STORAGE)}
${layout_declare_tensor(B, "w", "t_rstd", DTYPE, STORAGE)}

${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, STORAGE)}

${layout_declare_ubo(B, "ivec3", "out_limits")}
${layout_declare_ubo(B, "ivec4", "sizes")}
${layout_declare_ubo(B, "float", "epsilon")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);
const lowp int in_packed_dim = unhash_packed_dim(in_layout);

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int out_packed_dim = unhash_packed_dim(out_layout);

void main() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(lpos, out_limits))) {
    return;
  }

  const int width = int(sizes.x);

  VEC4_T mean = VEC4_T(0);
  VEC4_T delta = VEC4_T(0);
  VEC4_T delta2 = VEC4_T(0);
  VEC4_T M2 = VEC4_T(0);

  // Use Welford's online algorithm to compute mean and variance in one pass
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  ivec3 in_pos = lpos_to_pos(lpos, in_axis_map);
  for (int w = 0; w < width; ++w) {
    in_pos[in_axis_map.x] = w;
    VEC4_T v = load_texel(t_in, in_pos);
    delta = v - mean;
    mean += delta / (w + 1);
    delta2 = v - mean;
    M2 += delta * delta2;
  }

  VEC4_T var = M2 / width;
  VEC4_T rstd = pow(var + epsilon, VEC4_T(-0.5));
  VEC4_T offset = -rstd * mean;

  for (int w = 0; w < width; ++w) {
    in_pos[in_axis_map.x] = w;
    VEC4_T v = load_texel(t_in, in_pos);
    // broadcasting
    VEC4_T weight = load_texel(t_weight, ivec3(w, 0, 0)).xxxx;
    VEC4_T bias = load_texel(t_bias, ivec3(w, 0, 0)).xxxx;
    VEC4_T outtex = (v * rstd + offset) * weight + bias;
    write_texel_lpos(t_out, ivec3(w, lpos.y, lpos.z), outtex, out_axis_map);
  }

  write_texel(t_mean, lpos, mean);
  write_texel(t_rstd, lpos, rstd);
}
