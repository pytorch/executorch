/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define op1(X) ${OPERATOR1}

#define op2(X, Y) ${OPERATOR2}

${define_active_storage_type(STORAGE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "tout", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "tin", DTYPE, STORAGE)}

${layout_declare_ubo(B, "TextureMetadata", "in_meta")}
${layout_declare_ubo(B, "TextureMetadata", "out_meta")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
const int packed_dim = get_packed_dim(out_layout);

${layout_declare_spec_const(C, "int", "reduce_dim", "0")}
${layout_declare_spec_const(C, "int", "group_dim", "1")}

#define NWORKERS 4
#define MAX_NTHREADS 16

shared vec4 shared_max[MAX_NTHREADS];
shared vec4 shared_sum[MAX_NTHREADS];

int tid_to_smi(const ivec2 tid) {
  return tid.x + tid.y * NWORKERS;
}

/*
 * Computes softmax where the reduction dim is orthogonal to the packed dim.
 * This case is simpler because each element of a texel belongs to a separate
 * reduction dim, meaning we don't have to perform reduction along a texel.
 */
void softmax_nonpacked_dim(const ivec2 tid, ivec3 scan_pos) {
  const int smi = tid_to_smi(tid);
  int group_i;

  scan_pos[reduce_dim] = tid.x;
  vec4 max_elements = texelFetch(tin, scan_pos, 0);
  for (int i = tid.x; i < safe_idx(in_meta.sizes, reduce_dim);
       i += NWORKERS, scan_pos[reduce_dim] += NWORKERS) {
    max_elements = max(max_elements, texelFetch(tin, scan_pos, 0));
  }
  shared_max[smi] = max_elements;
  barrier();
  group_i = tid.y * NWORKERS;
  max_elements = shared_max[group_i++];
  for (int i = 1; i < NWORKERS; ++i, group_i++) {
    max_elements = max(max_elements, shared_max[group_i]);
  }

  scan_pos[reduce_dim] = tid.x;
  vec4 denominators = vec4(0);
  for (int i = tid.x; i < safe_idx(in_meta.sizes, reduce_dim);
       i += NWORKERS, scan_pos[reduce_dim] += NWORKERS) {
    denominators += exp(texelFetch(tin, scan_pos, 0) - max_elements);
  }
  shared_sum[smi] = denominators;
  barrier();
  group_i = tid.y * NWORKERS;
  denominators = shared_sum[group_i++];
  for (int i = 1; i < NWORKERS; ++i, group_i++) {
    denominators += shared_sum[group_i];
  }

  const int nspill = mod_4(safe_idx(in_meta.sizes, packed_dim));
  const bool is_last_texel =
      scan_pos[packed_dim] == (safe_idx(out_meta.limits, packed_dim) - 1);

  scan_pos[reduce_dim] = tid.x;
  for (int i = tid.x; i < safe_idx(in_meta.sizes, reduce_dim);
       i += NWORKERS, scan_pos[reduce_dim] += NWORKERS) {
    const vec4 numerators = op1(texelFetch(tin, scan_pos, 0) - max_elements);
    const vec4 safe_denom = max(denominators, vec4(1e-37));
    vec4 outtex = op2(numerators, safe_denom);
    {
      uvec4 bits = floatBitsToUint(outtex);
      uvec4 nan_inf_mask = uvec4(
          ((bits.x & 0x7F800000u) == 0x7F800000u) ? 0xFFFFFFFFu : 0u,
          ((bits.y & 0x7F800000u) == 0x7F800000u) ? 0xFFFFFFFFu : 0u,
          ((bits.z & 0x7F800000u) == 0x7F800000u) ? 0xFFFFFFFFu : 0u,
          ((bits.w & 0x7F800000u) == 0x7F800000u) ? 0xFFFFFFFFu : 0u);
      outtex = uintBitsToFloat(bits & ~nan_inf_mask);
    }
    if (is_last_texel && nspill > 0) {
      [[unroll]] for (int i = nspill; i < 4; ++i) {
        outtex[i] = 0;
      }
    }
    imageStore(tout, scan_pos, outtex);
  }
  memoryBarrierImage();
}

/*
 * Compute softmax where the reduction dim is also the packed dim. This case is
 * complex because the reduction needs to occur over the individual texels.
 * Therefore, in this algorithm each element of the accumulator texels are
 * themselves partial outputs. Special care has to be taken to ignore padding
 * elements in texels (which occur when the size of the packed dim is not a
 * multiple of 4) so that they do not influence the output of reduction.
 */
void softmax_packed_dim(const ivec2 tid, ivec3 scan_pos) {
  const int smi = tid_to_smi(tid);
  int group_i;

  const int nspill = mod_4(safe_idx(in_meta.sizes, packed_dim));
  const int reduce_len = safe_idx(in_meta.sizes, packed_dim) - nspill;

  scan_pos[reduce_dim] = tid.x;
  vec4 max_elements = vec4(-3.402823e+38);
  for (int i = tid.x * 4; i < reduce_len;
       i += NWORKERS * 4, scan_pos[reduce_dim] += NWORKERS) {
    max_elements = max(max_elements, texelFetch(tin, scan_pos, 0));
  }
  if (scan_pos[reduce_dim] == safe_idx(out_meta.limits, reduce_dim) - 1 && nspill > 0) {
    const vec4 intex = texelFetch(tin, scan_pos, 0);
    for (int i = 0; i < nspill; ++i) {
      max_elements.x = max(intex[i], max_elements.x);
    }
  }
  shared_max[smi] = max_elements;
  barrier();
  group_i = tid.y * NWORKERS;
  max_elements = shared_max[group_i++];
  for (int i = 1; i < NWORKERS; ++i, group_i++) {
    max_elements = max(max_elements, shared_max[group_i]);
  }
  float max_element = max_elements.x;
  [[unroll]] for (int i = 1; i < 4; ++i) {
    max_element = max(max_elements[i], max_element);
  }

  scan_pos[reduce_dim] = tid.x;
  vec4 denominators = vec4(0);
  for (int i = tid.x * 4; i < reduce_len;
       i += NWORKERS * 4, scan_pos[reduce_dim] += NWORKERS) {
    denominators += exp(texelFetch(tin, scan_pos, 0) - max_element);
  }
  if (nspill > 0 && scan_pos[reduce_dim] == safe_idx(out_meta.limits, reduce_dim) - 1) {
    const vec4 intex = texelFetch(tin, scan_pos, 0);
    for (int i = 0; i < nspill; ++i) {
      denominators.x += exp(intex[i] - max_element);
    }
  }
  shared_sum[smi] = denominators;
  barrier();
  group_i = tid.y * NWORKERS;
  denominators = shared_sum[group_i++];
  for (int i = 1; i < NWORKERS; ++i, group_i++) {
    denominators += shared_sum[group_i];
  }
  float denominator = 0;
  [[unroll]] for (int i = 0; i < 4; ++i) {
    denominator += denominators[i];
  }
  const float safe_denominator = max(denominator, 1e-37);

  scan_pos[reduce_dim] = tid.x;
  for (int i = tid.x * 4; i < reduce_len;
       i += NWORKERS * 4, scan_pos[reduce_dim] += NWORKERS) {
    const vec4 numerators = op1(texelFetch(tin, scan_pos, 0) - max_element);
    imageStore(tout, scan_pos, op2(numerators, safe_denominator));
  }
  if (nspill > 0 && scan_pos[reduce_dim] == safe_idx(out_meta.limits, reduce_dim) - 1) {
    const vec4 numerator = op1(texelFetch(tin, scan_pos, 0) - max_element);
    vec4 outtex = op2(numerator, safe_denominator);
    [[unroll]] for (int i = nspill; i < 4; ++i) {
      outtex[i] = 0;
    }
    imageStore(tout, scan_pos, outtex);
  }
}

void main() {
  ivec3 scan_pos = ivec3(gl_GlobalInvocationID);
  scan_pos[reduce_dim] = 0;

  const ivec2 tid = ivec2(
      gl_LocalInvocationID[reduce_dim],
      gl_LocalInvocationID[group_dim]);

  if (any(greaterThanEqual(scan_pos, out_meta.limits))) {
    return;
  }

  if (reduce_dim != packed_dim) {
    softmax_nonpacked_dim(tid, scan_pos);
  } else {
    softmax_packed_dim(tid, scan_pos);
  }
}
