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

${layout_declare_tensor(B, "w", "tout", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "tin", DTYPE, STORAGE)}

layout(push_constant) uniform restrict Block {
  ivec4 tin_sizes;
  ivec3 tout_limits;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = 0;
layout(constant_id = 4) const int reduce_dim = 0;
layout(constant_id = 5) const int group_dim = 1;

// A more verbose name would be NWORKERS_PER_GROUP. This describes the number of
// threads that will co-operate to compute one reduction output. There may be
// multiple groups computing distinct reduction outputs within one work group.
#define NWORKERS 4

// Sets an upper limit on the total size of a work group based on how many
// elements are allocated in the shared memory array below. Each thread in the
// work group will write into its assigned element in the shared array.
#define MAX_NTHREADS 16

shared vec4 shared_max[MAX_NTHREADS];
shared vec4 shared_sum[MAX_NTHREADS];

#include "indexing_utils.h"

int tid_to_smi(const ivec2 tid) {
  return tid.x + tid.y * NWORKERS;
}

/*
 * The shaders below compute softmax for a tensor. Softmax is an interesting mix
 * between a reduction operator and a unary elementwise operator, defined as
 * exp(x) / (sum of exp(x)). The general flow of the computation is:
 *
 * First, find the maximum element along the reduction dim. The maximum element
 * is used to preserve numerical stability, since division of exponents is
 * translation invariant.
 *
 * Next, compute the sum of exp(x - max_element) along the reduction dim.
 *
 * Finally, for each element along the reduction dim, we compute the output as
 * exp(x - max_element) / sum_of_exponents.
 *
 * The shaders below also utilize shared memory to have multiple threads help
 * compute the max and sum reduction operations. A total of NGROUPS x NWORKERS
 * threads are launched. Each group works on a unique reduction "row", and
 * within a group NWORKERS threads co-operate to compute the max and sum of one
 * "row". Each worker in the group is responsible for computing a partial output
 * of the "row" and uploading it to shared memory; the overall reduction output
 * can then be determined by aggregating the partial outputs stored in shared
 * memory.
 *
 * As a caveat, this shader does not currently support cases where `batch` > 1
 * and the reduce dim happens to also be the batch concatenation dim.  To support
 * this, there will need to be additional logic to set the starting value of
 * `scan_pos[reduce_dim]`. Since this is not expected to be a common use-case,
 * supporting this case is left as an exercise for when it is required.
 *
 * As a final note, log softmax is supported with this shader as well since via
 * the op1 and op2 macro definitions. See the corresponding YAML file for more
 * details.
 */

/*
 * Computes softmax where the reduction dim is orthogonal to the packed dim.
 * This case is simpler because each element of a texel belongs to a separate
 * reduction dim, meaning we don't have to perform reduction along a texel.
 */
void softmax_nonpacked_dim(const ivec2 tid, ivec3 scan_pos) {
  // shared memory index of this thread
  const int smi = tid_to_smi(tid);
  // used to iterate over all shared memory in the group
  int group_i;

  scan_pos[reduce_dim] = tid.x;
  vec4 max_elements = load_texel(tin, scan_pos);
  // This thread computes a partial maximum
  for (int i = tid.x; i < tin_sizes[reduce_dim];
       i += NWORKERS, scan_pos[reduce_dim] += NWORKERS) {
    max_elements = max(max_elements, load_texel(tin, scan_pos));
  }
  shared_max[smi] = max_elements;
  barrier();
  // Iterate over the partial maximums to obtain the overall maximum
  group_i = tid.y * NWORKERS;
  max_elements = shared_max[group_i++];
  for (int i = 1; i < NWORKERS; ++i, group_i++) {
    max_elements = max(max_elements, shared_max[group_i]);
  }

  scan_pos[reduce_dim] = tid.x;
  vec4 denominators = vec4(0);
  // Compute partial sum
  for (int i = tid.x; i < tin_sizes[reduce_dim];
       i += NWORKERS, scan_pos[reduce_dim] += NWORKERS) {
    denominators += exp(load_texel(tin, scan_pos) - max_elements);
  }
  shared_sum[smi] = denominators;
  barrier();
  // Iterate over the partial sums to obtain the overall sum
  group_i = tid.y * NWORKERS;
  denominators = shared_sum[group_i++];
  for (int i = 1; i < NWORKERS; ++i, group_i++) {
    denominators += shared_sum[group_i];
  }

  // Determine if there are any padding elements in the final texel of the
  // packed dimension
  const int nspill = mod4(tin_sizes[packed_dim]);
  // Detect if this thread is working on the final texels of the packed
  // dimension, which may have padding elements
  const bool is_last_texel =
      scan_pos[packed_dim] == (tout_limits[packed_dim] - 1);

  scan_pos[reduce_dim] = tid.x;
  for (int i = tid.x; i < tin_sizes[reduce_dim];
       i += NWORKERS, scan_pos[reduce_dim] += NWORKERS) {
    const vec4 numerators = op1(load_texel(tin, scan_pos) - max_elements);
    vec4 outtex = op2(numerators, denominators);
    // For the last texel in the packed dim, make sure that the padding elements
    // are explicitly set to 0. Otherwise, they may influence computations later
    // down the line.
    if (is_last_texel && nspill > 0) {
      [[unroll]] for (int i = nspill; i < 4; ++i) {
        outtex[i] = 0;
      }
    }
    write_texel(tout, scan_pos, outtex);
  }
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
  // shared memory index of this thread
  const int smi = tid_to_smi(tid);
  // used to iterate over all shared memory in the group
  int group_i;

  const int nspill = mod4(tin_sizes[packed_dim]);
  const int reduce_len = tin_sizes[packed_dim] - nspill;

  scan_pos[reduce_dim] = tid.x;
  vec4 max_elements = vec4(load_texel(tin, scan_pos).x);
  for (int i = tid.x * 4; i < reduce_len;
       i += NWORKERS * 4, scan_pos[reduce_dim] += NWORKERS) {
    max_elements = max(max_elements, load_texel(tin, scan_pos));
  }
  // For the last texel in the dim, if there are padding elements then each
  // element of the texel needs to be processed individually such that the
  // padding elements are ignored
  if (scan_pos[reduce_dim] == tout_limits[reduce_dim] - 1 && nspill > 0) {
    const vec4 intex = load_texel(tin, scan_pos);
    for (int i = 0; i < nspill; ++i) {
      max_elements.x = max(intex[i], max_elements.x);
    }
  }
  shared_max[smi] = max_elements;
  barrier();
  // Iterate over the partial maximums to obtain the overall maximum
  group_i = tid.y * NWORKERS;
  max_elements = shared_max[group_i++];
  for (int i = 1; i < NWORKERS; ++i, group_i++) {
    max_elements = max(max_elements, shared_max[group_i]);
  }
  // Each element of the texel is itself a partial maximum; iterate over the
  // texel to find the actual maximum
  float max_element = max_elements.x;
  [[unroll]] for (int i = 1; i < 4; ++i) {
    max_element = max(max_elements[i], max_element);
  }

  scan_pos[reduce_dim] = tid.x;
  vec4 denominators = vec4(0);
  for (int i = tid.x * 4; i < reduce_len;
       i += NWORKERS * 4, scan_pos[reduce_dim] += NWORKERS) {
    denominators += exp(load_texel(tin, scan_pos) - max_element);
  }
  // For the last texel in the dim, if there are padding elements then each
  // element of the texel needs to be processed individually such that the
  // padding elements are ignored
  if (nspill > 0 && scan_pos[reduce_dim] == tout_limits[reduce_dim] - 1) {
    const vec4 intex = load_texel(tin, scan_pos);
    for (int i = 0; i < nspill; ++i) {
      denominators.x += exp(intex[i] - max_element);
    }
  }
  shared_sum[smi] = denominators;
  barrier();
  // Iterate over the partial sums to obtain the overall sum
  group_i = tid.y * NWORKERS;
  denominators = shared_sum[group_i++];
  for (int i = 1; i < NWORKERS; ++i, group_i++) {
    denominators += shared_sum[group_i];
  }
  // Reduce over the accumulated texel to find the overall sum
  float denominator = 0;
  [[unroll]] for (int i = 0; i < 4; ++i) {
    denominator += denominators[i];
  }

  scan_pos[reduce_dim] = tid.x;
  for (int i = tid.x * 4; i < reduce_len;
       i += NWORKERS * 4, scan_pos[reduce_dim] += NWORKERS) {
    const vec4 numerators = op1(load_texel(tin, scan_pos) - max_element);
    write_texel(tout, scan_pos, op2(numerators, denominator));
  }
  // For the last texel in the dim, if there are padding elements then the
  // padding elements need to be set to 0 explicitly, otherwise they may
  // influence subsequent operations.
  if (nspill > 0 && scan_pos[reduce_dim] == tout_limits[reduce_dim] - 1) {
    const vec4 numerator = op1(load_texel(tin, scan_pos) - max_element);
    vec4 outtex = op2(numerator, denominator);
    [[unroll]] for (int i = nspill; i < 4; ++i) {
      outtex[i] = 0;
    }
    write_texel(tout, scan_pos, outtex);
  }
}

void main() {
  ivec3 scan_pos = ivec3(gl_GlobalInvocationID);
  scan_pos[reduce_dim] = 0;

  const ivec2 tid = ivec2(
      gl_LocalInvocationID[reduce_dim],
      gl_LocalInvocationID[group_dim]);

  if (any(greaterThanEqual(scan_pos, tout_limits))) {
    return;
  }

  if (reduce_dim != packed_dim) {
    softmax_nonpacked_dim(tid, scan_pos);
  } else {
    softmax_packed_dim(tid, scan_pos);
  }
}
