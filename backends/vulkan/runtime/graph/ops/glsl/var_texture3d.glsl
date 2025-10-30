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

${define_active_storage_type(STORAGE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "tout", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "tin", DTYPE, STORAGE)}

${layout_declare_ubo(B, "ivec3", "tin_limits")}
${layout_declare_ubo(B, "ivec4", "tin_sizes")}

layout(push_constant) uniform PushConstants {
  int unbiased;
} pc;

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

shared VEC4_T shared_sum[MAX_NTHREADS];
shared VEC4_T shared_sum_sq[MAX_NTHREADS];
shared int shared_count[MAX_NTHREADS];

#include "indexing_utils.h"

int tid_to_smi(const ivec2 tid) {
  return tid.x + tid.y * NWORKERS;
}

VEC4_T calculate_variance(VEC4_T sum, VEC4_T sum_sq, int count) {
  VEC4_T mean = sum / float(count);
  VEC4_T variance = (sum_sq / float(count)) - (mean * mean);

  if ((pc.unbiased != 0) && (count > 1)) {
    variance = variance * (float(count) / float(count - 1.0));
  }

  return variance;
}

void reduce_nonpacked_dim(const ivec2 tid, ivec3 scan_pos) {
  // shared memory index of this thread
  const int smi = tid_to_smi(tid);

  VEC4_T sum = VEC4_T(0);
  VEC4_T sum_sq = VEC4_T(0);
  int count = 0;

  scan_pos[reduce_dim] = tid.x;
  for (int i = tid.x; i < tin_sizes[reduce_dim];
       i += NWORKERS, scan_pos[reduce_dim] += NWORKERS) {
    VEC4_T val = load_texel(tin, scan_pos);
    sum += val;
    sum_sq += val * val;
    count += 1;
  }
  // Write partial output to shared memory and synchronize work group
  shared_sum[smi] = sum;
  shared_sum_sq[smi] = sum_sq;
  shared_count[smi] = count;
  barrier();

  // Since the reduction row is reduced to only one element, only the "main"
  // thread in the group needs aggregate the partial outputs
  if (tid.x == 0) {
    int group_i = tid.y * NWORKERS;
    sum = shared_sum[group_i];
    sum_sq = shared_sum_sq[group_i];
    count = shared_count[group_i];

    for (int i = 1; i < NWORKERS; i++) {
      int idx = tid.y * NWORKERS + i;
      sum += shared_sum[idx];
      sum_sq += shared_sum_sq[idx];
      count += shared_count[idx];
    }

    // Determine if there are any padding elements in the final texel of the
    // packed dimension
    const int nspill = mod4(tin_sizes[packed_dim]);
    // Detect if this thread is working on the final texels of the packed
    // dimension, which may have padding elements
    const bool is_last_texel =
        scan_pos[packed_dim] == (tin_limits[packed_dim] - 1);

    VEC4_T variance = calculate_variance(sum, sum_sq, count);

    // Explicitly set padding elements to 0
    if (is_last_texel && nspill > 0) {
      [[unroll]] for (int i = nspill; i < 4; i++) {
        variance[i] = 0;
      }
    }

    scan_pos[reduce_dim] = tid.x;
    write_texel(tout, scan_pos, variance);
  }
}

/*
 * Compute reduction where the reduction dim is also the packed dim. This case is
 * complex because the reduction needs to occur over the individual texels.
 * Therefore, in this algorithm each element of the accumulator texels are
 * themselves partial outputs. Special care has to be taken to ignore padding
 * elements in texels (which occur when the size of the packed dim is not a
 * multiple of 4) so that they do not influence the output of reduction.
 */
void reduce_packed_dim(const ivec2 tid, ivec3 scan_pos) {
  // shared memory index of this thread
  const int smi = tid_to_smi(tid);

  // Number of non-padding elements in the last texel in the reduction row
  const int nspill = mod4(tin_sizes[packed_dim]);
  // Only reduce up to the last "complete" texel. The last texel will need to be
  // handled specially if it has padding elements.
  const int reduce_len = tin_sizes[packed_dim] - nspill;

  VEC4_T sum = VEC4_T(0);
  VEC4_T sum_sq = VEC4_T(0);
  int count = 0;

  // Partially accumulate over elements i, i + NWORKERS, i + 2*NWORKERS, ... of
  // the reduction row
  scan_pos[reduce_dim] = tid.x;
  for (int i = tid.x * 4; i < reduce_len;
       i += NWORKERS * 4, scan_pos[reduce_dim] += NWORKERS) {
    VEC4_T val = load_texel(tin, scan_pos);
    sum += val;
    sum_sq += val * val;
    count += 4;
  }
  // For the last texel in the dim, if there are padding elements then each
  // element of the texel needs to be processed individually such that the
  // padding elements are ignored
  if (scan_pos[reduce_dim] == tin_limits[reduce_dim] - 1 && nspill > 0) {
    const VEC4_T val = load_texel(tin, scan_pos);
    for (int i = 0; i < nspill; i++) {
      sum.x += val[i];
      sum_sq.x += val[i] * val[i];
      count += 1;
    }
  }
  // Write partial output to shared memory and synchronize work group
  shared_sum[smi] = sum;
  shared_sum_sq[smi] = sum_sq;
  shared_count[smi] = count;
  barrier();

  // Since the reduction row is reduced to only one element, only the "main"
  // thread in the group needs aggregate the partial outputs
  if (tid.x == 0) {
    sum = shared_sum[tid.y * NWORKERS];
    sum_sq = shared_sum_sq[tid.y * NWORKERS];
    count = shared_count[tid.y * NWORKERS];
    for (int i = 1; i < NWORKERS; i++) {
      int idx = tid.y * NWORKERS + i;
      sum += shared_sum[idx];
      sum_sq += shared_sum_sq[idx];
      count += shared_count[idx];
    }

    // Combine across the elements of the combined state
    float total_sum = sum.x + sum.y + sum.z + sum.w;
    float total_sum_sq = sum_sq.x + sum_sq.y + sum_sq.z + sum_sq.w;
    int total_count = count;

    float mean = total_sum / float(total_count);
    float variance = (total_sum_sq / float(total_count)) - (mean * mean);

    if ((pc.unbiased != 0) && (total_count > 1)) {
      variance = variance * (float(total_count) / float(total_count - 1.0));
    }

    scan_pos[reduce_dim] = tid.x;
    write_texel(tout, scan_pos, VEC4_T(variance, 0, 0, 0));
  }
}

void main() {
  ivec3 scan_pos = ivec3(gl_GlobalInvocationID);
  scan_pos[reduce_dim] = 0;

  const ivec2 tid = ivec2(
      gl_LocalInvocationID[reduce_dim],
      gl_LocalInvocationID[group_dim]);

  if (any(greaterThanEqual(scan_pos, tin_limits))) {
    return;
  }

  if (reduce_dim != packed_dim) {
    reduce_nonpacked_dim(tid, scan_pos);
  } else {
    reduce_packed_dim(tid, scan_pos);
  }
}
