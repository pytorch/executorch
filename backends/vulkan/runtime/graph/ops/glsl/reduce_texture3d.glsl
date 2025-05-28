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

$if VARIANCE_MODE:
  #define VARIANCE_MODE

// A more verbose name would be NWORKERS_PER_GROUP. This describes the number of
// threads that will co-operate to compute one reduction output. There may be
// multiple groups computing distinct reduction outputs within one work group.
#define NWORKERS 4

// Sets an upper limit on the total size of a work group based on how many
// elements are allocated in the shared memory array below. Each thread in the
// work group will write into its assigned element in the shared array.
#define MAX_NTHREADS 16

shared VEC4_T shared_vecs[MAX_NTHREADS];
// Second accumulator for variance mode - used for sum of values, prev
// accumulator is used for sum of squares
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

/*
 * The functions below compute reduction along a single dimension for a tensor.
 * The shader template generalize reduction by abstracting the initial value of
 * the accumulator, the calculation used to update the accumulator with new
 * values, and a postprocessing calculation that can be used to modify the
 * accumulator before writing to output.
 *
 * This shader also utilize shared memory to have multiple threads help compute
 * the max and sum reduction operations. A total of NGROUPS x NWORKERS threads
 * are expected to be launched. Each group works on a unique reduction "row", and
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
 */

// Initializing the accumulator accepts the first value in the reduction row,
// since some reduction operations (i.e. amax, amin) prefer to initialize with
// a data point instead of a static value.
#define INIT_ACCUM(first_val) ${INIT_ACCUM}
#define UPDATE_ACCUM(accum, new_val) ${UPDATE_ACCUM}
// Useful for operators such as mean which want to perform a final calculation
// with the accumulator.
#define POSTPROCESS(accum) ${POSTPROCESS}

/*
 * Computes reduction where the reduction dim is orthogonal to the packed dim.
 * This case is simpler because each element of a texel belongs to a separate
 * reduction "group", meaning we don't have to perform reduction along a texel.
 */
void reduce_nonpacked_dim(const ivec2 tid, ivec3 scan_pos) {
  // shared memory index of this thread
  const int smi = tid_to_smi(tid);

  scan_pos[reduce_dim] = 0;
  VEC4_T accum = INIT_ACCUM(load_texel(tin, scan_pos));

#ifdef VARIANCE_MODE
  VEC4_T sum_sq = VEC4_T(0);
  int count = 0;
#endif

  scan_pos[reduce_dim] = tid.x;
  // Partially accumulate over elements i, i + NWORKERS, i + 2*NWORKERS, ... of
  // the reduction row
  for (int i = tid.x; i < tin_sizes[reduce_dim];
       i += NWORKERS, scan_pos[reduce_dim] += NWORKERS) {
    VEC4_T val = load_texel(tin, scan_pos);
    accum = UPDATE_ACCUM(accum, val);
#ifdef VARIANCE_MODE
    sum_sq += val * val;
    count += 1;
#endif
  }
  // Write partial output to shared memory and synchronize work group
  shared_vecs[smi] = accum;
#ifdef VARIANCE_MODE
  shared_sum_sq[smi] = sum_sq;
  shared_count[smi] = count;
#endif
  barrier();

  // Since the reduction row is reduced to only one element, only the "main"
  // thread in the group needs aggregate the partial outputs
  if (tid.x == 0) {
    // Iterate over the partial outputs to obtain the overall output
    int group_i = tid.y * NWORKERS;
    accum = shared_vecs[group_i];
#ifdef VARIANCE_MODE
    sum_sq = shared_sum_sq[group_i];
    count = shared_count[group_i];
#endif
    for (int i = 1; i < NWORKERS; i++) {
      int idx = tid.y * NWORKERS + i;
      accum = UPDATE_ACCUM(accum, shared_vecs[idx]);
#ifdef VARIANCE_MODE
      sum_sq += shared_sum_sq[idx];
      count += shared_count[idx];
#endif
    }

    // Determine if there are any padding elements in the final texel of the
    // packed dimension
    const int nspill = mod4(tin_sizes[packed_dim]);
    // Detect if this thread is working on the final texels of the packed
    // dimension, which may have padding elements
    const bool is_last_texel =
        scan_pos[packed_dim] == (tin_limits[packed_dim] - 1);

#ifdef VARIANCE_MODE
    VEC4_T variance = calculate_variance(accum, sum_sq, count);
#endif

    // Explicitly set padding elements to 0
    if (is_last_texel && nspill > 0) {
      [[unroll]] for (int i = nspill; i < 4; i++) {
#ifdef VARIANCE_MODE
        variance[i] = 0;
#else
        accum[i] = 0;
#endif
      }
    }

    scan_pos[reduce_dim] = tid.x;
#ifdef VARIANCE_MODE
    write_texel(tout, scan_pos, variance);
#else
    write_texel(tout, scan_pos, POSTPROCESS(accum));
#endif
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

  scan_pos[reduce_dim] = 0;
  VEC4_T accum = INIT_ACCUM(VEC4_T(load_texel(tin, scan_pos).x));

#ifdef VARIANCE_MODE
  VEC4_T sum_sq = VEC4_T(0);
  int count = 0;
#endif

  // Partially accumulate over elements i, i + NWORKERS, i + 2*NWORKERS, ... of
  // the reduction row
  scan_pos[reduce_dim] = tid.x;
  for (int i = tid.x * 4; i < reduce_len;
       i += NWORKERS * 4, scan_pos[reduce_dim] += NWORKERS) {
    VEC4_T val = load_texel(tin, scan_pos);
    accum = UPDATE_ACCUM(accum, val);
#ifdef VARIANCE_MODE
    sum_sq += val * val;
    count += 4; // Each texel has 4 elements
#endif
  }
  // For the last texel in the dim, if there are padding elements then each
  // element of the texel needs to be processed individually such that the
  // padding elements are ignored
  if (scan_pos[reduce_dim] == tin_limits[reduce_dim] - 1 && nspill > 0) {
    const VEC4_T val = load_texel(tin, scan_pos);
    for (int i = 0; i < nspill; i++) {
      accum.x = UPDATE_ACCUM(accum.x, val[i]);
#ifdef VARIANCE_MODE
      sum_sq.x += val[i] * val[i];
      count += 1;
#endif
    }
  }
  // Write partial output to shared memory and synchronize work group
  shared_vecs[smi] = accum;
#ifdef VARIANCE_MODE
  shared_sum_sq[smi] = sum_sq;
  shared_count[smi] = count;
#endif
  barrier();

  // Since the reduction row is reduced to only one element, only the "main"
  // thread in the group needs aggregate the partial outputs
  if (tid.x == 0) {
    // Iterate over the partial maximums to obtain the overall maximum
    int group_i = tid.y * NWORKERS;
    accum = shared_vecs[group_i];
#ifdef VARIANCE_MODE
    sum_sq = shared_sum_sq[group_i];
    count = shared_count[group_i];
#endif
    for (int i = 1; i < NWORKERS; i++, group_i++) {
      int idx = tid.y * NWORKERS + i;
      accum = UPDATE_ACCUM(accum, shared_vecs[idx]);
#ifdef VARIANCE_MODE
      sum_sq += shared_sum_sq[idx];
      count += shared_count[idx];
#endif
    }

#ifdef VARIANCE_MODE
    float total_sum = accum.x + accum.y + accum.z + accum.w;
    float total_sum_sq = sum_sq.x + sum_sq.y + sum_sq.z + sum_sq.w;
    int total_count = count;

    float mean = total_sum / float(total_count);
    float variance = (total_sum_sq / float(total_count)) - (mean * mean);

    if ((pc.unbiased != 0) && (total_count > 1)) {
      variance = variance * (float(total_count) / float(total_count - 1.0));
    }

    scan_pos[reduce_dim] = tid.x;
    write_texel(tout, scan_pos, VEC4_T(variance, 0, 0, 0));
#else
    // Each element of the texel is itself a partial maximum; iterate over the
    // texel to find the actual maximum
    float accum_final = accum.x;
    [[unroll]] for (int i = 1; i < 4; i++) {
      accum_final = UPDATE_ACCUM(accum[i], accum_final);
    }

    scan_pos[reduce_dim] = tid.x;
    write_texel(tout, scan_pos, POSTPROCESS(VEC4_T(accum_final, 0, 0, 0)));
#endif
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
