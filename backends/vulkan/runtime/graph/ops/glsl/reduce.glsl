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


shared vec4 shared_vecs[MAX_NTHREADS];

#include "indexing_utils.h"

int tid_to_smi(const ivec2 tid) {
  return tid.x + tid.y * NWORKERS;
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
  vec4 accum = INIT_ACCUM(load_texel(tin, scan_pos));

  scan_pos[reduce_dim] = tid.x;
  // Partially accumulate over elements i, i + NWORKERS, i + 2*NWORKERS, ... of
  // the reduction row
  for (int i = tid.x; i < tin_sizes[reduce_dim];
       i += NWORKERS, scan_pos[reduce_dim] += NWORKERS) {
    accum = UPDATE_ACCUM(accum, load_texel(tin, scan_pos));
  }
  // Write partial output to shared memory and synchronize work group
  shared_vecs[smi] = accum;
  barrier();

  // Since the reduction row is reduced to only one element, only the "main"
  // thread in the group needs aggregate the partial outputs
  if (tid.x == 0) {
    // Iterate over the partial outputs to obtain the overall output
    int group_i = tid.y * NWORKERS;
    accum = shared_vecs[group_i++];
    for (int i = 1; i < NWORKERS; i++, group_i++) {
      accum = UPDATE_ACCUM(accum, shared_vecs[group_i]);
    }

    // Determine if there are any padding elements in the final texel of the
    // packed dimension
    const int nspill = mod4(tin_sizes[packed_dim]);
    // Detect if this thread is working on the final texels of the packed
    // dimension, which may have padding elements
    const bool is_last_texel =
        scan_pos[packed_dim] == (tin_limits[packed_dim] - 1);

    // Explicitly set padding elements to 0
    if (is_last_texel && nspill > 0) {
      [[unroll]] for (int i = nspill; i < 4; i++) {
        accum[i] = 0;
      }
    }
    scan_pos[reduce_dim] = tid.x;
    write_texel(tout, scan_pos, POSTPROCESS(accum));
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
  vec4 accum = INIT_ACCUM(vec4(load_texel(tin, scan_pos).x));

  // Partially accumulate over elements i, i + NWORKERS, i + 2*NWORKERS, ... of
  // the reduction row
  scan_pos[reduce_dim] = tid.x;
  for (int i = tid.x * 4; i < reduce_len;
       i += NWORKERS * 4, scan_pos[reduce_dim] += NWORKERS) {
    accum = UPDATE_ACCUM(accum, load_texel(tin, scan_pos));
  }
  // For the last texel in the dim, if there are padding elements then each
  // element of the texel needs to be processed individually such that the
  // padding elements are ignored
  if (scan_pos[reduce_dim] == tin_limits[reduce_dim] - 1 && nspill > 0) {
    const vec4 intex = load_texel(tin, scan_pos);
    for (int i = 0; i < nspill; i++) {
      accum.x = UPDATE_ACCUM(accum.x, intex[i]);
    }
  }
  // Write partial output to shared memory and synchronize work group
  shared_vecs[smi] = accum;
  barrier();

  // Since the reduction row is reduced to only one element, only the "main"
  // thread in the group needs aggregate the partial outputs
  if (tid.x == 0) {
    // Iterate over the partial maximums to obtain the overall maximum
    int group_i = tid.y * NWORKERS;
    accum = shared_vecs[group_i++];
    for (int i = 1; i < NWORKERS; i++, group_i++) {
      accum = UPDATE_ACCUM(accum, shared_vecs[group_i]);
    }
    // Each element of the texel is itself a partial maximum; iterate over the
    // texel to find the actual maximum
    float accum_final = accum.x;
    [[unroll]] for (int i = 1; i < 4; i++) {
      accum_final = UPDATE_ACCUM(accum[i], accum_final);
    }

    scan_pos[reduce_dim] = tid.x;
    write_texel(tout, scan_pos, POSTPROCESS(vec4(accum_final, 0, 0, 0)));
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
