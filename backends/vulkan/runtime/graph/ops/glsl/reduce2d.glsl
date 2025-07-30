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
layout(constant_id = 4) const int reduce_dim1 = 0;
layout(constant_id = 5) const int reduce_dim2 = 1;
layout(constant_id = 6) const int group_dim = 2;

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

// Initializing the accumulator accepts the first value in the reduction row,
// since some reduction operations (i.e. amax, amin) prefer to initialize with
// a data point instead of a static value.
#define INIT_ACCUM(first_val) ${INIT_ACCUM}
#define UPDATE_ACCUM(accum, new_val) ${UPDATE_ACCUM}
// Useful for operators such as mean which want to perform a final calculation
// with the accumulator.
#define POSTPROCESS(accum) ${POSTPROCESS}

void reduce_2d_non_packed_dim(const ivec2 tid, ivec3 scan_pos) {
  // shared memory index of this thread
  const int smi = tid_to_smi(tid);

  scan_pos[reduce_dim1] = 0;
  scan_pos[reduce_dim2] = 0;
  vec4 accum = INIT_ACCUM(load_texel(tin, scan_pos));
  
  // First dimension reduction
  scan_pos[reduce_dim1] = tid.x;
  for (int i = tid.x; i < tin_sizes[reduce_dim1]; 
       i += NWORKERS, scan_pos[reduce_dim1] += NWORKERS) {
    
    // Second dimension reduction
    scan_pos[reduce_dim2] = 0;
    for (int j = 0; j < tin_sizes[reduce_dim2]; j++, scan_pos[reduce_dim2]++) {
      accum = UPDATE_ACCUM(accum, load_texel(tin, scan_pos));
    }
  }
  
  // Write partial output to shared memory and synchronize
  shared_vecs[smi] = accum;
  barrier();
  
  // Main thread aggregates results
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
    scan_pos[reduce_dim1] = 0;
    scan_pos[reduce_dim2] = 0;
    write_texel(tout, scan_pos, POSTPROCESS(accum));
  }
}

void main() {
  ivec3 scan_pos = ivec3(gl_GlobalInvocationID);
  scan_pos[reduce_dim1] = 0;
  scan_pos[reduce_dim2] = 0;

  const ivec2 tid = ivec2(
      gl_LocalInvocationID[reduce_dim1],
      gl_LocalInvocationID[group_dim]);

  if (any(greaterThanEqual(scan_pos, tin_limits))) {
    return;
  }

  reduce_2d_non_packed_dim(tid, scan_pos);
}