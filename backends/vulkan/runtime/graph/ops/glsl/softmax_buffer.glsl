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

#define op1(X) ${OPERATOR1}

#define op2(X, Y) ${OPERATOR2}

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "out_buf", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "in_buf", DTYPE, STORAGE)}

${layout_declare_ubo(B, "BufferMetadata", "in_meta")}
${layout_declare_ubo(B, "BufferMetadata", "out_meta")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int reduce_dim = 0;

// Supplied by the dispatch so it can scale with the length of the row being
// normalised. Four threads walking a 254-element row left the GPU idle.
layout(constant_id = 4) const int NWORKERS = 4;
// Upper bound on NWORKERS, and the size of the shared arrays below.
#define MAX_NTHREADS 256

shared T shared_max[MAX_NTHREADS];
shared T shared_sum[MAX_NTHREADS];

/*
 * Buffer-based softmax. Each workgroup processes one "row" along the reduction
 * dimension. Within a workgroup, NWORKERS threads cooperate to compute the max
 * and sum reductions, then each thread writes its portion of the final outputs.
 *
 * Thread mapping: the global WG size has 1 along reduce_dim, and all other
 * dimensions correspond to output tensor sizes (WHCN order, with z encoding
 * C*N). The local WG size has NWORKERS along reduce_dim. Each workgroup
 * identifies a unique reduction "row" via the non-reduce dimensions of
 * gl_GlobalInvocationID, and the NWORKERS threads within that workgroup
 * cooperate on the reduction.
 */
void main() {
  // Build the base 4D index for this workgroup's reduction row.
  // gl_GlobalInvocationID has 0..NWORKERS-1 along reduce_dim; zero it out
  // since the tid will iterate over the reduce_dim explicitly.
  ivec3 gid = ivec3(gl_GlobalInvocationID);
  gid[reduce_dim] = 0;

  const int c_size = int(size_at(in_meta, 2));
  TensorIndex4D base_idx;
  base_idx.data = ivec4(gid.x, gid.y, gid.z % c_size, gid.z / c_size);

  if (out_of_bounds(base_idx, in_meta)) {
    return;
  }

  const uint tid = gl_LocalInvocationID[reduce_dim];
  const int R = int(size_at(in_meta, reduce_dim));

  // Phase 1: Find maximum along reduce_dim
  TensorIndex4D in_idx = base_idx;

  T local_max = T(-3.402823e+38);
  for (int i = int(tid); i < R; i += NWORKERS) {
    in_idx.data[reduce_dim] = i;
    T v = in_buf[tensor4d_idx_to_linear_idx(in_meta, in_idx)];
    local_max = max(local_max, v);
  }
  shared_max[tid] = local_max;
  barrier();

  // Combine the partials as a tree. This was a serial walk of all NWORKERS
  // entries run by EVERY thread, so its cost grew with the worker count and
  // cancelled most of the benefit of widening the group. The loop bound is
  // uniform and barrier() sits outside the guard, so every thread reaches
  // every barrier.
  for (uint s = uint(NWORKERS) >> 1; s > 0u; s >>= 1) {
    if (tid < s) {
      shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
    }
    barrier();
  }
  const T max_val = shared_max[0];

  // Phase 2: Compute sum of exp(x - max_val)
  T local_sum = T(0);
  for (int i = int(tid); i < R; i += NWORKERS) {
    in_idx.data[reduce_dim] = i;
    T v = in_buf[tensor4d_idx_to_linear_idx(in_meta, in_idx)];
    local_sum += exp(v - max_val);
  }
  shared_sum[tid] = local_sum;
  barrier();

  // Same tree reduction for the sum.
  for (uint s = uint(NWORKERS) >> 1; s > 0u; s >>= 1) {
    if (tid < s) {
      shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
    }
    barrier();
  }
  // Clamp denominator to avoid 0/0 = NaN when all exp values underflow.
  const T sum_val = max(shared_sum[0], T(1e-37));

  // Phase 3: Write outputs
  for (int i = int(tid); i < R; i += NWORKERS) {
    in_idx.data[reduce_dim] = i;
    uint in_buf_idx = tensor4d_idx_to_linear_idx(in_meta, in_idx);
    T v = in_buf[in_buf_idx];
    T numerator = op1(v - max_val);
    T result = op2(numerator, sum_val);

    // Replace NaN/Inf with 0 using IEEE 754 bit-level manipulation
    uint bits = floatBitsToUint(result);
    if ((bits & 0x7F800000u) == 0x7F800000u) {
      result = T(0);
    }

    out_buf[tensor4d_idx_to_linear_idx(out_meta, in_idx)] = result;
  }
}
