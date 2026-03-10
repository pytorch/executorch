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

layout(std430) buffer;

${layout_declare_tensor(B, "w", "out_buf", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "in_buf", DTYPE, STORAGE)}

${layout_declare_ubo(B, "ivec4", "in_sizes")}
${layout_declare_ubo(B, "ivec4", "in_strides")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}
${layout_declare_ubo(B, "ivec4", "out_strides")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int reduce_dim = 0;

#define NWORKERS 4
#define MAX_NTHREADS 16

shared T shared_max[NWORKERS];
shared T shared_sum[NWORKERS];

#include "indexing_utils.h"

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

  ivec4 base_idx = ivec4(gid.x, gid.y, gid.z % in_sizes.z, gid.z / in_sizes.z);

  if (any(greaterThanEqual(base_idx, in_sizes))) {
    return;
  }

  const uint tid = gl_LocalInvocationID[reduce_dim];
  const int R = in_sizes[reduce_dim];

  // Phase 1: Find maximum along reduce_dim
  ivec4 in_idx = base_idx;

  T local_max = T(-3.402823e+38);
  for (int i = int(tid); i < R; i += NWORKERS) {
    in_idx[reduce_dim] = i;
    T v = in_buf[tidx_to_bufi(in_idx, in_strides)];
    local_max = max(local_max, v);
  }
  shared_max[tid] = local_max;
  barrier();

  // Reduce partial maximums across workers
  T max_val = shared_max[0];
  for (int i = 1; i < NWORKERS; ++i) {
    max_val = max(max_val, shared_max[i]);
  }

  // Phase 2: Compute sum of exp(x - max_val)
  T local_sum = T(0);
  for (int i = int(tid); i < R; i += NWORKERS) {
    in_idx[reduce_dim] = i;
    T v = in_buf[tidx_to_bufi(in_idx, in_strides)];
    local_sum += exp(v - max_val);
  }
  shared_sum[tid] = local_sum;
  barrier();

  // Reduce partial sums across workers
  T sum_val = shared_sum[0];
  for (int i = 1; i < NWORKERS; ++i) {
    sum_val += shared_sum[i];
  }
  // Clamp denominator to avoid 0/0 = NaN when all exp values underflow.
  sum_val = max(sum_val, T(1e-37));

  // Phase 3: Write outputs
  for (int i = int(tid); i < R; i += NWORKERS) {
    in_idx[reduce_dim] = i;
    int in_buf_idx = tidx_to_bufi(in_idx, in_strides);
    T v = in_buf[in_buf_idx];
    T numerator = op1(v - max_val);
    T result = op2(numerator, sum_val);

    // Replace NaN/Inf with 0 using IEEE 754 bit-level manipulation
    uint bits = floatBitsToUint(result);
    if ((bits & 0x7F800000u) == 0x7F800000u) {
      result = T(0);
    }

    out_buf[tidx_to_bufi(in_idx, out_strides)] = result;
  }
}
