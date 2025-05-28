/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define T ${buffer_scalar_type(DTYPE)}

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "out_buf", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "in_buf", DTYPE, STORAGE)}

${layout_declare_ubo(B, "ivec4", "in_sizes")}
${layout_declare_ubo(B, "ivec4", "in_strides")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}
${layout_declare_ubo(B, "ivec4", "out_strides")}

layout(push_constant) uniform PushConstants {
  int unbiased;
} pc;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int reduce_dim = 0;

$if VARIANCE_MODE:
  #define VARIANCE_MODE

#define NWORKERS 4
#define MAX_THREADS 16

shared T shared_accum[NWORKERS];
#ifdef VARIANCE_MODE
shared T shared_sum_sq[NWORKERS];
shared int shared_count[NWORKERS];
#endif

#include "indexing_utils.h"

#define INIT_ACCUM(first_val) ${INIT_ACCUM}
#define UPDATE_ACCUM(accum, new_val) ${UPDATE_ACCUM}
#define POSTPROCESS(accum) ${POSTPROCESS}

void main() {
  const ivec4 out_idx = ivec4(
      gl_GlobalInvocationID.x,
      gl_GlobalInvocationID.y,
      gl_GlobalInvocationID.z % out_sizes.z,
      gl_GlobalInvocationID.z / out_sizes.z);

  const uint tid = gl_LocalInvocationID[reduce_dim];

  shared_accum[tid] = T(0);
#ifdef VARIANCE_MODE
  shared_sum_sq[tid] = T(0);
  shared_count[tid] = 0;
#endif
  barrier();

  const int R = in_sizes[reduce_dim];
  const uint N = gl_WorkGroupSize[reduce_dim];

  // Each workgroup processes a contiguous chunk of the input tensor
  // along the reduce_dim. Each thread processes a part of this chunk.
  uint q = R / N;
  uint rem = R % N;

  uint len = q + (tid < rem ? 1u : 0u);
  uint base = tid * q + min(tid, rem);

  // Get the first value for initializing the accumulator if needed
  T first_val = T(0);
  if (R > 0) {
    ivec4 first_idx = out_idx;
    first_idx[reduce_dim] = 0;

    if (reduce_dim == 2) {
      first_idx[reduce_dim + 1] = 0;
    }

    first_val = in_buf[tidx_to_bufi(first_idx, in_strides)];
  }

  // Initialize accumulator
  T accum = INIT_ACCUM(first_val);
#ifdef VARIANCE_MODE
  T sum_sq = T(0);
  int count = 0;
#endif

  ivec4 in_idx = out_idx;
  for (uint off = 0u; off < len; ++off) {
    uint i = base + off;
    in_idx[reduce_dim] = int(i);

    // out_idx is a 4D index, so for tensors with reduce_dim == 2,
    // we need to set the reduce_dim + 1 to 0 as gl_GlobalInvocationID.z
    // is influenced by the tid
    if (reduce_dim == 2) {
      in_idx[reduce_dim + 1] -= int(tid);
    }

    T v = in_buf[tidx_to_bufi(in_idx, in_strides)];

    accum = UPDATE_ACCUM(accum, v);

#ifdef VARIANCE_MODE
    sum_sq += v * v;
    count += 1;
#endif
  }

  shared_accum[tid] = accum;
#ifdef VARIANCE_MODE
  shared_sum_sq[tid] = sum_sq;
  shared_count[tid] = count;
#endif
  barrier();

  if (tid == 0u) {
    T result = shared_accum[0];

#ifdef VARIANCE_MODE
    T tot_sum = shared_accum[0];
    T tot_sum_sq = shared_sum_sq[0];
    int tot_count = shared_count[0];
#endif

    for (uint i = 1; i < N; ++i) {
#ifdef VARIANCE_MODE
      tot_sum += shared_accum[i];
      tot_sum_sq += shared_sum_sq[i];
      tot_count += shared_count[i];
#else
      result = UPDATE_ACCUM(result, shared_accum[i]);
#endif
    }

#ifdef VARIANCE_MODE
    if (tot_count > 0) {
      T mean = tot_sum / T(tot_count);
      result = (tot_sum_sq / T(tot_count)) - (mean * mean);
      if (pc.unbiased != 0 && tot_count > 1) {
        result *= T(tot_count) / T(tot_count - 1);
      }
    } else {
      // NaN to match PyTorch behavior
      result = T(0.0/0.0);
    }
#else
    result = POSTPROCESS(result);
#endif

    out_buf[tidx_to_bufi(out_idx, out_strides)] = result;
  }
}
