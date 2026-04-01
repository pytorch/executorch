/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "w", "t_mean", DTYPE, "buffer")}
${layout_declare_tensor(B, "w", "t_rstd", DTYPE, "buffer")}

${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}
${layout_declare_ubo(B, "BufferMetadata", "mean_meta")}

layout(push_constant) uniform PRECISION restrict Block {
  float epsilon;
};

#define NUM_WORKERS 64

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "in_layout", "CONTIG_LAYOUT_INT")}

shared T shared_sum[NUM_WORKERS];

void reduce_shared(const uint worker_id) {
  memoryBarrierShared();
  barrier();

  [[unroll]] for (int stride = NUM_WORKERS / 2; stride > 0; stride >>= 1) {
    if (worker_id < stride) {
      shared_sum[worker_id] += shared_sum[worker_id + stride];
    }
    memoryBarrierShared();
    barrier();
  }
}

void main() {
  // Each workgroup handles one output row (one mean/rstd element).
  // gl_GlobalInvocationID.y = row index
  // gl_LocalInvocationID.x = worker_id within the row
  const uint row_idx = gl_GlobalInvocationID.y;
  const uint worker_id = gl_LocalInvocationID.x;

  const uint row_width = width(inp);

  if (row_idx >= numel(mean_meta)) {
    return;
  }

  // Convert row_idx to a tensor index using the mean/rstd metadata.
  // The mean/rstd tensor has shape [..., 1] (width dimension is 1).
  // This gives us the outer dimension indices for this row.
  TensorIndex row_tidx = linear_idx_to_tensor_idx(mean_meta, row_idx, in_layout);

  // The width stride in the input buffer tells us how to step through width
  // elements. For contiguous layout, stride_at(inp, 0) == 1; for other
  // layouts it may differ.
  const uint width_stride = stride_at(inp, 0);

  // Compute the base buffer index for this row in the input tensor.
  // Set width component to 0 and compute the buffer offset.
  row_tidx.data[0][0] = 0;
  const uint base_bufi = tensor_idx_to_linear_idx(inp, row_tidx);

  // Phase 1: Compute mean via cooperative reduction
  T local_sum = T(0);
  for (uint x = worker_id; x < row_width; x += NUM_WORKERS) {
    const uint in_bufi = base_bufi + x * width_stride;
    local_sum += t_in[in_bufi];
  }

  shared_sum[worker_id] = local_sum;
  reduce_shared(worker_id);

  const T mean_val = shared_sum[0] / T(row_width);

  memoryBarrierShared();
  barrier();

  // Phase 2: Compute variance via cooperative reduction
  T local_var = T(0);
  for (uint x = worker_id; x < row_width; x += NUM_WORKERS) {
    const uint in_bufi = base_bufi + x * width_stride;
    const T delta = t_in[in_bufi] - mean_val;
    local_var += delta * delta;
  }

  shared_sum[worker_id] = local_var;
  reduce_shared(worker_id);

  const T var_val = shared_sum[0] / T(row_width);
  const T rstd_val = pow(var_val + T(epsilon), T(-0.5));

  // Phase 3: Normalize and write output
  // Weight and bias are 1D tensors of size [width], indexed directly by x.
  for (uint x = worker_id; x < row_width; x += NUM_WORKERS) {
    const uint in_bufi = base_bufi + x * width_stride;
    const T in_val = t_in[in_bufi];
    const T normalized = (in_val - mean_val) * rstd_val;
    const T w = t_weight[x];
    const T b = t_bias[x];
    t_out[in_bufi] = normalized * w + b;
  }

  // Write mean and rstd (only one thread per row)
  if (worker_id == 0) {
    const uint mean_bufi = tensor_idx_to_linear_idx(mean_meta, row_tidx);
    t_mean[mean_bufi] = mean_val;
    t_rstd[mean_bufi] = rstd_val;
  }
}
