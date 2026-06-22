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
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(push_constant) uniform PRECISION restrict Block {
  float epsilon;
};

#define NUM_WORKERS 64

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "in_layout", "CONTIG_LAYOUT_INT")}

shared float shared_sum[NUM_WORKERS];

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
  const uint row_idx = gl_GlobalInvocationID.y;
  const uint worker_id = gl_LocalInvocationID.x;

  const uint row_width = width(inp);
  const uint num_rows = numel(inp) / row_width;

  if (row_idx >= num_rows) {
    return;
  }

  // Decompose row_idx into H, C, N indices using inp sizes (WHCN order)
  uint remaining = row_idx;
  const uint H = uint(inp.sizes[0][1]);
  const uint C = uint(inp.sizes[0][2]);

  const uint h = remaining % H;
  remaining /= H;
  const uint c = remaining % C;
  const uint n = remaining / C;

  // Build tensor index with w=0 to get base buffer index for this row
  TensorIndex tidx;
  tidx.data[0] = uvec4(0, h, c, n);
  tidx.data[1] = uvec4(0);
  const uint base_bufi = tensor_idx_to_linear_idx(inp, tidx);
  const uint width_stride = stride_at(inp, 0);

  // Phase 1: Compute mean(x^2) via cooperative reduction in fp32
  float local_sq_sum = 0.0;
  for (uint x = worker_id; x < row_width; x += NUM_WORKERS) {
    const uint in_bufi = base_bufi + x * width_stride;
    const float val = float(t_in[in_bufi]);
    local_sq_sum += val * val;
  }

  shared_sum[worker_id] = local_sq_sum;
  reduce_shared(worker_id);

  const float mean_sq = shared_sum[0] / float(row_width);
  const float rstd = inversesqrt(mean_sq + epsilon);

  // Phase 2: Normalize and write output
  for (uint x = worker_id; x < row_width; x += NUM_WORKERS) {
    const uint in_bufi = base_bufi + x * width_stride;
    const float val = float(t_in[in_bufi]);
    const float w = float(t_weight[x]);
    t_out[in_bufi] = T(val * rstd * w);
  }
}
