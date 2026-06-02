/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace executorch::backends::webgpu {

// WGSL shader source for rms_norm: y = x * w * rsqrt(mean(x^2) + eps)
inline constexpr const char* kRmsNormWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_in: array<f32>;
@group(0) @binding(2) var<storage, read> t_weight: array<f32>;

struct Params {
  num_rows: u32,
  row_width: u32,
  epsilon: f32,
  _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

const WG_SIZE: u32 = 64u;

var<workgroup> shared_sum: array<f32, WG_SIZE>;

fn reduce_shared(worker_id: u32) {
  workgroupBarrier();
  var stride: u32 = WG_SIZE / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (worker_id < stride) {
      shared_sum[worker_id] = shared_sum[worker_id] + shared_sum[worker_id + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>) {
  let row_idx = wid.x;
  let worker_id = lid.x;

  if (row_idx >= params.num_rows) {
    return;
  }

  let base = row_idx * params.row_width;

  var local_sq_sum: f32 = 0.0;
  var x: u32 = worker_id;
  loop {
    if (x >= params.row_width) {
      break;
    }
    let v = t_in[base + x];
    local_sq_sum = local_sq_sum + v * v;
    x = x + WG_SIZE;
  }

  shared_sum[worker_id] = local_sq_sum;
  reduce_shared(worker_id);

  let mean_sq = shared_sum[0] / f32(params.row_width);
  let rstd = inverseSqrt(mean_sq + params.epsilon);

  x = worker_id;
  loop {
    if (x >= params.row_width) {
      break;
    }
    let v = t_in[base + x];
    let w = t_weight[x];
    t_out[base + x] = v * rstd * w;
    x = x + WG_SIZE;
  }
}
)";

inline constexpr uint32_t kRmsNormWorkgroupSize = 64;

} // namespace executorch::backends::webgpu
