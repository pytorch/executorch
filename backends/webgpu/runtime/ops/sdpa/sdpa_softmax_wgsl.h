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

// @generated from sdpa_softmax.wgsl - DO NOT EDIT.
// wgsl-sha256: 774a372c0eae85b6656618408e40fff4b8af5647ea5353b51d3a34b9b6a7ac8a
inline constexpr const char* kSdpaSoftmaxWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_in: array<f32>;

struct Params {
  num_rows: u32,
  row_width: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

// wg_size must be a power of two (the tree reduction halves the stride).
override wg_size: u32 = 64u;

// WGSL forbids literal -inf; a large finite negative inits the running max.
const NEG_INF: f32 = -1.0e30;

var<workgroup> shared_max: array<f32, wg_size>;
var<workgroup> shared_sum: array<f32, wg_size>;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // One workgroup per (h, s) row of length context_len (= row_width).
  // 2D dispatch fold: recover the linear row index. Keep the `valid` predicate
  // below (NOT an early return) — the workgroupBarrier()s require uniform flow.
  let row_idx = wid.x + wid.y * num_workgroups.x;
  let worker_id = lid.x;

  let base = row_idx * params.row_width;
  let valid = row_idx < params.num_rows;
  let width = params.row_width;

  // Pass 1: row max (stable softmax). Threads stride over the row.
  var local_max: f32 = NEG_INF;
  if (valid) {
    var x: u32 = worker_id;
    loop {
      if (x >= width) {
        break;
      }
      local_max = max(local_max, t_in[base + x]);
      x = x + wg_size;
    }
  }
  shared_max[worker_id] = local_max;

  // Reduce max. workgroupBarrier() calls are in uniform control flow.
  workgroupBarrier();
  var stride: u32 = wg_size / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (worker_id < stride) {
      shared_max[worker_id] = max(shared_max[worker_id], shared_max[worker_id + stride]);
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let row_max = shared_max[0];

  // Pass 2: sum of exp(x - max).
  var local_sum: f32 = 0.0;
  if (valid) {
    var x: u32 = worker_id;
    loop {
      if (x >= width) {
        break;
      }
      local_sum = local_sum + exp(t_in[base + x] - row_max);
      x = x + wg_size;
    }
  }
  shared_sum[worker_id] = local_sum;

  workgroupBarrier();
  stride = wg_size / 2u;
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
  let row_sum = shared_sum[0];

  // Pass 3: normalize. Guard division by zero defensively.
  if (valid) {
    let inv = select(0.0, 1.0 / row_sum, row_sum > 0.0);
    var x: u32 = worker_id;
    loop {
      if (x >= width) {
        break;
      }
      t_out[base + x] = exp(t_in[base + x] - row_max) * inv;
      x = x + wg_size;
    }
  }
}
)";

inline constexpr uint32_t kSdpaSoftmaxWorkgroupSizeX = 64;
inline constexpr uint32_t kSdpaSoftmaxWorkgroupSizeY = 1;
inline constexpr uint32_t kSdpaSoftmaxWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
