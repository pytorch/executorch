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

// @generated from fused_ce.wgsl - DO NOT EDIT.
// wgsl-sha256: 416484a64dc4f31940ca63965e28b5747b34cd6b3a64e141ffaa5835ec28891d
inline constexpr const char* kFusedCeWGSL = R"(

@group(0) @binding(0) var<storage, read_write> dlogits: array<f32>;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> labels: array<i32>;
@group(0) @binding(3) var<storage, read_write> loss_partial: array<f32>;

struct Params { vocab: u32, n_rows: u32, n_valid: f32, pad: f32 }
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 256u;
var<workgroup> red_m: array<f32, 256>;
var<workgroup> red_l: array<f32, 256>;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  if (row >= params.n_rows) { return; }
  let tid = lid.x;
  let V = params.vocab;
  let base = row * V;
  let lbl = labels[row];

  if (lbl < 0) {
    for (var j = tid; j < V; j = j + wg_size) { dlogits[base + j] = 0.0; }
    if (tid == 0u) { loss_partial[row] = 0.0; }
    return;
  }

  // -3.4e38 not 3.4028235e38 (Tint f32 overflow)
  var m = -3.4e38;
  var l = 0.0;
  for (var j = tid; j < V; j = j + wg_size) {
    let x = logits[base + j];
    if (x > m) {
      l = l * exp(m - x) + 1.0;
      m = x;
    } else {
      l = l + exp(x - m);
    }
  }
  red_m[tid] = m;
  red_l[tid] = l;
  workgroupBarrier();
  for (var s = wg_size / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      let ma = red_m[tid];
      let la = red_l[tid];
      let mb = red_m[tid + s];
      let lb = red_l[tid + s];
      let mm = max(ma, mb);
      red_m[tid] = mm;
      red_l[tid] = la * exp(ma - mm) + lb * exp(mb - mm);
    }
    workgroupBarrier();
  }
  let row_max = red_m[0];
  let denom = red_l[0];
  workgroupBarrier();

  let inv = 1.0 / denom;
  let scale = 1.0 / params.n_valid;
  if (tid == 0u) {
    let lse = row_max + log(denom);
    loss_partial[row] = (lse - logits[base + u32(lbl)]) * scale;
  }
  workgroupBarrier();
  for (var j = tid; j < V; j = j + wg_size) {
    var g = exp(logits[base + j] - row_max) * inv * scale;
    if (j == u32(lbl)) { g = g - scale; }
    dlogits[base + j] = g;
  }
}
)";

inline constexpr uint32_t kFusedCeWorkgroupSizeX = 256;
inline constexpr uint32_t kFusedCeWorkgroupSizeY = 1;
inline constexpr uint32_t kFusedCeWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
