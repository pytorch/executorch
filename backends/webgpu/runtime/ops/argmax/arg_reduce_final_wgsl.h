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

// @generated from arg_reduce_final.wgsl - DO NOT EDIT.
// wgsl-sha256: efded2fdd0607954d05602c55f7ae6ec6e0dd732cb09e5468d16d7a788489284
inline constexpr const char* kArgReduceFinalWGSL = R"(
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

struct Partial {
  val: f32,
  idx: u32,
}
@group(0) @binding(0) var<storage, read> t_part: array<Partial>;
@group(0) @binding(1) var<storage, read_write> t_out: array<u32>;

struct Params {
  num_rows: u32,
  reduce_size: u32,
  is_argmin: u32,
  num_parts: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

// Stage 2: one workgroup per row merges that row's `num_parts` stage-1
// partials and writes the winning index. Same lower-index-wins rule as
// arg_reduce.wgsl:57-61; every partial is a real (value, index) pair drawn
// from the row, so merging them in any order yields the row's lowest-index
// extremum -- the identical result to the single-workgroup kernel.
const WG: u32 = 64u;
var<workgroup> part_val: array<f32, WG>;
var<workgroup> part_idx: array<u32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let row = wid.x + wid.y * num_workgroups.x;
  if (row >= params.num_rows) {
    return;
  }
  let pbase = row * params.num_parts;

  var best = t_part[pbase].val;
  var best_idx = t_part[pbase].idx;
  var t = lid.x;
  while (t < params.num_parts) {
    let v = t_part[pbase + t].val;
    let idx = t_part[pbase + t].idx;
    if (params.is_argmin != 0u) {
      if (v < best || (v == best && idx < best_idx)) { best = v; best_idx = idx; }
    } else {
      if (v > best || (v == best && idx < best_idx)) { best = v; best_idx = idx; }
    }
    t = t + WG;
  }
  part_val[lid.x] = best;
  part_idx[lid.x] = best_idx;
  workgroupBarrier();

  var stride: u32 = WG >> 1u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (lid.x < stride) {
      let bv = part_val[lid.x];
      let bi = part_idx[lid.x];
      let v = part_val[lid.x + stride];
      let idx = part_idx[lid.x + stride];
      if (params.is_argmin != 0u) {
        if (v < bv || (v == bv && idx < bi)) {
          part_val[lid.x] = v;
          part_idx[lid.x] = idx;
        }
      } else {
        if (v > bv || (v == bv && idx < bi)) {
          part_val[lid.x] = v;
          part_idx[lid.x] = idx;
        }
      }
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (lid.x == 0u) {
    t_out[row] = part_idx[0];
  }
}
)";

inline constexpr uint32_t kArgReduceFinalWorkgroupSizeX = 64;
inline constexpr uint32_t kArgReduceFinalWorkgroupSizeY = 1;
inline constexpr uint32_t kArgReduceFinalWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
