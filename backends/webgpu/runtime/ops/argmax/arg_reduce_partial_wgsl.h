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

// @generated from arg_reduce_partial.wgsl - DO NOT EDIT.
// wgsl-sha256: 12dc45f7d391b2b531e0a028786bde612541e061934466b68dc0d1806899a9bb
inline constexpr const char* kArgReducePartialWGSL = R"(
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

@group(0) @binding(0) var<storage, read> t_in: array<f32>;

struct Partial {
  val: f32,
  idx: u32,
}
@group(0) @binding(1) var<storage, read_write> t_part: array<Partial>;

struct Params {
  num_rows: u32,
  reduce_size: u32,
  is_argmin: u32,
  num_parts: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

// Stage 1 of the two-stage arg-reduction. The single-workgroup kernel
// (arg_reduce.wgsl) puts an entire row on ONE 64-lane workgroup; here each row
// is split into `num_parts` contiguous chunks, one workgroup per chunk, so the
// row's elements are scanned by num_parts x WG lanes instead of WG.
//
// Semantics are held IDENTICAL to arg_reduce.wgsl: every lane is seeded with
// (t_in[base], 0) -- element 0 is always a valid candidate -- and scans with a
// STRICT compare, so the lowest index wins inside a lane. The in-workgroup
// merge is a halving tree using the same lower-index-wins rule; because no
// partial can ever hold a NaN unless t_in[base] itself is NaN (a strict
// compare never selects NaN), the merge operator is a total order and the tree
// is order-independent -> bit-exact index equality with the serial tail.
const WG: u32 = 64u;
var<workgroup> part_val: array<f32, WG>;
var<workgroup> part_idx: array<u32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // 2D-folded grid of num_rows * num_parts workgroups.
  let slot = wid.x + wid.y * num_workgroups.x;
  let row = slot / params.num_parts;
  if (row >= params.num_rows) {
    return;
  }
  let part = slot - row * params.num_parts;
  let base = row * params.reduce_size;

  let chunk = (params.reduce_size + params.num_parts - 1u) / params.num_parts;
  let start = part * chunk;
  var end = start + chunk;
  if (end > params.reduce_size) {
    end = params.reduce_size;
  }

  var best = t_in[base];
  var best_idx: u32 = 0u;
  var k = start + lid.x;
  while (k < end) {
    let v = t_in[base + k];
    if (params.is_argmin != 0u) {
      if (v < best) { best = v; best_idx = k; }
    } else {
      if (v > best) { best = v; best_idx = k; }
    }
    k = k + WG;
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
    t_part[slot].val = part_val[0];
    t_part[slot].idx = part_idx[0];
  }
}
)";

inline constexpr uint32_t kArgReducePartialWorkgroupSizeX = 64;
inline constexpr uint32_t kArgReducePartialWorkgroupSizeY = 1;
inline constexpr uint32_t kArgReducePartialWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
