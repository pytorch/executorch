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

// @generated from q8ta_add.wgsl - DO NOT EDIT.
// wgsl-sha256: 67ce678dbe548b3b29a264dcd2b3070af903e17306ba23cc7ca14dbf629e72c6
inline constexpr const char* kQ8taAddWGSL = R"(
@group(0) @binding(0) var<storage, read> t_a: array<u32>;
@group(0) @binding(1) var<storage, read> t_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> t_out: array<u32>;

struct Params {
  inv_output_scale: f32,
  a_scale: f32,
  b_scale: f32,
  alpha: f32,
  a_zero_point: i32,
  b_zero_point: i32,
  output_zero_point: i32,
  numel: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64u;

fn unpack_i8(word: u32, j: u32) -> i32 {
  return i32(((word >> (j * 8u)) & 0xFFu) ^ 0x80u) - 128;
}

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // One thread per packed u32 word (4 int8 elems); 2D-fold lifts the 65535 cap.
    let widx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (widx >= params.numel / 4u) {
        return;
    }
    let wa = t_a[widx];
    let wb = t_b[widx];
    var packed: u32 = 0u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let da = params.a_scale * f32(unpack_i8(wa, j) - params.a_zero_point);
        let db = params.b_scale * f32(unpack_i8(wb, j) - params.b_zero_point);
        let deq = da + params.alpha * db;
        var q = i32(round(deq * params.inv_output_scale)) + params.output_zero_point;
        q = clamp(q, -128, 127);
        packed = packed | ((bitcast<u32>(q) & 0xFFu) << (j * 8u));
    }
    t_out[widx] = packed;
}
)";

inline constexpr uint32_t kQ8taAddWorkgroupSizeX = 64;
inline constexpr uint32_t kQ8taAddWorkgroupSizeY = 1;
inline constexpr uint32_t kQ8taAddWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
