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

// @generated from quantize_per_tensor.wgsl - DO NOT EDIT.
// wgsl-sha256: 45111988635560213805aafd36e5235f4c0e75fba4c94c27c85cc981e06c0473
inline constexpr const char* kQuantizePerTensorWGSL = R"(
@group(0) @binding(0) var<storage, read> t_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> t_out: array<u32>;

struct Params {
  inv_scale: f32,
  zero_point: i32,
  numel: u32,
  pad0: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // One thread per packed u32 word (4 int8 elems); 2D-fold lifts the 65535 cap.
    let widx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (widx >= params.numel / 4u) {
        return;
    }
    var packed: u32 = 0u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        // Multiply by inv_scale, matching torch round(x * (1/scale)).
        var q = i32(round(t_in[widx * 4u + j] * params.inv_scale)) + params.zero_point;
        q = clamp(q, -128, 127);
        packed = packed | ((bitcast<u32>(q) & 0xFFu) << (j * 8u));
    }
    t_out[widx] = packed;
}
)";

inline constexpr uint32_t kQuantizePerTensorWorkgroupSizeX = 64;
inline constexpr uint32_t kQuantizePerTensorWorkgroupSizeY = 1;
inline constexpr uint32_t kQuantizePerTensorWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
