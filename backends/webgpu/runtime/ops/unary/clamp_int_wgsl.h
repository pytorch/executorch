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

// @generated from clamp_int.wgsl - DO NOT EDIT.
// wgsl-sha256: 480340b676271dd5218227a5977ca745829371ff2a89c1dbbf0866ec5145c053
inline constexpr const char* kClampIntWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

struct Params {
  num_elements: u32,
  minimum: i32,
  maximum: i32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = clamp(input[idx], params.minimum, params.maximum);
}
)";

inline constexpr uint32_t kClampIntWorkgroupSizeX = 256;
inline constexpr uint32_t kClampIntWorkgroupSizeY = 1;
inline constexpr uint32_t kClampIntWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
