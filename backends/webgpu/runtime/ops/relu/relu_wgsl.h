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

// @generated from relu.wgsl - DO NOT EDIT.
// wgsl-sha256: 2f7fac19d55cb7e55749f2bc1856278c1f2afaf0bc7ff8e663fb9e14b4188199
inline constexpr const char* kReluWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = max(input[idx], 0.0);
}
)";

inline constexpr uint32_t kReluWorkgroupSizeX = 256;
inline constexpr uint32_t kReluWorkgroupSizeY = 1;
inline constexpr uint32_t kReluWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
