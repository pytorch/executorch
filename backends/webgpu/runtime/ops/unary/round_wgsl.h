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

// @generated from round.wgsl - DO NOT EDIT.
// wgsl-sha256: 8f3e0edbeb81aa50f35e691c78554e8057fa8d78fe8a86454f4f42e5e8871452
inline constexpr const char* kRoundWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
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
    let x = input[idx];
    output[idx] = round(x);
}
)";

inline constexpr uint32_t kRoundWorkgroupSizeX = 256;
inline constexpr uint32_t kRoundWorkgroupSizeY = 1;
inline constexpr uint32_t kRoundWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
