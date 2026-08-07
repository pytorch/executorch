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

// @generated from to_copy_convert.wgsl - DO NOT EDIT.
// wgsl-sha256: 9506570f98b9888a65603157f75d380f7784539205610ccbcc459d6afb6629c5
inline constexpr const char* kToCopyIntToFloatWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = f32(input[idx]);
}
)";

inline constexpr uint32_t kToCopyIntToFloatWorkgroupSizeX = 64;
inline constexpr uint32_t kToCopyIntToFloatWorkgroupSizeY = 1;
inline constexpr uint32_t kToCopyIntToFloatWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
