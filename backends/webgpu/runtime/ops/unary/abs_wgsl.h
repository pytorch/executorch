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

// @generated from abs.wgsl - DO NOT EDIT.
// wgsl-sha256: 39d3c163fdf6a92286828f4b3217e00294e3ca5634a878ed5fd34e3b1cdf0a27
inline constexpr const char* kAbsWGSL = R"(
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
    output[idx] = abs(x);
}
)";

inline constexpr uint32_t kAbsWorkgroupSizeX = 256;
inline constexpr uint32_t kAbsWorkgroupSizeY = 1;
inline constexpr uint32_t kAbsWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
