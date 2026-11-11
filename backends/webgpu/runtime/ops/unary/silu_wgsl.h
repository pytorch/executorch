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

// @generated from unary.wgsl - DO NOT EDIT.
// wgsl-sha256: 7059eb8b9e92780c9e9c40a6e0d3ffe70aa1f5d28a340e66387c97997ad7eaa3
inline constexpr const char* kSiluWGSL = R"(
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
    output[idx] = x * select(1.0, exp(-abs(x)), x < 0.0) / (1.0 + exp(-abs(x)));
}
)";

inline constexpr uint32_t kSiluWorkgroupSizeX = 256;
inline constexpr uint32_t kSiluWorkgroupSizeY = 1;
inline constexpr uint32_t kSiluWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
