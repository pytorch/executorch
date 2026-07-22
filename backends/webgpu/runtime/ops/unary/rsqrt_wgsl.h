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

// @generated from rsqrt.wgsl - DO NOT EDIT.
// wgsl-sha256: 108765d5a23b87473f34651875d08abf2a5fa8980bd92fc8cbe3617295097747
inline constexpr const char* kRsqrtWGSL = R"(
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
    output[idx] = 1.0 / sqrt(x);
}
)";

inline constexpr uint32_t kRsqrtWorkgroupSizeX = 256;
inline constexpr uint32_t kRsqrtWorkgroupSizeY = 1;
inline constexpr uint32_t kRsqrtWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
