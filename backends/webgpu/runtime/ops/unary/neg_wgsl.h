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

// @generated from neg.wgsl - DO NOT EDIT.
// wgsl-sha256: 8851b9f42d14153f6f04484fee2f8bf67bda26dea892ff48768e09e6ad49cee1
inline constexpr const char* kNegWGSL = R"(
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
    output[idx] = -x;
}
)";

inline constexpr uint32_t kNegWorkgroupSizeX = 256;
inline constexpr uint32_t kNegWorkgroupSizeY = 1;
inline constexpr uint32_t kNegWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
