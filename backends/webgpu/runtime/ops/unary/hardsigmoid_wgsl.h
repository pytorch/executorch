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
// wgsl-sha256: ac2cf23e81ffb740f3a52d21cd20c865e602038235e6b78f846c8d6253e096ff
inline constexpr const char* kHardsigmoidWGSL = R"(
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
    output[idx] = select(select((x + 3.0) / 6.0, 1.0, x >= 3.0), 0.0, x <= -3.0);
}
)";

inline constexpr uint32_t kHardsigmoidWorkgroupSizeX = 256;
inline constexpr uint32_t kHardsigmoidWorkgroupSizeY = 1;
inline constexpr uint32_t kHardsigmoidWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
