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

// @generated from clamp.wgsl - DO NOT EDIT.
// wgsl-sha256: 5499a682a06ed900a41c31de2c4e2d11db85051dc464315878e671beae662819
inline constexpr const char* kClampWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
  minimum: f32,
  maximum: f32,
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

inline constexpr uint32_t kClampWorkgroupSizeX = 256;
inline constexpr uint32_t kClampWorkgroupSizeY = 1;
inline constexpr uint32_t kClampWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
