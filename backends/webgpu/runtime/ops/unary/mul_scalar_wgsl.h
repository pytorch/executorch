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

// @generated from mul_scalar.wgsl - DO NOT EDIT.
// wgsl-sha256: 31c39096da58d01129a1c09c499c1a39aa0b06e61b47b5ddb9fba2e66d38968e
inline constexpr const char* kMulScalarWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
  scalar: f32,
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
    output[idx] = input[idx] * params.scalar;
}
)";

inline constexpr uint32_t kMulScalarWorkgroupSizeX = 256;
inline constexpr uint32_t kMulScalarWorkgroupSizeY = 1;
inline constexpr uint32_t kMulScalarWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
