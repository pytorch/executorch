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

// @generated from pow_scalar.wgsl - DO NOT EDIT.
// wgsl-sha256: a9176c6a6b0da421cd3649e396390cd10859c1256f6d38fce6bc730d3e3788e6
inline constexpr const char* kPowScalarWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
  minimum: f32,
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
    output[idx] = pow(input[idx], params.minimum);
}
)";

inline constexpr uint32_t kPowScalarWorkgroupSizeX = 256;
inline constexpr uint32_t kPowScalarWorkgroupSizeY = 1;
inline constexpr uint32_t kPowScalarWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
