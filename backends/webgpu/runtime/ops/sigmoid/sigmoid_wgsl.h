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

// @generated from sigmoid.wgsl - DO NOT EDIT.
// wgsl-sha256: 79a1554e9d957e10c0f4379b5e0240191743952198e9ed9c0342402bacd356de
inline constexpr const char* kSigmoidWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}
)";

inline constexpr uint32_t kSigmoidWorkgroupSizeX = 256;
inline constexpr uint32_t kSigmoidWorkgroupSizeY = 1;
inline constexpr uint32_t kSigmoidWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
