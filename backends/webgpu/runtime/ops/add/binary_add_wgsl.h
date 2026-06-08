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

// @generated from binary_add.wgsl - DO NOT EDIT.
// wgsl-sha256: c1ceec80c8d4d3d56986ad91ce0d7f9a57cd8467b8c3aa07a28da70e51d141d9
inline constexpr const char* kBinaryAddWGSL = R"(
@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
  alpha: f32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = input1[idx] + params.alpha * input2[idx];
}
)";

inline constexpr uint32_t kBinaryAddWorkgroupSizeX = 256;
inline constexpr uint32_t kBinaryAddWorkgroupSizeY = 1;
inline constexpr uint32_t kBinaryAddWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
