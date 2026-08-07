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

// @generated from to_copy_convert.wgsl - DO NOT EDIT.
// wgsl-sha256: 241d51293095623126da4f106092bd2d7327c26e00c9cad39bdb8dc546f48149
inline constexpr const char* kToCopyFloatToIntWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = i32(input[idx]);
}
)";

inline constexpr uint32_t kToCopyFloatToIntWorkgroupSizeX = 64;
inline constexpr uint32_t kToCopyFloatToIntWorkgroupSizeY = 1;
inline constexpr uint32_t kToCopyFloatToIntWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
