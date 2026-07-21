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

// @generated from fill.wgsl - DO NOT EDIT.
// wgsl-sha256: 9acfd8079cf404b22856c157b339922911e15bb0e808aa76b5a7fb81a8505cf7
inline constexpr const char* kFillWGSL = R"(
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
  fill_value: f32,
}
@group(0) @binding(1) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = params.fill_value;
}
)";

inline constexpr uint32_t kFillWorkgroupSizeX = 256;
inline constexpr uint32_t kFillWorkgroupSizeY = 1;
inline constexpr uint32_t kFillWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
