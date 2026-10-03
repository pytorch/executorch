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

// @generated from arange.wgsl - DO NOT EDIT.
// wgsl-sha256: 98b1e8ec11e51be25284f3ee578d97bcb405c9b680ac76b2981b54a7b79c0163
inline constexpr const char* kArangeWGSL = R"(
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
  start: f32,
  step: f32,
  _pad: u32,
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
    output[idx] = params.start + f32(idx) * params.step;
}
)";

inline constexpr uint32_t kArangeWorkgroupSizeX = 256;
inline constexpr uint32_t kArangeWorkgroupSizeY = 1;
inline constexpr uint32_t kArangeWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
