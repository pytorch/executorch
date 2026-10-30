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

// @generated from arange_int.wgsl - DO NOT EDIT.
// wgsl-sha256: 893b2f7aded63ce284f64fde49c58b5526b0329aef74c64fd97bee725afa7431
inline constexpr const char* kArangeIntWGSL = R"(
@group(0) @binding(0) var<storage, read_write> output: array<i32>;

struct Params {
  num_elements: u32,
  start: i32,
  step: i32,
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
    output[idx] = params.start + i32(idx) * params.step;
}
)";

inline constexpr uint32_t kArangeIntWorkgroupSizeX = 256;
inline constexpr uint32_t kArangeIntWorkgroupSizeY = 1;
inline constexpr uint32_t kArangeIntWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
