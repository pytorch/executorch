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

// @generated from view_copy.wgsl - DO NOT EDIT.
// wgsl-sha256: 0613efa86b05e85df1dcbee59f90d3c98ac326ced877afd1a1f928e1bed978e7
inline constexpr const char* kViewCopyWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = input[idx];
}
)";

inline constexpr uint32_t kViewCopyWorkgroupSizeX = 64;
inline constexpr uint32_t kViewCopyWorkgroupSizeY = 1;
inline constexpr uint32_t kViewCopyWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
