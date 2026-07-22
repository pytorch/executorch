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

// @generated from amin.wgsl - DO NOT EDIT.
// wgsl-sha256: 974a28fd80f089c8a52bf54d73f3bd03c195b2f9d7904bb4c31e6545813e0459
inline constexpr const char* kAminWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_rows: u32,
  reduce_size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let row = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (row >= params.num_rows) {
        return;
    }
    let base = row * params.reduce_size;
    var acc = input[base];
    for (var j = 1u; j < params.reduce_size; j = j + 1u) {
        acc = min(acc, input[base + j]);
    }
    output[row] = acc;
}
)";

inline constexpr uint32_t kAminWorkgroupSizeX = 256;
inline constexpr uint32_t kAminWorkgroupSizeY = 1;
inline constexpr uint32_t kAminWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
