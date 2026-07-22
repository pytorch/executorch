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

// @generated from amax.wgsl - DO NOT EDIT.
// wgsl-sha256: 35fc059d7c72caa17f9cb1128823ecfd8f75be4ce24b6cd4f9629a97b52f64c0
inline constexpr const char* kAmaxWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_rows: u32,
  reduce_size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256u;

// Cooperative shared-memory reduction; mirrors Vulkan reduce.glsl (a group of
// threads co-operates per reduction row, partials aggregated in shared memory).
// Fixed upper bound (>= any clamped wg_size); only [0, wg_size) is used.
var<workgroup> partials: array<f32, 256>;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // One workgroup per reduction row; 2D-fold lifts the 65535 grid cap.
    let row = wid.x + wid.y * num_workgroups.x;
    if (row >= params.num_rows) {
        return;
    }
    let base = row * params.reduce_size;

    // Each thread reduces a strided slice of the row into a partial. Seed with
    // the row's first element (always valid; reduce_size >= 1) so threads that
    // own no element contribute a real value, not an out-of-range identity.
    var acc = input[base];
    var i = lid.x;
    while (i < params.reduce_size) {
        acc = max(acc, input[base + i]);
        i = i + wg_size;
    }
    partials[lid.x] = acc;
    workgroupBarrier();

    // Thread 0 aggregates the wg_size partials (mirrors Vulkan's group aggregate).
    if (lid.x == 0u) {
        var m = partials[0];
        for (var t = 1u; t < wg_size; t = t + 1u) {
            m = max(m, partials[t]);
        }
        output[row] = m;
    }
}
)";

inline constexpr uint32_t kAmaxWorkgroupSizeX = 256;
inline constexpr uint32_t kAmaxWorkgroupSizeY = 1;
inline constexpr uint32_t kAmaxWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
