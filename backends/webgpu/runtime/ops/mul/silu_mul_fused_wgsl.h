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

// @generated from silu_mul_fused.wgsl - DO NOT EDIT.
// wgsl-sha256: 7ba46c3ec15bfe4ab77a6a3e8e9f81dcb53a82328da953a3a9252fc7d470f461
inline constexpr const char* kSiluMulFusedWGSL = R"(
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.num_elements) {
        return;
    }
    let g = gate[idx];
    let sig = 1.0 / (1.0 + exp(-g));
    output[idx] = (g * sig) * up[idx];
}
)";

inline constexpr uint32_t kSiluMulFusedWorkgroupSizeX = 64;
inline constexpr uint32_t kSiluMulFusedWorkgroupSizeY = 1;
inline constexpr uint32_t kSiluMulFusedWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
