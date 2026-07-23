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
// wgsl-sha256: 4b8ede66c5dbc9829ff48f745eb9ad48fa5a5200058baa532fbf34f78ec2f560
inline constexpr const char* kSiluMulFusedWGSL = R"(
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

// Fused SwiGLU activation: output = (g * sigmoid(g)) * up, folding the separate
// sigmoid(gate) -> mul(gate,sig)=silu -> mul(silu,up) triple into one dispatch.
// sigmoid + silu are computed in registers (never written to memory), so gate + up
// are read once and one output is written. The sigmoid form (1/(1+exp(-x))) and the
// multiply order match the original ops -> bit-exact.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
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
