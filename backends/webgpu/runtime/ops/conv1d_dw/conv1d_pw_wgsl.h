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

// @generated from conv1d_pw.wgsl - DO NOT EDIT.
// wgsl-sha256: d47893e191276cf99ee6fd30002ff70f4a0656bcc8e8f90eaca95d9916b54448
inline constexpr const char* kConv1dPwWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  in_channels: u32,
  out_channels: u32,
  length: u32,
  numel: u32,
  has_bias: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // 2D-folded flat index (lifts the 65535 1D-dispatch cap for large numel).
    let oi = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (oi >= params.numel) {
        return;
    }

    // Pointwise (K=1): out[n,oc,l] = sum_ic weight[oc,ic] * input[n,ic,l].
    let l = oi % params.length;
    let oc = (oi / params.length) % params.out_channels;
    let n = oi / (params.length * params.out_channels);

    let w_base = oc * params.in_channels;
    let in_base = n * params.in_channels * params.length + l;
    var s = 0.0;
    for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
        s = s + weight[w_base + ic] * input[in_base + ic * params.length];
    }
    if (params.has_bias != 0u) {
        s = s + bias[oc];
    }
    output[oi] = s;
}
)";

inline constexpr uint32_t kConv1dPwWorkgroupSizeX = 64;
inline constexpr uint32_t kConv1dPwWorkgroupSizeY = 1;
inline constexpr uint32_t kConv1dPwWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
