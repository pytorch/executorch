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

// @generated from group_norm.wgsl - DO NOT EDIT.
// wgsl-sha256: 911ba488b46495c887637a36d813cb81c639295881873c921c5939bdb1c397aa
inline constexpr const char* kGroupNormWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> mean: array<f32>;
@group(0) @binding(5) var<storage, read> rstd: array<f32>;

struct Params {
  n_channels: u32,
  hxw: u32,
  num_groups: u32,
  chans_per_group: u32,
  numel: u32,
  mean_numel: u32,
  group_size: u32,
  eps: f32,
}
@group(0) @binding(6) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // 2D-folded flat index (lifts the 65535 1D-dispatch cap for large numel).
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.numel) {
        return;
    }

    // NCHW channel -> its group; apply mean/rstd + per-channel affine.
    let n = idx / (params.n_channels * params.hxw);
    let c = (idx / params.hxw) % params.n_channels;
    let g = c / params.chans_per_group;
    let mg = n * params.num_groups + g;
    output[idx] = (input[idx] - mean[mg]) * rstd[mg] * weight[c] + bias[c];
}
)";

inline constexpr uint32_t kGroupNormWorkgroupSizeX = 64;
inline constexpr uint32_t kGroupNormWorkgroupSizeY = 1;
inline constexpr uint32_t kGroupNormWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
