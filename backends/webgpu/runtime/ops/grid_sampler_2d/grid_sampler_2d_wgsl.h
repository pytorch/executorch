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

// @generated from grid_sampler_2d.wgsl - DO NOT EDIT.
// wgsl-sha256: c69f408f77ce89d1ca8363e1ca7e2064be910c99d902a762bcbfcf1a2b059588
inline constexpr const char* kGridSampler2dWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> grid: array<f32>;

struct Params {
  in_h: u32,
  in_w: u32,
  out_h: u32,
  out_w: u32,
  channels: u32,
  numel: u32,
  pad0: u32,
  pad1: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

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

    // grid[N, out_h, out_w, 2] gives the normalized (x, y) sample coord; Vulkan
    // grid_sampler_2d config: bilinear, border padding, align_corners=true.
    let ow = oi % params.out_w;
    let oh = (oi / params.out_w) % params.out_h;
    let c = (oi / (params.out_w * params.out_h)) % params.channels;
    let n = oi / (params.out_w * params.out_h * params.channels);

    let gb = ((n * params.out_h + oh) * params.out_w + ow) * 2u;
    let gx_norm = grid[gb];
    let gy_norm = grid[gb + 1u];

    let maxx = f32(params.in_w - 1u);
    let maxy = f32(params.in_h - 1u);
    let gx = clamp((gx_norm + 1.0) * 0.5 * maxx, 0.0, maxx);
    let gy = clamp((gy_norm + 1.0) * 0.5 * maxy, 0.0, maxy);

    let mxi = i32(params.in_w) - 1;
    let myi = i32(params.in_h) - 1;
    let lx = i32(floor(gx));
    let ly = i32(floor(gy));
    let ux = clamp(lx + 1, 0, mxi);
    let uy = clamp(ly + 1, 0, myi);
    let wx = gx - f32(lx);
    let wy = gy - f32(ly);

    let base = (n * params.channels + c) * params.in_h * params.in_w;
    let s00 = input[base + u32(ly) * params.in_w + u32(lx)];
    let s10 = input[base + u32(ly) * params.in_w + u32(ux)];
    let s01 = input[base + u32(uy) * params.in_w + u32(lx)];
    let s11 = input[base + u32(uy) * params.in_w + u32(ux)];
    let top = s00 + wx * (s10 - s00);
    let bot = s01 + wx * (s11 - s01);
    output[oi] = top + wy * (bot - top);
}
)";

inline constexpr uint32_t kGridSampler2dWorkgroupSizeX = 64;
inline constexpr uint32_t kGridSampler2dWorkgroupSizeY = 1;
inline constexpr uint32_t kGridSampler2dWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
