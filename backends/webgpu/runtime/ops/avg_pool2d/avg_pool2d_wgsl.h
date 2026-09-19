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

// @generated from avg_pool2d.wgsl - DO NOT EDIT.
// wgsl-sha256: 39d22da5d8dd6975bd3d98450ec2db052fde77e89805557cde0d86d9da2530e3
inline constexpr const char* kAvgPool2dWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  kh: u32,
  kw: u32,
  sh: u32,
  sw: u32,
  ph: u32,
  pw: u32,
  in_h: u32,
  in_w: u32,
  out_h: u32,
  out_w: u32,
  channels: u32,
  numel: u32,
  divisor_override: i32,
  count_include_pad: u32,
  has_divisor_override: u32,
  pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

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

    // Unravel NCHW out index; average the clipped input window (Vulkan glsl).
    let ow = oi % params.out_w;
    let oh = (oi / params.out_w) % params.out_h;
    let c = (oi / (params.out_w * params.out_h)) % params.channels;
    let n = oi / (params.out_w * params.out_h * params.channels);

    let iph = i32(oh) * i32(params.sh) - i32(params.ph);
    let ipw = i32(ow) * i32(params.sw) - i32(params.pw);
    let sh0 = max(0, iph);
    let eh = min(iph + i32(params.kh), i32(params.in_h));
    let sw0 = max(0, ipw);
    let ew = min(ipw + i32(params.kw), i32(params.in_w));

    let cbase = (n * params.channels + c) * params.in_h * params.in_w;
    var acc = 0.0;
    for (var ih = sh0; ih < eh; ih = ih + 1) {
        for (var iw = sw0; iw < ew; iw = iw + 1) {
            acc = acc + input[cbase + u32(ih) * params.in_w + u32(iw)];
        }
    }

    var divv: i32;
    if (params.has_divisor_override != 0u) {
        divv = params.divisor_override;
    } else if (params.count_include_pad != 0u) {
        // Cells the window extends past the padded input's right/bottom edge.
        let beh = iph + i32(params.kh) - i32(params.ph) - i32(params.in_h);
        let bew = ipw + i32(params.kw) - i32(params.pw) - i32(params.in_w);
        divv = (i32(params.kh) - max(beh, 0)) * (i32(params.kw) - max(bew, 0));
    } else {
        divv = (eh - sh0) * (ew - sw0);
    }
    // Empty window (fully in padding): no cells summed, so avoid 0/0.
    if (params.has_divisor_override == 0u && divv <= 0) {
        output[oi] = 0.0;
        return;
    }
    output[oi] = acc / f32(divv);
}
)";

inline constexpr uint32_t kAvgPool2dWorkgroupSizeX = 64;
inline constexpr uint32_t kAvgPool2dWorkgroupSizeY = 1;
inline constexpr uint32_t kAvgPool2dWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
