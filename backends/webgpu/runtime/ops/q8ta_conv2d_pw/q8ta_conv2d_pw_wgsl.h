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

// @generated from q8ta_conv2d_pw.wgsl - DO NOT EDIT.
// wgsl-sha256: b4d6ddfd2e0101bbff9a668895014c8822aa7d92147a982c1e6fe2c3bff847d0
inline constexpr const char* kQ8taConv2dPwWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_out: array<u32>;
@group(0) @binding(1) var<storage, read> t_x: array<u32>;
@group(0) @binding(2) var<storage, read> t_weight: array<u32>;
@group(0) @binding(3) var<storage, read> t_scales: array<f32>;
@group(0) @binding(4) var<storage, read> t_bias: array<f32>;

struct Params {
  N: u32,
  OC: u32,
  IC: u32,
  H: u32,
  W: u32,
  input_zero_point: i32,
  output_zero_point: i32,
  input_scale: f32,
  inv_output_scale: f32,
  has_bias: u32,
  pad0: u32,
  pad1: u32,
}
@group(0) @binding(5) var<uniform> params: Params;

override wg_size: u32 = 64u;

fn unpack_i8(bi: u32, word: u32) -> i32 {
  return i32(((word >> ((bi & 3u) * 8u)) & 0xFFu) ^ 0x80u) - 128;
}

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // One thread per output word = 4 W-positions of fixed (n,oc,h); W%4==0.
  let widx = gid.x + gid.y * (num_workgroups.x * wg_size);
  let words = (params.N * params.OC * params.H * params.W) / 4u;
  if (widx >= words) {
    return;
  }
  let flat0 = widx * 4u;
  let w0 = flat0 % params.W;
  var r = flat0 / params.W;
  let h = r % params.H;
  r = r / params.H;
  let oc = r % params.OC;
  let n = r / params.OC;

  var acc: array<i32, 4>;
  for (var j: u32 = 0u; j < 4u; j = j + 1u) {
    acc[j] = 0;
  }
  // 1x1 conv = per-position channel dot over input channels.
  for (var ic: u32 = 0u; ic < params.IC; ic = ic + 1u) {
    let wbi = oc * params.IC + ic;
    let wv = unpack_i8(wbi, t_weight[wbi >> 2u]);
    let base = ((n * params.IC + ic) * params.H + h) * params.W + w0;
    // W%4==0 and w0%4==0 => base%4==0, so all 4 j share one input word.
    let xword = t_x[base >> 2u];
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
      let xbi = base + j;
      acc[j] = acc[j] +
          (unpack_i8(xbi, xword) - params.input_zero_point) * wv;
    }
  }

  let wscale = t_scales[oc];
  let bias = t_bias[oc];
  var packed: u32 = 0u;
  for (var j: u32 = 0u; j < 4u; j = j + 1u) {
    var v = f32(acc[j]) * params.input_scale * wscale;
    if (params.has_bias != 0u) {
      v = v + bias;
    }
    var q = i32(round(v * params.inv_output_scale)) + params.output_zero_point;
    q = clamp(q, -128, 127);
    packed = packed | ((bitcast<u32>(q) & 0xFFu) << (j * 8u));
  }
  t_out[widx] = packed;
}
)";

inline constexpr uint32_t kQ8taConv2dPwWorkgroupSizeX = 64;
inline constexpr uint32_t kQ8taConv2dPwWorkgroupSizeY = 1;
inline constexpr uint32_t kQ8taConv2dPwWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
