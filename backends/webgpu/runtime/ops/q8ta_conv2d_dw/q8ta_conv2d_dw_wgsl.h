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

// @generated from q8ta_conv2d_dw.wgsl - DO NOT EDIT.
// wgsl-sha256: daabc65120a540a43e0199d8c0a61cf510db056854013693cea39869c78359f3
inline constexpr const char* kQ8taConv2dDwWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_out: array<u32>;
@group(0) @binding(1) var<storage, read> t_x: array<u32>;
@group(0) @binding(2) var<storage, read> t_weight: array<u32>;
@group(0) @binding(3) var<storage, read> t_scales: array<f32>;
@group(0) @binding(4) var<storage, read> t_bias: array<f32>;

struct Params {
  N: u32,
  C: u32,
  H_in: u32,
  W_in: u32,
  H_out: u32,
  W_out: u32,
  Kh: u32,
  Kw: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  dil_h: u32,
  dil_w: u32,
  input_zero_point: i32,
  output_zero_point: i32,
  input_scale: f32,
  inv_output_scale: f32,
  has_bias: u32,
  pad0: u32,
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
  // One thread per output word = 4 W-positions of fixed (n,c,oh); W_out%4==0.
  let widx = gid.x + gid.y * (num_workgroups.x * wg_size);
  let words = (params.N * params.C * params.H_out * params.W_out) / 4u;
  if (widx >= words) {
    return;
  }
  let flat0 = widx * 4u;
  let ow0 = flat0 % params.W_out;
  var r = flat0 / params.W_out;
  let oh = r % params.H_out;
  r = r / params.H_out;
  let c = r % params.C;
  let n = r / params.C;

  var acc: array<i32, 4>;
  for (var j: u32 = 0u; j < 4u; j = j + 1u) {
    acc[j] = 0;
  }
  // Depthwise per-channel [Kh,Kw] window; weight laid out [Kh,Kw,OC].
  for (var kh: u32 = 0u; kh < params.Kh; kh = kh + 1u) {
    let ih = i32(oh) * i32(params.stride_h) - i32(params.pad_h)
        + i32(kh) * i32(params.dil_h);
    if (ih < 0 || ih >= i32(params.H_in)) {
      continue;
    }
    for (var kw: u32 = 0u; kw < params.Kw; kw = kw + 1u) {
      let wbi = (kh * params.Kw + kw) * params.C + c;
      let wv = unpack_i8(wbi, t_weight[wbi >> 2u]);
      let in_row = (n * params.C + c) * params.H_in + u32(ih);
      for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let iw = i32(ow0 + j) * i32(params.stride_w) - i32(params.pad_w)
            + i32(kw) * i32(params.dil_w);
        if (iw < 0 || iw >= i32(params.W_in)) {
          continue;
        }
        let xbi = in_row * params.W_in + u32(iw);
        acc[j] = acc[j] +
            (unpack_i8(xbi, t_x[xbi >> 2u]) - params.input_zero_point) * wv;
      }
    }
  }

  var packed: u32 = 0u;
  for (var j: u32 = 0u; j < 4u; j = j + 1u) {
    var v = f32(acc[j]) * params.input_scale * t_scales[c];
    if (params.has_bias != 0u) {
      v = v + t_bias[c];
    }
    var q = i32(round(v * params.inv_output_scale)) + params.output_zero_point;
    q = clamp(q, -128, 127);
    packed = packed | ((bitcast<u32>(q) & 0xFFu) << (j * 8u));
  }
  t_out[widx] = packed;
}
)";

inline constexpr uint32_t kQ8taConv2dDwWorkgroupSizeX = 64;
inline constexpr uint32_t kQ8taConv2dDwWorkgroupSizeY = 1;
inline constexpr uint32_t kQ8taConv2dDwWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
