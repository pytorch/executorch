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

// @generated from upsample_bilinear2d.wgsl - DO NOT EDIT.
// wgsl-sha256: fa8d89f2ec0e316650315ce39a426efec9ed34daef1fb5b7bc19ae9cea64565a
inline constexpr const char* kUpsampleBilinear2dWGSL = R"(
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;

struct Params {
  N: u32,
  C: u32,
  IH: u32,
  IW: u32,
  OH: u32,
  OW: u32,
  align_corners: u32,
  _p0: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill

// Bilinear NCHW fp32 upsample; src-index matches ATen upsample_bilinear2d.
fn src_index(dst: u32, insz: u32, outsz: u32, align: u32) -> f32 {
  if (align == 1u) {
    if (outsz <= 1u) {
      return 0.0;
    }
    return f32(dst) * f32(insz - 1u) / f32(outsz - 1u);
  }
  return (f32(dst) + 0.5) * f32(insz) / f32(outsz) - 0.5;
}

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.N * params.C * params.OH * params.OW;
  let i = gid.y * stride_x + gid.x;
  if (i >= total) {
    return;
  }
  let ow = i % params.OW;
  let oh = (i / params.OW) % params.OH;
  let c = (i / (params.OW * params.OH)) % params.C;
  let n = i / (params.OW * params.OH * params.C);

  let sh = src_index(oh, params.IH, params.OH, params.align_corners);
  let sw = src_index(ow, params.IW, params.OW, params.align_corners);

  let h0 = i32(floor(sh));
  let w0 = i32(floor(sw));
  let lh = sh - f32(h0);
  let lw = sw - f32(w0);

  let ih_max = i32(params.IH) - 1;
  let iw_max = i32(params.IW) - 1;
  let h0c = u32(clamp(h0, 0, ih_max));
  let h1c = u32(clamp(h0 + 1, 0, ih_max));
  let w0c = u32(clamp(w0, 0, iw_max));
  let w1c = u32(clamp(w0 + 1, 0, iw_max));

  let base = (n * params.C + c) * params.IH;
  let r0 = (base + h0c) * params.IW;
  let r1 = (base + h1c) * params.IW;
  let v00 = inp[r0 + w0c];
  let v01 = inp[r0 + w1c];
  let v10 = inp[r1 + w0c];
  let v11 = inp[r1 + w1c];

  let top = v00 + (v01 - v00) * lw;
  let bot = v10 + (v11 - v10) * lw;
  out[i] = top + (bot - top) * lh;
}
)";

inline constexpr uint32_t kUpsampleBilinear2dWorkgroupSizeX = 256;
inline constexpr uint32_t kUpsampleBilinear2dWorkgroupSizeY = 1;
inline constexpr uint32_t kUpsampleBilinear2dWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
