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

// @generated from upsample_nearest2d.wgsl - DO NOT EDIT.
// wgsl-sha256: 3eb5bb5604a34371e493ecc150f02995374ab183fed7e7c4a4e036b5b45563c0
inline constexpr const char* kUpsampleNearest2dWGSL = R"(
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;

struct Params {
  N: u32,
  C: u32,
  IH: u32,
  IW: u32,
  OH: u32,
  OW: u32,
  _p0: u32,
  _p1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill

// Nearest-neighbour 2D upsample, NCHW row-major, fp32. One thread per output
// element (n, c, oh, ow). oh -> floor(oh*IH/OH), matching real ATen's
// nearest_neighbor_compute_source_index (UpSample.h) — intentionally NOT
// mirroring Vulkan's upsample_2d.glsl, which uses the half-pixel-center
// formula correct for bilinear but not for non-exact nearest (see diff summary).
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

  let ih = (oh * params.IH) / params.OH;
  let iw = (ow * params.IW) / params.OW;
  let in_idx = ((n * params.C + c) * params.IH + ih) * params.IW + iw;
  out[i] = inp[in_idx];
}
)";

inline constexpr uint32_t kUpsampleNearest2dWorkgroupSizeX = 256;
inline constexpr uint32_t kUpsampleNearest2dWorkgroupSizeY = 1;
inline constexpr uint32_t kUpsampleNearest2dWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
