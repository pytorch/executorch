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

// @generated from sdpa_compute_out.wgsl - DO NOT EDIT.
// wgsl-sha256: 67b9c64fbffdcb72264dda42e24b59e414719411c64c504f84f2ba57b5dcfc0f
inline constexpr const char* kSdpaComputeOutWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_attn_weights_softmax: array<f32>;
@group(0) @binding(2) var<storage, read> t_v_cache: array<f32>;

struct Params {
  S: u32,
  Hq: u32,
  Hkv: u32,
  D: u32,
  context_len: u32,
  g: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.S * params.Hq * params.D;
  let idx = gid.x;
  if (idx >= total) {
    return;
  }
  let d = idx % params.D;
  let h = (idx / params.D) % params.Hq;
  let s = idx / (params.D * params.Hq);

  let kvh = h / params.g;

  let aw_base = h * params.S * params.context_len + s * params.context_len;

  var acc: f32 = 0.0;
  var c: u32 = 0u;
  loop {
    if (c >= params.context_len) {
      break;
    }
    let v_off = c * params.Hkv * params.D + kvh * params.D + d;
    acc = acc + t_attn_weights_softmax[aw_base + c] * t_v_cache[v_off];
    c = c + 1u;
  }

  t_out[idx] = acc;
}
)";

inline constexpr uint32_t kSdpaComputeOutWorkgroupSizeX = 64;
inline constexpr uint32_t kSdpaComputeOutWorkgroupSizeY = 1;
inline constexpr uint32_t kSdpaComputeOutWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
