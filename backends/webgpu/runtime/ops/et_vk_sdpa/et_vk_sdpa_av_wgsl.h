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

// @generated from et_vk_sdpa_av.wgsl - DO NOT EDIT.
// wgsl-sha256: 89cc190126bbf816145901db7a09451d0ef571a0bdccb023ee478a6c7305290d
inline constexpr const char* kEtVkSdpaAvWGSL = R"(
@group(0) @binding(0) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> sm: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;

struct Params {
  B: u32,
  Hq: u32,
  Hkv: u32,
  S_q: u32,
  S_kv: u32,
  D: u32,
  g: u32,
  tensor_layout: u32,
  bh_lo: u32,
  bh_count: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64;

fn v_row4(b: u32, h: u32, s: u32, d4_count: u32) -> u32 {
  if (params.tensor_layout == 0u) {
    return ((b * params.Hkv + h) * params.S_kv + s) * d4_count;
  }
  return ((b * params.S_kv + s) * params.Hkv + h) * d4_count;
}

fn out_row4(b: u32, h: u32, s: u32, d4_count: u32) -> u32 {
  if (params.tensor_layout == 0u) {
    return ((b * params.Hq + h) * params.S_q + s) * d4_count;
  }
  return ((b * params.S_q + s) * params.Hq + h) * d4_count;
}

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>) {
  let d4_count = params.D / 4u;
  let chunk_total = params.bh_count * params.S_q * d4_count;
  let tid = gid.x + gid.y * (nwg.x * wg_size);
  if (tid >= chunk_total) {
    return;
  }
  let i = params.bh_lo * params.S_q * d4_count + tid;
  let d4 = i % d4_count;
  let s = (i / d4_count) % params.S_q;
  let h = (i / (d4_count * params.S_q)) % params.Hq;
  let b = i / (d4_count * params.S_q * params.Hq);
  let kv_h = h / params.g;
  let smbase =
      ((b * params.Hq + h - params.bh_lo) * params.S_q + s) * params.S_kv;

  var acc: vec4<f32> = vec4<f32>(0.0);
  for (var c: u32 = 0u; c < params.S_kv; c = c + 1u) {
    acc = acc +
        sm[smbase + c] * v[v_row4(b, kv_h, c, d4_count) + d4];
  }
  out[out_row4(b, h, s, d4_count) + d4] = acc;
}
)";

inline constexpr uint32_t kEtVkSdpaAvWorkgroupSizeX = 64;
inline constexpr uint32_t kEtVkSdpaAvWorkgroupSizeY = 1;
inline constexpr uint32_t kEtVkSdpaAvWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
