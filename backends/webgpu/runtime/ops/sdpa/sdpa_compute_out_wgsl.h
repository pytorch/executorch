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
// wgsl-sha256: 545f624567b08eba407954034df821010e49124fa6f8fd6b05c64ca4354ee4cc
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

const TM: u32 = 4u;
const TN: u32 = 4u;

fn load_a_vec4(s: u32, h: u32, c4: u32) -> vec4<f32> {
  var r = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  if (s >= params.S) {
    return r;
  }
  let base = h * params.S * params.context_len + s * params.context_len;
  if (c4 + 0u < params.context_len) { r.x = t_attn_weights_softmax[base + c4 + 0u]; }
  if (c4 + 1u < params.context_len) { r.y = t_attn_weights_softmax[base + c4 + 1u]; }
  if (c4 + 2u < params.context_len) { r.z = t_attn_weights_softmax[base + c4 + 2u]; }
  if (c4 + 3u < params.context_len) { r.w = t_attn_weights_softmax[base + c4 + 3u]; }
  return r;
}

fn load_v_d4(c: u32, kvh: u32, d0: u32) -> vec4<f32> {
  var r = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  if (c >= params.context_len) {
    return r;
  }
  let base = c * params.Hkv * params.D + kvh * params.D + d0;
  if (d0 + 0u < params.D) { r.x = t_v_cache[base + 0u]; }
  if (d0 + 1u < params.D) { r.y = t_v_cache[base + 1u]; }
  if (d0 + 2u < params.D) { r.z = t_v_cache[base + 2u]; }
  if (d0 + 3u < params.D) { r.w = t_v_cache[base + 3u]; }
  return r;
}

fn store_out(s: u32, d: u32, h: u32, val: f32) {
  if (s >= params.S || d >= params.D) {
    return;
  }
  let idx = s * params.Hq * params.D + h * params.D + d;
  t_out[idx] = val;
}

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nrt = (params.S + TM - 1u) / TM;
  let nct = (params.D + TN - 1u) / TN;
  let tiles = nrt * nct;
  let total = tiles * params.Hq;
  if (gid.x >= total) {
    return;
  }

  let h = gid.x / tiles;
  let rem = gid.x % tiles;
  let row_tile = rem / nct;
  let col_tile = rem % nct;
  let kvh = h / params.g;
  let s0 = row_tile * TM;
  let d0 = col_tile * TN;

  var acc: array<vec4<f32>, 4>;
  acc[0] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  acc[1] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  acc[2] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  acc[3] = vec4<f32>(0.0, 0.0, 0.0, 0.0);

  var c4: u32 = 0u;
  loop {
    if (c4 >= params.context_len) {
      break;
    }
    let a0 = load_a_vec4(s0 + 0u, h, c4);
    let a1 = load_a_vec4(s0 + 1u, h, c4);
    let a2 = load_a_vec4(s0 + 2u, h, c4);
    let a3 = load_a_vec4(s0 + 3u, h, c4);
    let v0 = load_v_d4(c4 + 0u, kvh, d0);
    let v1 = load_v_d4(c4 + 1u, kvh, d0);
    let v2 = load_v_d4(c4 + 2u, kvh, d0);
    let v3 = load_v_d4(c4 + 3u, kvh, d0);
    acc[0] += a0.x * v0 + a0.y * v1 + a0.z * v2 + a0.w * v3;
    acc[1] += a1.x * v0 + a1.y * v1 + a1.z * v2 + a1.w * v3;
    acc[2] += a2.x * v0 + a2.y * v1 + a2.z * v2 + a2.w * v3;
    acc[3] += a3.x * v0 + a3.y * v1 + a3.z * v2 + a3.w * v3;
    c4 = c4 + 4u;
  }

  var m: u32 = 0u;
  loop {
    if (m >= TM) {
      break;
    }
    let ov = acc[m];
    store_out(s0 + m, d0 + 0u, h, ov.x);
    store_out(s0 + m, d0 + 1u, h, ov.y);
    store_out(s0 + m, d0 + 2u, h, ov.z);
    store_out(s0 + m, d0 + 3u, h, ov.w);
    m = m + 1u;
  }
}
)";

inline constexpr uint32_t kSdpaComputeOutWorkgroupSizeX = 64;
inline constexpr uint32_t kSdpaComputeOutWorkgroupSizeY = 1;
inline constexpr uint32_t kSdpaComputeOutWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
