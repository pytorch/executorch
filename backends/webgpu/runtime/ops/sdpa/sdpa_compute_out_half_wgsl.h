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
// wgsl-sha256: ed9709c966538edf2cbc6be97c284b89a9d921b6a4dbf115c6cbd76af301a1be
inline constexpr const char* kSdpaComputeOutHalfWGSL = R"(
enable f16;
@group(0) @binding(0) var<storage, read_write> t_out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> t_attn_weights_softmax: array<f32>;
@group(0) @binding(2) var<storage, read> t_v_cache: array<vec4<f16>>;

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

// Checked loaders mask context lanes past context_len (D%4==0, host-guarded).
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
  if (c >= params.context_len) {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }
  let base = c * params.Hkv * params.D + kvh * params.D + d0;
  return vec4<f32>(t_v_cache[base / 4u]);
}

// Branch-free loaders for the aligned body: caller guarantees c4..c4+3 < context_len.
fn load_a_vec4_nc(s: u32, h: u32, c4: u32) -> vec4<f32> {
  if (s >= params.S) {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }
  let base = h * params.S * params.context_len + s * params.context_len + c4;
  return vec4<f32>(t_attn_weights_softmax[base], t_attn_weights_softmax[base + 1u], t_attn_weights_softmax[base + 2u], t_attn_weights_softmax[base + 3u]);
}

fn load_v_d4_nc(c: u32, kvh: u32, d0: u32) -> vec4<f32> {
  let base = c * params.Hkv * params.D + kvh * params.D + d0;
  return vec4<f32>(t_v_cache[base / 4u]);
}

fn store_out_vec4(s: u32, d0: u32, h: u32, val: vec4<f32>) {
  if (s >= params.S) {
    return;
  }
  let idx = s * params.Hq * params.D + h * params.D + d0;
  t_out[idx / 4u] = val;
}

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let nrt = (params.S + TM - 1u) / TM;
  let nct = (params.D + TN - 1u) / TN;
  let tiles = nrt * nct;
  let total = tiles * params.Hq;
  // 2D dispatch fold: recover the linear tile index across x/y.
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= total) {
    return;
  }

  let h = idx / tiles;
  let rem = idx % tiles;
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

  // Branch-free aligned body + checked tail; mirrors Vulkan out_tiled.glsl.
  let ctx_aligned = params.context_len - (params.context_len & 3u);
  var c4: u32 = 0u;
  loop {
    if (c4 >= ctx_aligned) {
      break;
    }
    let a0 = load_a_vec4_nc(s0 + 0u, h, c4);
    let a1 = load_a_vec4_nc(s0 + 1u, h, c4);
    let a2 = load_a_vec4_nc(s0 + 2u, h, c4);
    let a3 = load_a_vec4_nc(s0 + 3u, h, c4);
    let v0 = load_v_d4_nc(c4 + 0u, kvh, d0);
    let v1 = load_v_d4_nc(c4 + 1u, kvh, d0);
    let v2 = load_v_d4_nc(c4 + 2u, kvh, d0);
    let v3 = load_v_d4_nc(c4 + 3u, kvh, d0);
    acc[0] += a0.x * v0 + a0.y * v1 + a0.z * v2 + a0.w * v3;
    acc[1] += a1.x * v0 + a1.y * v1 + a1.z * v2 + a1.w * v3;
    acc[2] += a2.x * v0 + a2.y * v1 + a2.z * v2 + a2.w * v3;
    acc[3] += a3.x * v0 + a3.y * v1 + a3.z * v2 + a3.w * v3;
    c4 = c4 + 4u;
  }
  if (c4 < params.context_len) {
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
  }

  var m: u32 = 0u;
  loop {
    if (m >= TM) {
      break;
    }
    store_out_vec4(s0 + m, d0, h, acc[m]);
    m = m + 1u;
  }
}
)";

inline constexpr uint32_t kSdpaComputeOutHalfWorkgroupSizeX = 64;
inline constexpr uint32_t kSdpaComputeOutHalfWorkgroupSizeY = 1;
inline constexpr uint32_t kSdpaComputeOutHalfWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
