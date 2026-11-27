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

// @generated from et_vk_sdpa_qk_entry.wgsl - DO NOT EDIT.
// wgsl-sha256: 95ff9f19757d23c41d82054b779c02fb8d83531eef3e8cda5f3a78dde42a4e58
inline constexpr const char* kEtVkSdpaQkEntryWGSL = R"(
@group(0) @binding(0) var<storage, read_write> attn: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;

struct Params {
  B: u32,
  H: u32,
  S_q: u32,
  S_kv: u32,
  D: u32,
  has_mask: u32,
  _pad0: u32,
  scale: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64;

// Non-causal fused SDPA, QK phase. DSHB layout, row-major: q [B, H, S_q, D],
// k [B, H, S_kv, D]. ONE thread per ENTRY (b,h,s,c) of attn_weights
// [B, H, S_q, S_kv] = one D-length dot. Previously this was one thread per ROW
// (looping c) — fine for window/self attention but catastrophic for DaViT
// CHANNEL attention where S_q = head_dim (~32), so B*H*S_q was 128/256/512/1024
// → only 2/4/8/16 workgroups serial over the huge spatial D (the (2,1,1)@103ms
// dispatch). Parallelizing over all B*H*S_q*S_kv entries (2D-folded past the
// 65535 ceiling, mirroring the softmax phase) gives S_kv× more threads.
@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>) {
  let aw_numel = params.B * params.H * params.S_q * params.S_kv;
  let idx = gid.x + gid.y * (nwg.x * wg_size); // 2D-folded linear entry id
  if (idx >= aw_numel) {
    return;
  }
  let c = idx % params.S_kv;
  let row = idx / params.S_kv; // (b,h,s) flattened
  let s = row % params.S_q;
  let h = (row / params.S_q) % params.H;
  let b = row / (params.S_q * params.H);

  let qbase = ((b * params.H + h) * params.S_q + s) * params.D;
  let kbase = ((b * params.H + h) * params.S_kv + c) * params.D;
  var acc: f32 = 0.0;
  for (var d: u32 = 0u; d < params.D; d = d + 1u) {
    acc = acc + q[qbase + d] * k[kbase + d];
  }
  acc = acc * params.scale;
  if (params.has_mask != 0u) {
    acc = acc + mask[idx];
  }
  attn[idx] = acc; // attn is [B,H,S_q,S_kv] row-major -> index == idx
}
)";

inline constexpr uint32_t kEtVkSdpaQkEntryWorkgroupSizeX = 64;
inline constexpr uint32_t kEtVkSdpaQkEntryWorkgroupSizeY = 1;
inline constexpr uint32_t kEtVkSdpaQkEntryWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
