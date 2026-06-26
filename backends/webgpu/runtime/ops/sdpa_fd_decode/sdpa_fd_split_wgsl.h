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

// @generated from sdpa_fd_split.wgsl - DO NOT EDIT.
// wgsl-sha256: f8a392ab021e4f1453abc2dc615254985d8045bfa187fc6a1d8fab722276ec17
inline constexpr const char* kSdpaFdSplitWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_part_o: array<f32>;
@group(0) @binding(1) var<storage, read_write> t_part_ml: array<f32>;
@group(0) @binding(2) var<storage, read> t_q: array<f32>;
@group(0) @binding(3) var<storage, read> t_k_cache: array<f32>;
@group(0) @binding(4) var<storage, read> t_v_cache: array<f32>;

struct Params {
  Hq: u32,
  Hkv: u32,
  D: u32,
  context_len: u32,
  g: u32,
  num_splits: u32,
  split_len: u32,
  scale: f32,
}
@group(0) @binding(5) var<uniform> params: Params;

const WG_SIZE: u32 = 64u;
const MAX_SPLITS: u32 = 128u;
const MAX_D_PER_LANE: u32 = 2u;
const NEG_INF: f32 = -1.0e30;

// sh_s: block scores then softmax weights; sh_red: max/sum reduction scratch.
var<workgroup> sh_s: array<f32, WG_SIZE>;
var<workgroup> sh_red: array<f32, WG_SIZE>;

// FlashDecoding pass 1: per-(head,split) unnormalized softmax partial.
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x / params.num_splits;
  let split_i = wid.x % params.num_splits;
  let t = lid.x;
  let D = params.D;
  let D4 = D / 4u; // D is a multiple of 4 (guarded host-side); vec4 QK dot
  let ctx = params.context_len;
  let kv = h / params.g;
  let q_base = h * D;
  let kv_row_stride = params.Hkv * D;

  let c0 = split_i * params.split_len;
  var c1 = c0 + params.split_len;
  if (c1 > ctx) { c1 = ctx; }

  var m: f32 = NEG_INF;
  var l: f32 = 0.0;
  var o_acc: array<f32, MAX_D_PER_LANE>;
  for (var nd: u32 = 0u; nd < MAX_D_PER_LANE; nd = nd + 1u) { o_acc[nd] = 0.0; }

  // Stream the split in blocks of WG_SIZE KV positions.
  var block: u32 = c0;
  loop {
    if (block >= c1) { break; }
    var n: u32 = c1 - block;
    if (n > WG_SIZE) { n = WG_SIZE; }

    // Phase 1: lane t computes the full QK dot for position block+t (vec4), one
    // K row read once. Out-of-block lanes hold NEG_INF (safe for the max).
    var s: f32 = NEG_INF;
    if (t < n) {
      let kvbase = (block + t) * kv_row_stride + kv * D;
      var acc4 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
      for (var i4: u32 = 0u; i4 < D4; i4 = i4 + 1u) {
        let qi = q_base + i4 * 4u;
        let ki = kvbase + i4 * 4u;
        let qv = vec4<f32>(t_q[qi], t_q[qi + 1u], t_q[qi + 2u], t_q[qi + 3u]);
        let kvv = vec4<f32>(
            t_k_cache[ki], t_k_cache[ki + 1u],
            t_k_cache[ki + 2u], t_k_cache[ki + 3u]);
        acc4 = acc4 + qv * kvv;
      }
      s = (acc4.x + acc4.y + acc4.z + acc4.w) * params.scale;
    }
    sh_s[t] = s;

    // Phase 2a: block max via tree reduction (sh_red written from register s).
    sh_red[t] = s;
    workgroupBarrier();
    for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
      if (t < stride) { sh_red[t] = max(sh_red[t], sh_red[t + stride]); }
      workgroupBarrier();
    }
    let m_new = max(m, sh_red[0]);
    let rescale = exp(m - m_new);

    // Phase 2b: each lane exponentiates ITS position once -> p (reuse sh_s),
    // and reduce the block sum of p.
    var p_t: f32 = 0.0;
    if (t < n) { p_t = exp(sh_s[t] - m_new); }
    workgroupBarrier(); // all reads of sh_s (the scores) done before overwrite
    sh_s[t] = p_t;
    sh_red[t] = p_t;
    workgroupBarrier();
    for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
      if (t < stride) { sh_red[t] = sh_red[t] + sh_red[t + stride]; }
      workgroupBarrier();
    }
    l = rescale * l + sh_red[0];

    // Phase 2c: each lane accumulates V for its own output dims over the block,
    // reading the shared per-position weights (no exp in this loop).
    for (var nd: u32 = 0u; nd < MAX_D_PER_LANE; nd = nd + 1u) {
      let d = t + nd * WG_SIZE;
      if (d < D) {
        var acc: f32 = rescale * o_acc[nd];
        for (var j: u32 = 0u; j < n; j = j + 1u) {
          let vbase = (block + j) * kv_row_stride + kv * D;
          acc = acc + sh_s[j] * t_v_cache[vbase + d];
        }
        o_acc[nd] = acc;
      }
    }
    m = m_new;
    workgroupBarrier(); // before the next block overwrites sh_s / sh_red
    block = block + WG_SIZE;
  }

  let part = h * MAX_SPLITS + split_i;
  for (var nd: u32 = 0u; nd < MAX_D_PER_LANE; nd = nd + 1u) {
    let d = t + nd * WG_SIZE;
    if (d < D) {
      t_part_o[part * D + d] = o_acc[nd];
    }
  }
  if (t == 0u) {
    t_part_ml[part * 2u + 0u] = m;
    t_part_ml[part * 2u + 1u] = l;
  }
}
)";

inline constexpr uint32_t kSdpaFdSplitWorkgroupSizeX = 64;
inline constexpr uint32_t kSdpaFdSplitWorkgroupSizeY = 1;
inline constexpr uint32_t kSdpaFdSplitWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
