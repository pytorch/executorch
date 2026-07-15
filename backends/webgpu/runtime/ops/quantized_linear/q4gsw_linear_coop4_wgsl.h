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

// @generated from q4gsw_linear_coop4.wgsl - DO NOT EDIT.
// wgsl-sha256: 3031886e68c375e617dfb263da39c492c6de4d8c1fb4073d70b18823a3e6a4fe
inline constexpr const char* kQ4gswLinearCoop4WGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_input: array<f32>;
@group(0) @binding(2) var<storage, read> t_weight: array<u32>;
@group(0) @binding(3) var<storage, read> t_scales: array<f32>;
@group(0) @binding(4) var<storage, read> t_bias: array<f32>;

struct Params {
  M: u32,
  N: u32,
  K: u32,
  K_packed: u32,
  group_size: u32,
  padded_N: u32,
  has_bias: u32,
  _pad: u32,
}
@group(0) @binding(5) var<uniform> params: Params;

// Cooperative-over-K GEMV with u32-batched coalesced weight loads (64 lanes).
const WG: u32 = 64u;
var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ngrp: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>) {
  let total = params.M * params.N;
  let stride = ngrp.x;
  let num_words = params.K >> 3u;          // K / 8 words per row
  let row_words = params.K_packed >> 2u;   // u32s per weight row (= K/8)
  var idx = wid.x;
  loop {
    if (idx >= total) {
      break;
    }
    let m = idx / params.N;
    let n = idx % params.N;
    let in_base = m * params.K;
    let wbase = n * row_words;

    var acc: f32 = 0.0;
    var w: u32 = lid.x;
    loop {
      if (w >= num_words) {
        break;
      }
      let word = t_weight[wbase + w];
      let k0 = w << 3u; // first K of this word
      let scale = t_scales[(k0 / params.group_size) * params.padded_N + n];
      let ib = in_base + k0;
      // 4 bytes, low+high nibble each -> 8 consecutive K.
      for (var bi: u32 = 0u; bi < 4u; bi = bi + 1u) {
        let byte = (word >> (bi * 8u)) & 0xFFu;
        let lo = f32(i32(byte & 0x0Fu) - 8);
        let hi = f32(i32((byte >> 4u) & 0x0Fu) - 8);
        let kk = bi << 1u;
        acc = acc + t_input[ib + kk] * lo * scale;
        acc = acc + t_input[ib + kk + 1u] * hi * scale;
      }
      w = w + WG;
    }

    partial[lid.x] = acc;
    workgroupBarrier();
    var s: u32 = WG >> 1u;
    loop {
      if (s == 0u) {
        break;
      }
      if (lid.x < s) {
        partial[lid.x] = partial[lid.x] + partial[lid.x + s];
      }
      workgroupBarrier();
      s = s >> 1u;
    }
    if (lid.x == 0u) {
      var o = partial[0];
      if (params.has_bias != 0u) {
        o = o + t_bias[n];
      }
      t_out[idx] = o;
    }
    workgroupBarrier();
    idx = idx + stride;
  }
}
)";

inline constexpr uint32_t kQ4gswLinearCoop4WorkgroupSizeX = 64;
inline constexpr uint32_t kQ4gswLinearCoop4WorkgroupSizeY = 1;
inline constexpr uint32_t kQ4gswLinearCoop4WorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
