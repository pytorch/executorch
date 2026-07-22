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

// @generated from q4gsw_steel_bk64.wgsl - DO NOT EDIT.
// wgsl-sha256: f7425201059aa7d7d7671f405bba65d446627474d89db87a12fac4a406f6f2b2
inline constexpr const char* kQ4gswSteelBk64WGSL = R"(
enable f16;

@group(0) @binding(0) var<storage, read_write> t_out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> t_input: array<vec4<f32>>;
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

// BK64 prefill variant: group_size=64 keeps one scale valid for all eight packed words.
const BM: u32 = 64u; const BN: u32 = 64u; const BK: u32 = 64u;
var<workgroup> As: array<f16, 4096>;
var<workgroup> Bs: array<f16, 4096>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let nbN = (params.N + BN - 1u) / BN;
  let bx = wid.x % nbN;
  let by = wid.x / nbN;
  let row0 = by * BM;
  let col0 = bx * BN;
  let tid = lid.y * 16u + lid.x;
  var acc: array<array<f16, 4>, 4>;
  for (var m: u32 = 0u; m < 4u; m = m + 1u) {
    for (var n: u32 = 0u; n < 4u; n = n + 1u) { acc[m][n] = 0.0h; }
  }
  let ar = tid / 4u;
  let ac = (tid % 4u) * 4u;

  var k0: u32 = 0u;
  loop {
    if (k0 >= params.K) { break; }
    let arow = row0 + ar;
    if (arow < params.M) {
      let base = arow * params.K + k0 + ac;
      let av0 = t_input[(base + 0u) >> 2u];
      let av1 = t_input[(base + 16u) >> 2u];
      let av2 = t_input[(base + 32u) >> 2u];
      let av3 = t_input[(base + 48u) >> 2u];
      As[ar * BK + ac + 0u] = f16(av0.x); As[ar * BK + ac + 1u] = f16(av0.y);
      As[ar * BK + ac + 2u] = f16(av0.z); As[ar * BK + ac + 3u] = f16(av0.w);
      As[ar * BK + ac + 16u] = f16(av1.x); As[ar * BK + ac + 17u] = f16(av1.y);
      As[ar * BK + ac + 18u] = f16(av1.z); As[ar * BK + ac + 19u] = f16(av1.w);
      As[ar * BK + ac + 32u] = f16(av2.x); As[ar * BK + ac + 33u] = f16(av2.y);
      As[ar * BK + ac + 34u] = f16(av2.z); As[ar * BK + ac + 35u] = f16(av2.w);
      As[ar * BK + ac + 48u] = f16(av3.x); As[ar * BK + ac + 49u] = f16(av3.y);
      As[ar * BK + ac + 50u] = f16(av3.z); As[ar * BK + ac + 51u] = f16(av3.w);
    } else {
      for (var segment: u32 = 0u; segment < 4u; segment = segment + 1u) {
        for (var ai: u32 = 0u; ai < 4u; ai = ai + 1u) {
          As[ar * BK + ac + segment * 16u + ai] = 0.0h;
        }
      }
    }
    if (tid < BN) {
      let c = tid;
      let n = col0 + c;
      if (n < params.N) {
        let scale_row = (k0 / params.group_size) * params.padded_N;
        let scale = f16(t_scales[scale_row + n]);
        let base_word = n * (params.K_packed >> 2u) + (k0 >> 3u);
        let w0 = t_weight[base_word + 0u];
        let w1 = t_weight[base_word + 1u];
        let w2 = t_weight[base_word + 2u];
        let w3 = t_weight[base_word + 3u];
        let w4 = t_weight[base_word + 4u];
        let w5 = t_weight[base_word + 5u];
        let w6 = t_weight[base_word + 6u];
        let w7 = t_weight[base_word + 7u];
        let words = array<u32, 8>(w0, w1, w2, w3, w4, w5, w6, w7);
        for (var br: u32 = 0u; br < BK; br = br + 1u) {
          let word = words[br >> 3u];
          let nib = (word >> ((br & 7u) * 4u)) & 0x0Fu;
          Bs[br * BN + c] = f16(i32(nib) - 8) * scale;
        }
      } else {
        for (var br: u32 = 0u; br < BK; br = br + 1u) { Bs[br * BN + c] = 0.0h; }
      }
    }
    workgroupBarrier();
    for (var k: u32 = 0u; k < BK; k = k + 1u) {
      var a: array<f16, 4>;
      var bvec: array<f16, 4>;
      for (var m: u32 = 0u; m < 4u; m = m + 1u) { a[m] = As[(lid.y * 4u + m) * BK + k]; }
      for (var n: u32 = 0u; n < 4u; n = n + 1u) { bvec[n] = Bs[k * BN + lid.x * 4u + n]; }
      for (var m: u32 = 0u; m < 4u; m = m + 1u) {
        for (var n: u32 = 0u; n < 4u; n = n + 1u) { acc[m][n] = fma(a[m], bvec[n], acc[m][n]); }
      }
    }
    workgroupBarrier();
    k0 = k0 + BK;
  }
  for (var m: u32 = 0u; m < 4u; m = m + 1u) {
    let r = row0 + lid.y * 4u + m;
    let c0 = col0 + lid.x * 4u;
    if (r < params.M && c0 < params.N) {
      var vv = vec4<f32>(
          f32(acc[m][0]), f32(acc[m][1]), f32(acc[m][2]), f32(acc[m][3]));
      if (params.has_bias != 0u) {
        vv = vv + vec4<f32>(
            t_bias[c0], t_bias[c0 + 1u], t_bias[c0 + 2u], t_bias[c0 + 3u]);
      }
      t_out[(r * params.N + c0) >> 2u] = vv;
    }
  }
}
)";

inline constexpr uint32_t kQ4gswSteelBk64WorkgroupSizeX = 16;
inline constexpr uint32_t kQ4gswSteelBk64WorkgroupSizeY = 16;
inline constexpr uint32_t kQ4gswSteelBk64WorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
