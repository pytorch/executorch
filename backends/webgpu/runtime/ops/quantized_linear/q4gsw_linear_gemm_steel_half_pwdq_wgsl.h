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

// @generated from q4gsw_linear_gemm_steel_half_pwdq.wgsl - DO NOT EDIT.
// wgsl-sha256: d4e3ce11f873b8cec80b1a9457915d0e50a334e286f1d75753f9f3b023728ac4
inline constexpr const char* kQ4gswLinearGemmSteelHalfPwdqWGSL = R"(
enable f16;

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

// Packed-word-dequant f16 "steel" GEMM (the `half` variant of
// q4gsw_linear_gemm_steel). Loads each u32 weight word ONCE, unpacks all 16
// nibbles of one BN column, and hoists the per-column scale to one read (the
// per-nibble steel `half` re-reads each ~8x/~16x). 64x64 tile / 256-thread /
// BK=16 geometry. Two ACC variants from this one template:
//   ACC=float ("pwdq"): f32 accumulate -- BIT-EXACT to the steel `half` kernel.
//   ACC=half  ("pwdqf16acc"): f16 accumulate with fma() (MLC-style), cast to f32
//     in the epilogue -- LOSSY, perplexity-gated, opt-in via STEEL_F16ACC.
// Requires K%BK==0 (steel route guarantees it, so K_packed=K/2 is a multiple of 8
// and every column is u32-aligned) and group_size%BK==0 (hoisted scale constant
// across the BK tile).
const BM: u32 = 64u; const BN: u32 = 64u; const BK: u32 = 16u;
var<workgroup> As: array<f16, 1024>;   // BM*BK, staged as f16 (multiply operand only)
var<workgroup> Bs: array<f16, 1024>;   // BK*BN, dequantized straight to f16
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let nbN = (params.N + BN - 1u) / BN;
  let bx = wid.x % nbN;      // decode 2D tile id from 1D dispatch
  let by = wid.x / nbN;
  let row0 = by * BM;
  let col0 = bx * BN;
  let tid = lid.y * 16u + lid.x;
  var acc: array<array<f32, 4>, 4>;
  for (var m: u32 = 0u; m < 4u; m = m + 1u) {
    for (var n: u32 = 0u; n < 4u; n = n + 1u) { acc[m][n] = 0.0; }
  }
  let ar = tid / 4u;
  let ac = (tid % 4u) * 4u;

  var k0: u32 = 0u;
  loop {
    if (k0 >= params.K) { break; }
    let arow = row0 + ar;
    if (arow < params.M) {
      let base = arow * params.K + k0 + ac;
      As[ar * BK + ac + 0u] = f16(t_input[base]);
      As[ar * BK + ac + 1u] = f16(t_input[base + 1u]);
      As[ar * BK + ac + 2u] = f16(t_input[base + 2u]);
      As[ar * BK + ac + 3u] = f16(t_input[base + 3u]);
    } else {
      As[ar * BK + ac + 0u] = 0.0h; As[ar * BK + ac + 1u] = 0.0h;
      As[ar * BK + ac + 2u] = 0.0h; As[ar * BK + ac + 3u] = 0.0h;
    }
    // Packed-word dequant: threads [0,BN) each stage one full BK-column of Bs.
    if (tid < BN) {
      let c = tid;                 // Bs column within this tile
      let n = col0 + c;            // global output column
      if (n < params.N) {
        // Scale is constant across the BK tile (group_size % BK == 0 for all real
        // group sizes; K%BK==0 on the steel route), so hoist it to one read.
        let scale_row = (k0 / params.group_size) * params.padded_N;
        let scale = f16(t_scales[scale_row + n]);
        // Column n's 16-nibble K-slice for this tile = two consecutive words.
        // K_packed multiple of 8 => base_word stays inside column n's own region.
        let base_word = n * (params.K_packed >> 2u) + (k0 >> 3u);
        let w0 = t_weight[base_word];
        let w1 = t_weight[base_word + 1u];
        for (var br: u32 = 0u; br < BK; br = br + 1u) {
          let word = select(w1, w0, br < 8u);        // word0 holds K-slice [0,8)
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
        for (var n: u32 = 0u; n < 4u; n = n + 1u) { acc[m][n] = acc[m][n] + f32(a[m] * bvec[n]); }
      }
    }
    workgroupBarrier();
    k0 = k0 + BK;
  }
  for (var m: u32 = 0u; m < 4u; m = m + 1u) {
    for (var n: u32 = 0u; n < 4u; n = n + 1u) {
      let r = row0 + lid.y * 4u + m;
      let c = col0 + lid.x * 4u + n;
      if (r < params.M && c < params.N) {
        var v = acc[m][n];
        if (params.has_bias != 0u) { v = v + t_bias[c]; }
        t_out[r * params.N + c] = v;
      }
    }
  }
}
)";

inline constexpr uint32_t kQ4gswLinearGemmSteelHalfPwdqWorkgroupSizeX = 16;
inline constexpr uint32_t kQ4gswLinearGemmSteelHalfPwdqWorkgroupSizeY = 16;
inline constexpr uint32_t kQ4gswLinearGemmSteelHalfPwdqWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
