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

// @generated from q4gsw_linear_gemm_qkv_fused.wgsl - DO NOT EDIT.
// wgsl-sha256: 93e127e8ee4609d846015c8b75a600a29502e19a92bdf3a08e3429635f834085
inline constexpr const char* kQ4gswLinearGemmQkvFusedWGSL = R"(
enable f16;
// Fused QKV q4gsw GEMM (Llama attention projections): one [M, N=3072] pwdq + f16-accumulate GEMM
// (vec4<f32> activation load) that scatter-writes each output column range to a SEPARATE buffer --
// c<2048 -> q, [2048,2560) -> k, [2560,3072) -> v. Replaces the 3 separate q/k/v linear dispatches;
// fixes the N=512 K/V occupancy starvation (16 WGs -> 96 WGs at M~128). Boundaries are 64-tile-aligned
// so each 64-col tile maps to exactly one output (uniform branch per workgroup). Per-output ROW STRIDE:
// q=2048, k=v=512. BIT-EXACT to 3 separate pwdqf16acc linears (fusing along N does not change the
// per-column K-accumulation order). Validated on Canary M4 Pro: correct (maxRel ~1e-3), scatter overhead
// 1.02x (free), concat win 1.63x on the QKV block. Boundaries hardcoded for Llama-3.2-1B GQA (32Q/8KV).
@group(0) @binding(0) var<storage, read_write> t_out_q: array<f32>;
@group(0) @binding(1) var<storage, read_write> t_out_k: array<f32>;
@group(0) @binding(2) var<storage, read_write> t_out_v: array<f32>;
@group(0) @binding(3) var<storage, read> t_input: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> t_weight: array<u32>;
@group(0) @binding(5) var<storage, read> t_scales: array<f32>;
@group(0) @binding(6) var<storage, read> t_bias: array<f32>;
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
@group(0) @binding(7) var<uniform> params: Params;
const BM: u32 = 64u; const BN: u32 = 64u; const BK: u32 = 16u;
const N_Q: u32 = 2048u; const N_QK: u32 = 2560u; const N_KV: u32 = 512u;
var<workgroup> As: array<f16, 1024>;
var<workgroup> Bs: array<f16, 1024>;
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
      let av = t_input[base >> 2u];
      As[ar * BK + ac + 0u] = f16(av.x); As[ar * BK + ac + 1u] = f16(av.y);
      As[ar * BK + ac + 2u] = f16(av.z); As[ar * BK + ac + 3u] = f16(av.w);
    } else {
      As[ar * BK + ac + 0u] = 0.0h; As[ar * BK + ac + 1u] = 0.0h;
      As[ar * BK + ac + 2u] = 0.0h; As[ar * BK + ac + 3u] = 0.0h;
    }
    if (tid < BN) {
      let c = tid;
      let n = col0 + c;
      if (n < params.N) {
        let scale_row = (k0 / params.group_size) * params.padded_N;
        let scale = f16(t_scales[scale_row + n]);
        let base_word = n * (params.K_packed >> 2u) + (k0 >> 3u);
        let w0 = t_weight[base_word];
        let w1 = t_weight[base_word + 1u];
        for (var br: u32 = 0u; br < BK; br = br + 1u) {
          let word = select(w1, w0, br < 8u);
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
    for (var n: u32 = 0u; n < 4u; n = n + 1u) {
      let r = row0 + lid.y * 4u + m;
      let c = col0 + lid.x * 4u + n;   // global fused column [0, 3072)
      if (r < params.M && c < params.N) {
        var val = f32(acc[m][n]);
        if (params.has_bias != 0u) { val = val + t_bias[c]; }
        if (c < N_Q) { t_out_q[r * N_Q + c] = val; }
        else if (c < N_QK) { t_out_k[r * N_KV + (c - N_Q)] = val; }
        else { t_out_v[r * N_KV + (c - N_QK)] = val; }
      }
    }
  }
}
)";

inline constexpr uint32_t kQ4gswLinearGemmQkvFusedWorkgroupSizeX = 16;
inline constexpr uint32_t kQ4gswLinearGemmQkvFusedWorkgroupSizeY = 16;
inline constexpr uint32_t kQ4gswLinearGemmQkvFusedWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
