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

// @generated from q4gsw_linear_gemm_shmem.wgsl - DO NOT EDIT.
// wgsl-sha256: 0a1219d3b6781315a21066089fca3f92235587e8af8eb734185f35ea4bfc8a52
inline constexpr const char* kQ4gswLinearGemmShmemWGSL = R"(
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

override wg_size: u32 = 64u;

// Shmem-staged tiled GEMM (M>1): dequant weight into shmem once per K-tile.
const WG_M: u32 = 32u;   // output rows per workgroup
const WG_N: u32 = 32u;   // output cols per workgroup
const TM: u32 = 4u;      // rows per thread
const TN: u32 = 4u;      // cols per thread
const TK: u32 = 16u;     // K-tile depth
const THREADS: u32 = 64u; // 8x8 thread grid (must equal wg_size)

var<workgroup> in_sh: array<f32, 512>;  // WG_M * TK = 32 * 16
var<workgroup> w_sh: array<f32, 512>;   // TK * WG_N = 16 * 32

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  let nct = (params.N + WG_N - 1u) / WG_N;   // output-col tiles
  let tile_row = wid.x / nct;
  let tile_col = wid.x % nct;
  let m_base = tile_row * WG_M;              // first output row of this WG's tile
  let n_base = tile_col * WG_N;              // first output col of this WG's tile

  // This thread's 4x4 sub-tile origin within the 32x32 WG tile (8x8 thread grid).
  let tr = lid / 8u;                          // thread row in [0,8)
  let tc = lid % 8u;                          // thread col in [0,8)
  let m_thread = m_base + tr * TM;
  let n_thread = n_base + tc * TN;

  var acc: array<f32, 16>;                    // TM * TN
  for (var i: u32 = 0u; i < TM * TN; i = i + 1u) {
    acc[i] = 0.0;
  }

  var k0: u32 = 0u;
  loop {
    if (k0 >= params.K) { break; }

    // (a) Stage the 32xTK input block into shmem; 64 threads x 8 elems; OOB->0.
    for (var t: u32 = lid; t < WG_M * TK; t = t + THREADS) {
      let r = t / TK;                         // row within the WG tile [0,32)
      let c = t % TK;                          // col within the K-tile [0,16)
      let m = m_base + r;
      let k = k0 + c;
      var v: f32 = 0.0;
      if (m < params.M && k < params.K) {
        v = t_input[m * params.K + k];
      }
      in_sh[t] = v;
    }

    // (b) Dequant the TKx32 weight block into shmem once; dq = (nibble-8)*scale.
    for (var t: u32 = lid; t < TK * WG_N; t = t + THREADS) {
      let kk = t / WG_N;                       // k within the K-tile [0,16)
      let nn = t % WG_N;                        // col within the WG tile [0,32)
      let n = n_base + nn;
      let k = k0 + kk;
      var dq: f32 = 0.0;
      if (n < params.N && k < params.K) {
        let byte_idx = n * params.K_packed + (k >> 1u);
        let word = t_weight[byte_idx >> 2u];
        let b = (word >> ((byte_idx & 3u) * 8u)) & 0xFFu;
        var nib: u32;
        if ((k & 1u) == 0u) {
          nib = b & 0x0Fu;                      // even k -> low nibble
        } else {
          nib = (b >> 4u) & 0x0Fu;              // odd k -> high nibble
        }
        let q = f32(i32(nib) - 8);              // +8-shifted on pack; recover [-8,7]
        dq = q * t_scales[(k / params.group_size) * params.padded_N + n];
      }
      w_sh[t] = dq;
    }

    workgroupBarrier();

    // Accumulate this thread's 4x4 tile from shared memory.
    let k_lim = min(TK, params.K - k0);
    for (var kk: u32 = 0u; kk < k_lim; kk = kk + 1u) {
      var a: array<f32, 4>;
      for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
        a[ml] = in_sh[(tr * TM + ml) * TK + kk];
      }
      for (var nl: u32 = 0u; nl < TN; nl = nl + 1u) {
        let wv = w_sh[kk * WG_N + (tc * TN + nl)];
        for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
          acc[ml * TN + nl] = acc[ml * TN + nl] + a[ml] * wv;
        }
      }
    }

    workgroupBarrier();
    k0 = k0 + TK;
  }

  // Write the 4x4 register tile out (bounds-guarded; bias added per column).
  for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
    let m = m_thread + ml;
    for (var nl: u32 = 0u; nl < TN; nl = nl + 1u) {
      let n = n_thread + nl;
      if (m < params.M && n < params.N) {
        var v = acc[ml * TN + nl];
        if (params.has_bias != 0u) {
          v = v + t_bias[n];
        }
        t_out[m * params.N + n] = v;
      }
    }
  }
}
)";

inline constexpr uint32_t kQ4gswLinearGemmShmemWorkgroupSizeX = 64;
inline constexpr uint32_t kQ4gswLinearGemmShmemWorkgroupSizeY = 1;
inline constexpr uint32_t kQ4gswLinearGemmShmemWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
