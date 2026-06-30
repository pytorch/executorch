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

// @generated from q4gsw_linear.wgsl - DO NOT EDIT.
// wgsl-sha256: dc6a55014ae4543bd80e5e22c3fb52896aca96e0589f700803327d8121ada489
inline constexpr const char* kQ4gswLinearWGSL = R"(
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

// Register-tiled GEMM: dequant weight once per (n,k), reused across TM rows.
const TM: u32 = 4u;
const TN: u32 = 4u;
const TILE_ELEMS: u32 = TM * TN; // accumulator size; keeps acc in sync with TM/TN

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nrt = (params.M + TM - 1u) / TM;
  let nct = (params.N + TN - 1u) / TN;
  let tiles = nrt * nct;
  // M==0 or N==0 -> tiles==0 -> every thread returns here, so the M-1u/N-1u
  // clamps below never underflow (the host also rejects M==0/N==0).
  if (gid.x >= tiles) {
    return;
  }
  let row_tile = gid.x / nct;
  let col_tile = gid.x % nct;
  let m0 = row_tile * TM;
  let n0 = col_tile * TN;

  var acc: array<f32, TILE_ELEMS>;
  for (var i: u32 = 0u; i < TILE_ELEMS; i = i + 1u) {
    acc[i] = 0.0;
  }

  var k: u32 = 0u;
  loop {
    if (k >= params.K) {
      break;
    }
    // Load the TM input values for column k once; reused across all TN columns.
    var in_reg: array<f32, TM>;
    for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
      let m_eff = min(m0 + ml, params.M - 1u);
      in_reg[ml] = t_input[m_eff * params.K + k];
    }
    for (var nl: u32 = 0u; nl < TN; nl = nl + 1u) {
      // Clamp to last valid column; overhang result is never stored.
      let n_eff = min(n0 + nl, params.N - 1u);
      let byte_idx = n_eff * params.K_packed + (k >> 1u);
      let word = t_weight[byte_idx >> 2u];
      let b = (word >> ((byte_idx & 3u) * 8u)) & 0xFFu;
      var nib: u32;
      if ((k & 1u) == 0u) {
        nib = b & 0x0Fu;         // even k -> low nibble
      } else {
        nib = (b >> 4u) & 0x0Fu; // odd k -> high nibble
      }
      let q = f32(i32(nib) - 8); // +8-shifted on pack; recover signed [-8,7]
      let dq = q * t_scales[(k / params.group_size) * params.padded_N + n_eff];
      for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
        acc[ml * TN + nl] = acc[ml * TN + nl] + in_reg[ml] * dq;
      }
    }
    k = k + 1u;
  }

  for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
    let m = m0 + ml;
    for (var nl: u32 = 0u; nl < TN; nl = nl + 1u) {
      let n = n0 + nl;
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

inline constexpr uint32_t kQ4gswLinearWorkgroupSizeX = 64;
inline constexpr uint32_t kQ4gswLinearWorkgroupSizeY = 1;
inline constexpr uint32_t kQ4gswLinearWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
