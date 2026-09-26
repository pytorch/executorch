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

// @generated from qcs4w_linear.wgsl - DO NOT EDIT.
// wgsl-sha256: 59a17d804c020c01322cbbd25d0984183fa7016186b77bfedd53733185f83fc6
inline constexpr const char* kQcs4wLinearWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_input: array<f32>;
@group(0) @binding(2) var<storage, read> t_weight: array<u32>;
@group(0) @binding(3) var<storage, read> t_scales: array<f32>;

struct Params {
  M: u32,
  N: u32,
  K: u32,
  K_packed: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

// Register-tiled GEMM: dequant weight once per (n,k), reused across TM rows.
const TM: u32 = 4u;
const TN: u32 = 4u;
const TILE_ELEMS: u32 = TM * TN; // acc size; kept in sync with TM/TN

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let nrt = (params.M + TM - 1u) / TM;
  let nct = (params.N + TN - 1u) / TN;
  let tiles = nrt * nct;
  // 2D-folded flat tile index (lifts the 65535 1D-dispatch cap for large M/N).
  let tile = gid.x + gid.y * (num_workgroups.x * wg_size);
  // M==0 or N==0 -> tiles==0 -> every thread returns, so the M-1u/N-1u clamps
  // below never underflow (the host also rejects M==0/N==0).
  if (tile >= tiles) {
    return;
  }
  let row_tile = tile / nct;
  let col_tile = tile % nct;
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
      // qcs4w prepack packs (even<<4)|odd (swapped vs q4gsw's pack_4bit).
      if ((k & 1u) == 0u) {
        nib = (b >> 4u) & 0x0Fu; // even k -> high nibble
      } else {
        nib = b & 0x0Fu;         // odd k -> low nibble
      }
      let q = f32(i32(nib) - 8); // +8-shifted on pack; recover signed [-8,7]
      let dq = q * t_scales[n_eff]; // per-channel symmetric scale
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
        t_out[m * params.N + n] = acc[ml * TN + nl];
      }
    }
  }
}
)";

inline constexpr uint32_t kQcs4wLinearWorkgroupSizeX = 64;
inline constexpr uint32_t kQcs4wLinearWorkgroupSizeY = 1;
inline constexpr uint32_t kQcs4wLinearWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
