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

// @generated from linear_dq8ca_q4gsw.wgsl - DO NOT EDIT.
// wgsl-sha256: a3ec7f77b0bbb629b57f5a91a6ab33d96fcce3c7af3e23db1b599926ed04a5df
inline constexpr const char* kLinearDq8caQ4gswWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_input: array<f32>;
@group(0) @binding(2) var<storage, read> t_input_scale: array<f32>;
@group(0) @binding(3) var<storage, read> t_input_zp: array<u32>;
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

override wg_size: u32 = 64u;

// Register-tiled GEMM (mirrors q4gsw_linear.wgsl) with folded dynamic per-row
// int8 activation quant: out[m,n] = s[m] * sum_k (xq[m,k]-z[m]) * dequant(w).
// s[m]/z[m] come from choose_qparams_affine (per-row asymmetric). Weight side is
// the SAME symmetric 4-bit-group packing as linear_q4gsw (nibble - 8).
const TM: u32 = 4u;
const TN: u32 = 4u;
const TILE_ELEMS: u32 = TM * TN;

// int8 zero-point is packed 4-per-u32 (elem_size 1); extract the signed byte.
fn unpack_zp(idx: u32) -> i32 {
  let word = t_input_zp[idx >> 2u];
  return i32(((word >> ((idx & 3u) * 8u)) & 0xFFu) ^ 0x80u) - 128;
}

// xq = clamp(round(x/s) + z, -128, 127); returns (xq - z) as f32 (the value the
// dequant sum uses; s is applied once to the accumulator at the end).
fn quant_act_centered(x: f32, s: f32, z: i32) -> f32 {
  let q = clamp(i32(round(x / s)) + z, -128, 127);
  return f32(q - z);
}

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let nrt = (params.M + TM - 1u) / TM;
  let nct = (params.N + TN - 1u) / TN;
  let tiles = nrt * nct;
  // 2D-folded flat tile index (lifts the 65535 1D-dispatch cap on prefill).
  let tile = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (tile >= tiles) {
    return;
  }
  let row_tile = tile / nct;
  let col_tile = tile % nct;
  let m0 = row_tile * TM;
  let n0 = col_tile * TN;

  // Per-row activation scale/zp (clamp row index for the TM tile overhang).
  var s_reg: array<f32, TM>;
  var z_reg: array<i32, TM>;
  for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
    let m_eff = min(m0 + ml, params.M - 1u);
    s_reg[ml] = t_input_scale[m_eff];
    z_reg[ml] = unpack_zp(m_eff);
  }

  var acc: array<f32, TILE_ELEMS>;
  for (var i: u32 = 0u; i < TILE_ELEMS; i = i + 1u) {
    acc[i] = 0.0;
  }

  var k: u32 = 0u;
  loop {
    if (k >= params.K) {
      break;
    }
    // Quantize the TM activation values for column k once; reused across TN cols.
    var in_reg: array<f32, TM>;
    for (var ml: u32 = 0u; ml < TM; ml = ml + 1u) {
      let m_eff = min(m0 + ml, params.M - 1u);
      in_reg[ml] =
          quant_act_centered(t_input[m_eff * params.K + k], s_reg[ml], z_reg[ml]);
    }
    for (var nl: u32 = 0u; nl < TN; nl = nl + 1u) {
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
        var v = s_reg[ml] * acc[ml * TN + nl];
        if (params.has_bias != 0u) {
          v = v + t_bias[n];
        }
        t_out[m * params.N + n] = v;
      }
    }
  }
}
)";

inline constexpr uint32_t kLinearDq8caQ4gswWorkgroupSizeX = 64;
inline constexpr uint32_t kLinearDq8caQ4gswWorkgroupSizeY = 1;
inline constexpr uint32_t kLinearDq8caQ4gswWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
