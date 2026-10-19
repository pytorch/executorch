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

// @generated from conv_transpose2d_gemm.wgsl - DO NOT EDIT.
// wgsl-sha256: 4be3a45b3d0fa03ae54b41e3e1906f503b51bfd4048892ff893c223258b4351c
inline constexpr const char* kConvTranspose2dGemmWGSL = R"(
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  B: u32, IC: u32, IH: u32, IW: u32,
  OC: u32, OH: u32, OW: u32, KH: u32, KW: u32,
  sH: u32, sW: u32, pH: u32, pW: u32, dH: u32, dW: u32,
  groups: u32, has_bias: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

// Non-overlapping transposed conv (stride == kernel, no padding/dilation,
// groups == 1): every output pixel takes exactly one kernel tap, so it is a
// GEMM M=OC*KH*KW, N=B*IH*IW, K=IC followed by a depth-to-space scatter.
// A = weight [IC, OC, KH, KW] read transposed (A[m,k] = weight[k*M + m]); B =
// input [B, IC, IH*IW]. Same tiling as conv2d_gemm (TILE=32, RPT=4).
const TILE: u32 = 32u;
const RPT: u32 = 4u;

var<workgroup> a_sub: array<array<f32, 32>, 32>;
var<workgroup> b_sub: array<array<f32, 32>, 32>;

fn gM() -> u32 { return params.OC * params.KH * params.KW; }
fn gN() -> u32 { return params.B * params.IH * params.IW; }

fn read_a(m: u32, k: u32) -> f32 {
  if (m < gM() && k < params.IC) {
    return weight[k * gM() + m];
  }
  return 0.0;
}

fn read_b(k: u32, n: u32) -> f32 {
  if (k >= params.IC || n >= gN()) {
    return 0.0;
  }
  let ihw = params.IH * params.IW;
  return input[((n / ihw) * params.IC + k) * ihw + n % ihw];
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
  let M = gM();
  let N = gN();
  let K = params.IC;
  let tile_row0 = wg_id.y * TILE;
  let tile_col0 = wg_id.x * TILE;
  let tile_row = local_id.y * RPT;
  let tile_col = local_id.x * RPT;

  var acc: array<array<f32, 4>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      acc[ir][ic] = 0.0;
    }
  }

  let num_tiles = (K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k_start = t * TILE;
    for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
      let arow = local_id.y * RPT + ir;
      for (var kk: u32 = 0u; kk < RPT; kk = kk + 1u) {
        let col = local_id.x * RPT + kk;
        a_sub[arow][col] = read_a(tile_row0 + arow, k_start + col);
        b_sub[arow][col] = read_b(k_start + arow, tile_col0 + col);
      }
    }
    workgroupBarrier();
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
        let aval = a_sub[tile_row + ir][k];
        for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
          acc[ir][ic] = acc[ir][ic] + aval * b_sub[k][tile_col + ic];
        }
      }
    }
    workgroupBarrier();
  }

  let khw = params.KH * params.KW;
  let ihw = params.IH * params.IW;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    let m = tile_row0 + tile_row + ir;
    if (m >= M) {
      continue;
    }
    let oc = m / khw;
    let kh = (m % khw) / params.KW;
    let kw = m % params.KW;
    var bias_v = 0.0;
    if (params.has_bias != 0u) {
      bias_v = bias[oc];
    }
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      let n = tile_col0 + tile_col + ic;
      if (n < N) {
        let b = n / ihw;
        let sp = n % ihw;
        let oh = (sp / params.IW) * params.KH + kh;
        let ow = (sp % params.IW) * params.KW + kw;
        out[((b * params.OC + oc) * params.OH + oh) * params.OW + ow] =
            acc[ir][ic] + bias_v;
      }
    }
  }
}
)";

inline constexpr uint32_t kConvTranspose2dGemmWorkgroupSizeX = 8;
inline constexpr uint32_t kConvTranspose2dGemmWorkgroupSizeY = 8;
inline constexpr uint32_t kConvTranspose2dGemmWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
