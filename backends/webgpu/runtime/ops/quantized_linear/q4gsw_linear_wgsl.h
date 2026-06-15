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
// wgsl-sha256: 966cec5d4102eb7c8f6504d2a335a1bd2f235424933fe83b4d0f8f274d894f39
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

// One workgroup per row m, threads stride N; loop logical K only (in-bounds).
@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>) {
  let m = wid.x;
  if (m >= params.M) {
    return;
  }
  let in_base = m * params.K;

  var n: u32 = lid.x;
  loop {
    if (n >= params.N) {
      break;
    }
    var acc: f32 = 0.0;
    var k: u32 = 0u;
    loop {
      if (k >= params.K) {
        break;
      }
      // Packed weight byte for (n, k): row stride K_packed bytes, byte k/2.
      let byte_idx = n * params.K_packed + (k >> 1u);
      let word = t_weight[byte_idx >> 2u];
      let b = (word >> ((byte_idx & 3u) * 8u)) & 0xFFu;
      var nib: u32;
      if ((k & 1u) == 0u) {
        nib = b & 0x0Fu;       // even k -> low nibble
      } else {
        nib = (b >> 4u) & 0x0Fu; // odd k -> high nibble
      }
      let q = f32(i32(nib) - 8); // +8-shifted on pack; recover signed [-8,7]
      let scale = t_scales[(k / params.group_size) * params.padded_N + n];
      acc = acc + t_input[in_base + k] * q * scale;
      k = k + 1u;
    }
    if (params.has_bias != 0u) {
      acc = acc + t_bias[n];
    }
    t_out[m * params.N + n] = acc;
    n = n + wg_size;
  }
}
)";

inline constexpr uint32_t kQ4gswLinearWorkgroupSizeX = 64;
inline constexpr uint32_t kQ4gswLinearWorkgroupSizeY = 1;
inline constexpr uint32_t kQ4gswLinearWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
