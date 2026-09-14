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

// @generated from q4gsw_requant.wgsl - DO NOT EDIT.
// wgsl-sha256: 66cbf8061af4580be1773acd2e9d11d539d1a9e4a3fc8d08bd1d4def42e4d643
inline constexpr const char* kQ4gswRequantWGSL = R"(
// STE re-quant + int4 pack: round(latent/scale) in [-8,7], 2 nibbles/byte.

@group(0) @binding(0) var<storage, read_write> t_packed: array<u32>;
@group(0) @binding(1) var<storage, read> t_latent: array<f32>;       // [N, K]
@group(0) @binding(2) var<storage, read> t_scales: array<f32>;

struct Params {
  N: u32,
  K: u32,
  K_packed: u32,
  group_size: u32,
  padded_N: u32,
  num_words: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64u;

fn quant_nibble(n: u32, k: u32) -> u32 {
  let s = t_scales[(k / params.group_size) * params.padded_N + n];
  var q: i32 = 0;
  if (s != 0.0) {
    q = i32(round(t_latent[n * params.K + k] / s));
  }
  q = clamp(q, -8, 7);
  return u32(q + 8) & 0xFu;
}

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let word_idx = gid.x;
  if (word_idx >= params.num_words) {
    return;
  }
  var word: u32 = 0u;
  for (var bi: u32 = 0u; bi < 4u; bi = bi + 1u) {
    let byte_idx = word_idx * 4u + bi;
    let n = byte_idx / params.K_packed;
    let byte_in_row = byte_idx % params.K_packed;
    let k_lo = byte_in_row * 2u;
    var b: u32 = 0u;
    if (n < params.N && k_lo < params.K) {
      b = quant_nibble(n, k_lo);
      let k_hi = k_lo + 1u;
      if (k_hi < params.K) {
        b = b | (quant_nibble(n, k_hi) << 4u);
      }
    }
    word = word | (b << (bi * 8u));
  }
  t_packed[word_idx] = word;
}
)";

inline constexpr uint32_t kQ4gswRequantWorkgroupSizeX = 64;
inline constexpr uint32_t kQ4gswRequantWorkgroupSizeY = 1;
inline constexpr uint32_t kQ4gswRequantWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
