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

// @generated from bitwise_not.wgsl - DO NOT EDIT.
// wgsl-sha256: 06c95212af5f02c08978288f25552da0e2476c37f9421e941d059338ffa1aa9f
inline constexpr const char* kBitwiseNotWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_out: array<u32>;
@group(0) @binding(1) var<storage, read> t_a: array<u32>;

struct Params {
  num_words: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // bool NOT packed 4/word: per-byte 1-x == x^1; word-wise ^0x01010101.
  let w = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (w >= params.num_words) {
    return;
  }
  t_out[w] = t_a[w] ^ 0x01010101u;
}
)";

inline constexpr uint32_t kBitwiseNotWorkgroupSizeX = 64;
inline constexpr uint32_t kBitwiseNotWorkgroupSizeY = 1;
inline constexpr uint32_t kBitwiseNotWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
