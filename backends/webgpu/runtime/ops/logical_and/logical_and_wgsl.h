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

// @generated from logical_and.wgsl - DO NOT EDIT.
// wgsl-sha256: cf7c1d1dbba94e429120796c9c25a6717786cca03c08f3bd1e291d5627089c20
inline constexpr const char* kLogicalAndWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_out: array<u32>;
@group(0) @binding(1) var<storage, read> t_a: array<u32>;
@group(0) @binding(2) var<storage, read> t_b: array<u32>;

struct Params {
  num_words: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // bool packed 4/word; canonical 0/1 bytes -> word-wise AND == per-byte AND.
  let w = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (w >= params.num_words) {
    return;
  }
  t_out[w] = t_a[w] & t_b[w];
}
)";

inline constexpr uint32_t kLogicalAndWorkgroupSizeX = 64;
inline constexpr uint32_t kLogicalAndWorkgroupSizeY = 1;
inline constexpr uint32_t kLogicalAndWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
