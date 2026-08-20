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

// @generated from mm.wgsl - DO NOT EDIT.
// wgsl-sha256: bbd9aef967c5085c52eb88b8abcd68202cefe7fc4df04a25c4b00781c5b7ca5a
inline constexpr const char* kMmWGSL = R"(
struct Params {
  M: u32,
  N: u32,
  K: u32,
  pad_: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.M * params.N;
  if (idx >= total) {
    return;
  }
  let m = idx / params.N;
  let n = idx % params.N;
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < params.K; k = k + 1u) {
    acc = acc + a[m * params.K + k] * b[k * params.N + n];
  }
  out[idx] = acc;
}
)";

inline constexpr uint32_t kMmWorkgroupSizeX = 256;
inline constexpr uint32_t kMmWorkgroupSizeY = 1;
inline constexpr uint32_t kMmWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
