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

// @generated from leaky_relu.wgsl - DO NOT EDIT.
// wgsl-sha256: 98add607747a0f47521645f6cb046c500d59709f6641846a0099c5acd3b6a66b
inline constexpr const char* kLeakyReluWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
  neg_slope: f32,
  _p0: u32,
  _p1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill

// leaky_relu(x) = max(x,0) + neg_slope*min(x,0); fp32 elementwise, 2D-spill.
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * stride_x + gid.x;
  if (i >= params.num_elements) {
    return;
  }
  let x = input[i];
  output[i] = max(x, 0.0) + params.neg_slope * min(x, 0.0);
}
)";

inline constexpr uint32_t kLeakyReluWorkgroupSizeX = 256;
inline constexpr uint32_t kLeakyReluWorkgroupSizeY = 1;
inline constexpr uint32_t kLeakyReluWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
