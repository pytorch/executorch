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

// @generated from batch_norm.wgsl - DO NOT EDIT.
// wgsl-sha256: ac7208cec0fe1454698b9b16985172c1ef34c994880b568238eaeaf1d62c0342
inline constexpr const char* kBatchNormWGSL = R"(
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> mean: array<f32>;
@group(0) @binding(5) var<storage, read> var_: array<f32>;

struct Params {
  num_elements: u32,
  C: u32,
  HW: u32,
  has_weight: u32,
  has_bias: u32,
  eps: f32,
  _p0: u32,
  _p1: u32,
}
@group(0) @binding(6) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u;

// Inference batch_norm NCHW fp32: per-channel affine from running stats.
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * stride_x + gid.x;
  if (i >= params.num_elements) {
    return;
  }
  let c = (i / params.HW) % params.C;
  let inv = inverseSqrt(var_[c] + params.eps);
  var g: f32 = 1.0;
  if (params.has_weight == 1u) {
    g = weight[c];
  }
  var b: f32 = 0.0;
  if (params.has_bias == 1u) {
    b = bias[c];
  }
  out[i] = (inp[i] - mean[c]) * inv * g + b;
}
)";

inline constexpr uint32_t kBatchNormWorkgroupSizeX = 256;
inline constexpr uint32_t kBatchNormWorkgroupSizeY = 1;
inline constexpr uint32_t kBatchNormWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
