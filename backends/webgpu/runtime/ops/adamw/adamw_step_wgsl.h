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

// @generated from adamw_step.wgsl - DO NOT EDIT.
// wgsl-sha256: 0957c04168872db5e2b39cf5f26beefba27f3b5514ec69cece4de5145e97156f
inline constexpr const char* kAdamwStepWGSL = R"(

@group(0) @binding(0) var<storage, read_write> t_param: array<f32>;
@group(0) @binding(1) var<storage, read_write> t_m: array<f32>;
@group(0) @binding(2) var<storage, read_write> t_v: array<f32>;
@group(0) @binding(3) var<storage, read> t_grad: array<f32>;

struct Params {
  numel: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps: f32,
  weight_decay: f32,
  bias_correction1: f32,
  bias_correction2: f32,
  _pad3: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numel) {
    return;
  }
  let g = t_grad[i];
  var p = t_param[i];
  p = p - params.lr * params.weight_decay * p;
  let m = params.beta1 * t_m[i] + (1.0 - params.beta1) * g;
  let v = params.beta2 * t_v[i] + (1.0 - params.beta2) * g * g;
  t_m[i] = m;
  t_v[i] = v;
  let mhat = m / params.bias_correction1;
  let vhat = v / params.bias_correction2;
  t_param[i] = p - params.lr * mhat / (sqrt(vhat) + params.eps);
}
)";

inline constexpr uint32_t kAdamwStepWorkgroupSizeX = 64;
inline constexpr uint32_t kAdamwStepWorkgroupSizeY = 1;
inline constexpr uint32_t kAdamwStepWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
