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

// @generated from rms_norm_vec4_add_scale.wgsl - DO NOT EDIT.
// wgsl-sha256: 13387491431ae152eedd2f3e7e18cf4b7cd1bc24d6c6df4607efe8f2eb97eb1a
inline constexpr const char* kRmsNormVec4AddScaleWGSL = R"(
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

@group(0) @binding(0) var<storage, read_write> t_out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> t_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> t_weight: array<vec4<f32>>;

// Byte-identical to the unfused rms_norm_vec4 Params, so RmsNorm.cpp's existing
// resize hook keeps rewriting the SAME uniform buffer with the SAME 16 bytes.
struct Params {
  num_rows: u32,
  row_width: u32,
  epsilon: f32,
  _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@group(0) @binding(4) var<storage, read> t_resid: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> t_addout: array<vec4<f32>>;

@group(0) @binding(6) var<storage, read> t_scale: array<f32>;
@group(0) @binding(7) var<storage, read_write> t_scaleout: array<vec4<f32>>;

// Fixed workgroup size (Apple scalar ALU: no override loop bounds). 64 is the
// value clamp_workgroup_size_pow2 already yields for the unfused kernel, so the
// tree reduction visits the identical strides in the identical order and the
// normalized result is bit-identical.
const wg_size: u32 = 64u;

var<workgroup> shared_sum: array<f32, 64>;

fn reduce_shared(worker_id: u32) {
  workgroupBarrier();
  var stride: u32 = wg_size / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (worker_id < stride) {
      shared_sum[worker_id] = shared_sum[worker_id] + shared_sum[worker_id + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
}

// rms_norm_vec4 + the following aten.add.Tensor + the following
// aten.mul.Tensor by a single-element tensor (binary_mul's broadcast path
// collapses that operand to input2[0]). All three stores are kept, so no
// consumer liveness assumption is made: this is a dispatch merge, not a graph
// rewrite.
// The host guards alpha == 1, and IEEE addition is commutative, so both
// binary_add operand orders (resid + 1.0*n and n + 1.0*resid) collapse to the
// single expression `r + n` -- bit-identically, and under any fma contraction,
// because fma(1, n, r) == n + r exactly. There is therefore no alpha and no
// operand-order flag to carry.
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>) {
  // 2D-fold: rows can exceed the 65535 per-dim cap (QK-norm at prefill).
  let row_idx = wid.x + wid.y * num_workgroups.x;
  let worker_id = lid.x;

  if (row_idx >= params.num_rows) {
    return;
  }

  let rw4 = params.row_width / 4u;
  let base4 = row_idx * rw4;

  var local_sq_sum: f32 = 0.0;
  var x4: u32 = worker_id;
  loop {
    if (x4 >= rw4) {
      break;
    }
    let v = t_in[base4 + x4];
    local_sq_sum = local_sq_sum + dot(v, v);
    x4 = x4 + wg_size;
  }

  shared_sum[worker_id] = local_sq_sum;
  reduce_shared(worker_id);

  let mean_sq = shared_sum[0] / f32(params.row_width);
  let rstd = inverseSqrt(mean_sq + params.epsilon);

  let s = t_scale[0];

  x4 = worker_id;
  loop {
    if (x4 >= rw4) {
      break;
    }
    let r = t_resid[base4 + x4];
    let n = t_in[base4 + x4] * rstd * t_weight[x4];
    t_out[base4 + x4] = n;
    let a = r + n;
    t_addout[base4 + x4] = a;
    t_scaleout[base4 + x4] = a * s;
    x4 = x4 + wg_size;
  }
}
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
)";

inline constexpr uint32_t kRmsNormVec4AddScaleWorkgroupSizeX = 64;
inline constexpr uint32_t kRmsNormVec4AddScaleWorkgroupSizeY = 1;
inline constexpr uint32_t kRmsNormVec4AddScaleWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
