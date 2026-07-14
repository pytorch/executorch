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

// @generated from binary_op.wgsl - DO NOT EDIT.
// wgsl-sha256: 496b343cef6838c8316686916a1c8fd971afc7a9abfc9abca4eb28db60611371
inline constexpr const char* kBinarySubWGSL = R"(
@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: vec4<u32>,
  strides: vec4<u32>,
}
@group(0) @binding(3) var<uniform> out_meta: TensorMeta;
@group(0) @binding(4) var<uniform> in1_meta: TensorMeta;
@group(0) @binding(5) var<uniform> in2_meta: TensorMeta;

override wg_size: u32 = 64u;
override alpha: f32 = 1.0;

fn op(a: f32, b: f32) -> f32 {
  return a - alpha * b;
}

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // 2D-folded flat index (lifts the 65535 1D-dispatch cap for large numel).
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= out_meta.numel) {
        return;
    }

    var same = true;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        if (in1_meta.sizes[d] != out_meta.sizes[d] ||
            in2_meta.sizes[d] != out_meta.sizes[d]) {
            same = false;
        }
    }
    if (same) {
        output[idx] = op(input1[idx], input2[idx]);
        return;
    }

    var rem = idx;
    var l1: u32 = 0u;
    var l2: u32 = 0u;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        let coord = rem / out_meta.strides[d];
        rem = rem % out_meta.strides[d];
        l1 = l1 + min(coord, in1_meta.sizes[d] - 1u) * in1_meta.strides[d];
        l2 = l2 + min(coord, in2_meta.sizes[d] - 1u) * in2_meta.strides[d];
    }
    output[idx] = op(input1[l1], input2[l2]);
}
)";

inline constexpr uint32_t kBinarySubWorkgroupSizeX = 64;
inline constexpr uint32_t kBinarySubWorkgroupSizeY = 1;
inline constexpr uint32_t kBinarySubWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
