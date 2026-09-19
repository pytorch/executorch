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

// @generated from binary_add_int.wgsl - DO NOT EDIT.
// wgsl-sha256: 7164f926e44c8933429818a770b44c3a1644d56531aae80a5527310c7b2de0b6
inline constexpr const char* kBinaryAddIntWGSL = R"(
@group(0) @binding(0) var<storage, read> input1: array<i32>;
@group(0) @binding(1) var<storage, read> input2: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<i32>;

struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: array<vec4<u32>, 2>,
  strides: array<vec4<u32>, 2>,
}
@group(0) @binding(3) var<uniform> out_meta: TensorMeta;
@group(0) @binding(4) var<uniform> in1_meta: TensorMeta;
@group(0) @binding(5) var<uniform> in2_meta: TensorMeta;

override wg_size: u32 = 256u;
// add.Tensor alpha; read once from the graph and fixed at build (never resized).
override alpha: i32 = 1;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // 2D-folded flat index (lifts the 65535 1D-dispatch cap for large numel).
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= out_meta.numel) {
        return;
    }

    // Fast path: every input dim matches the output dim -> elementwise.
    var same = true;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        if (in1_meta.sizes[d >> 2u][d & 3u] != out_meta.sizes[d >> 2u][d & 3u] ||
            in2_meta.sizes[d >> 2u][d & 3u] != out_meta.sizes[d >> 2u][d & 3u]) {
            same = false;
        }
    }
    if (same) {
        output[idx] = input1[idx] + alpha * input2[idx];
        return;
    }

    // Broadcast: out idx -> per-input coord (clamp size-1 dims), relinearize.
    var rem = idx;
    var l1: u32 = 0u;
    var l2: u32 = 0u;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        let coord = rem / out_meta.strides[d >> 2u][d & 3u];
        rem = rem % out_meta.strides[d >> 2u][d & 3u];
        l1 = l1 + min(coord, in1_meta.sizes[d >> 2u][d & 3u] - 1u) * in1_meta.strides[d >> 2u][d & 3u];
        l2 = l2 + min(coord, in2_meta.sizes[d >> 2u][d & 3u] - 1u) * in2_meta.strides[d >> 2u][d & 3u];
    }
    output[idx] = input1[l1] + alpha * input2[l2];
}
)";

inline constexpr uint32_t kBinaryAddIntWorkgroupSizeX = 256;
inline constexpr uint32_t kBinaryAddIntWorkgroupSizeY = 1;
inline constexpr uint32_t kBinaryAddIntWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
