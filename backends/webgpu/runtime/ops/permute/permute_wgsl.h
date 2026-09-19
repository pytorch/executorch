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

// @generated from permute.wgsl - DO NOT EDIT.
// wgsl-sha256: ec71705cb9feb46edd5398986cfe4f5a40028b8d5a603f0d2bf06d056a844c26
inline constexpr const char* kPermuteWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: array<vec4<u32>, 2>,
  strides: array<vec4<u32>, 2>,
}
@group(0) @binding(2) var<uniform> out_meta: TensorMeta;
@group(0) @binding(3) var<uniform> in_meta: TensorMeta;

struct Params {
  perm: array<vec4<u32>, 2>,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // 2D-folded flat index (lifts the 65535 1D-dispatch cap for large numel).
    let out_bufi = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (out_bufi >= out_meta.numel) {
        return;
    }

    // Gather: out coord d -> in coord perm[d] (Vulkan permute_buffer.glsl).
    var rem = out_bufi;
    var in_bufi: u32 = 0u;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        let coord = rem / out_meta.strides[d >> 2u][d & 3u];
        rem = rem % out_meta.strides[d >> 2u][d & 3u];
        let p = params.perm[d >> 2u][d & 3u];
        in_bufi = in_bufi + coord * in_meta.strides[p >> 2u][p & 3u];
    }
    output[out_bufi] = input[in_bufi];
}
)";

inline constexpr uint32_t kPermuteWorkgroupSizeX = 64;
inline constexpr uint32_t kPermuteWorkgroupSizeY = 1;
inline constexpr uint32_t kPermuteWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
