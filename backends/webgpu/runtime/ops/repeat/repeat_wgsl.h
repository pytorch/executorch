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

// @generated from repeat.wgsl - DO NOT EDIT.
// wgsl-sha256: ae77e3d6846cd185e96b0fcae4b4bcee2096e659df1a7ba9d0129beb24d754c7
inline constexpr const char* kRepeatWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: vec4<u32>,
  strides: vec4<u32>,
}
@group(0) @binding(2) var<uniform> out_meta: TensorMeta;
@group(0) @binding(3) var<uniform> in_meta: TensorMeta;

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

    // Tile: gather in_coord[d] = out_coord[d] % in_size (Vulkan repeat_buffer).
    // in is right-aligned into out; the leading offset dims are pure repeats.
    let offset = out_meta.ndim - in_meta.ndim;
    var rem = out_bufi;
    var in_bufi: u32 = 0u;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        let out_coord = rem / out_meta.strides[d];
        rem = rem % out_meta.strides[d];
        if (d >= offset) {
            let in_d = d - offset;
            let in_coord = out_coord % in_meta.sizes[in_d];
            in_bufi = in_bufi + in_coord * in_meta.strides[in_d];
        }
    }
    output[out_bufi] = input[in_bufi];
}
)";

inline constexpr uint32_t kRepeatWorkgroupSizeX = 64;
inline constexpr uint32_t kRepeatWorkgroupSizeY = 1;
inline constexpr uint32_t kRepeatWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
