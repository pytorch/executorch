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

// @generated from index_select.wgsl - DO NOT EDIT.
// wgsl-sha256: 63d3a49ddd72a67b05bb3d03a05ba375e4e3b096a47ec67406d05c3ddc9eb241
inline constexpr const char* kIndexSelectWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<i32>;

struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: vec4<u32>,
  strides: vec4<u32>,
}
@group(0) @binding(3) var<uniform> out_meta: TensorMeta;
@group(0) @binding(4) var<uniform> in_meta: TensorMeta;

struct Params {
  info: vec4<u32>,
}
@group(0) @binding(5) var<uniform> params: Params;

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

    // Gather: in_coord = out_coord, but the dim coord is remapped by index[].
    let dim = params.info.x;
    var rem = out_bufi;
    var in_bufi: u32 = 0u;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        let coord = rem / out_meta.strides[d];
        rem = rem % out_meta.strides[d];
        var in_coord = coord;
        if (d == dim) {
            in_coord = u32(index[coord]);
        }
        in_bufi = in_bufi + in_coord * in_meta.strides[d];
    }
    output[out_bufi] = input[in_bufi];
}
)";

inline constexpr uint32_t kIndexSelectWorkgroupSizeX = 64;
inline constexpr uint32_t kIndexSelectWorkgroupSizeY = 1;
inline constexpr uint32_t kIndexSelectWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
