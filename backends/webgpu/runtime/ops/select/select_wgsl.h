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

// @generated from select.wgsl - DO NOT EDIT.
// wgsl-sha256: 200cf5a8190045aa0562e782f01c1cfaf9681f30f679f5112ccc3d347a0ed8df
inline constexpr const char* kSelectWGSL = R"(
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

struct Params {
  dim: u32,
  index: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_bufi = gid.x;
    if (out_bufi >= out_meta.numel) {
        return;
    }

    // Gather: out dim od -> in dim (od if od < dim else od+1); sel dim = index.
    var rem = out_bufi;
    var in_bufi: u32 = params.index * in_meta.strides[params.dim];
    for (var od: u32 = 0u; od < out_meta.ndim; od = od + 1u) {
        let coord = rem / out_meta.strides[od];
        rem = rem % out_meta.strides[od];
        var id = od;
        if (od >= params.dim) {
            id = od + 1u;
        }
        in_bufi = in_bufi + coord * in_meta.strides[id];
    }
    output[out_bufi] = input[in_bufi];
}
)";

inline constexpr uint32_t kSelectWorkgroupSizeX = 64;
inline constexpr uint32_t kSelectWorkgroupSizeY = 1;
inline constexpr uint32_t kSelectWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
