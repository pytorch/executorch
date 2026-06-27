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

// @generated from cat.wgsl - DO NOT EDIT.
// wgsl-sha256: d1fcb4da7e32c6295b80d581c093b78d0a4b43a972fe2d5d9d94c4f9ae459f4c
inline constexpr const char* kCatWGSL = R"(
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
  concat_dim: u32,
  off_k: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let in_bufi = gid.x;
    if (in_bufi >= in_meta.numel) {
        return;
    }

    // Scatter: in coord -> out coord, concat dim shifted by off_k (Vulkan concat).
    var rem = in_bufi;
    var out_bufi: u32 = 0u;
    for (var d: u32 = 0u; d < in_meta.ndim; d = d + 1u) {
        let coord = rem / in_meta.strides[d];
        rem = rem % in_meta.strides[d];
        var out_coord = coord;
        if (d == params.concat_dim) {
            out_coord = coord + params.off_k;
        }
        out_bufi = out_bufi + out_coord * out_meta.strides[d];
    }
    output[out_bufi] = input[in_bufi];
}
)";

inline constexpr uint32_t kCatWorkgroupSizeX = 64;
inline constexpr uint32_t kCatWorkgroupSizeY = 1;
inline constexpr uint32_t kCatWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
