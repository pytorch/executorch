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

// @generated from slice.wgsl - DO NOT EDIT.
// wgsl-sha256: 6639d985420d43a67de0847749918ab6216e0785399bdcae737d49b81c773526
inline constexpr const char* kSliceWGSL = R"(
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
  start: u32,
  step: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_bufi = gid.x;
    if (out_bufi >= out_meta.numel) {
        return;
    }

    // Gather: out_bufi -> in_bufi, sliced dim coord = start + coord*step.
    var rem = out_bufi;
    var in_bufi: u32 = 0u;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        let coord = rem / out_meta.strides[d];
        rem = rem % out_meta.strides[d];
        var in_coord = coord;
        if (d == params.dim) {
            in_coord = params.start + coord * params.step;
        }
        in_bufi = in_bufi + in_coord * in_meta.strides[d];
    }
    output[out_bufi] = input[in_bufi];
}
)";

inline constexpr uint32_t kSliceWorkgroupSizeX = 64;
inline constexpr uint32_t kSliceWorkgroupSizeY = 1;
inline constexpr uint32_t kSliceWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
