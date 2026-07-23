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

// @generated from gather.wgsl - DO NOT EDIT.
// wgsl-sha256: cd19a9ed2753e2a97e96fb90f7a72e8f40d08feb6c84fbed7f2c52e71d77a03c
inline constexpr const char* kGatherWGSL = R"(
@group(0) @binding(0) var<storage, read> self_: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: vec4<u32>,
  strides: vec4<u32>,
}
@group(0) @binding(3) var<uniform> out_meta: TensorMeta;
@group(0) @binding(4) var<uniform> self_meta: TensorMeta;

struct GatherParams {
  dim: u32,
}
@group(0) @binding(5) var<uniform> params: GatherParams;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= out_meta.numel) {
        return;
    }
    var rem = o;
    var self_idx: u32 = 0u;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        let c = rem / out_meta.strides[d];
        rem = rem % out_meta.strides[d];
        var coord = c;
        if (d == params.dim) {
            coord = u32(indices[o]);
        }
        self_idx = self_idx + coord * self_meta.strides[d];
    }
    output[o] = self_[self_idx];
}
)";

inline constexpr uint32_t kGatherWorkgroupSizeX = 256;
inline constexpr uint32_t kGatherWorkgroupSizeY = 1;
inline constexpr uint32_t kGatherWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
