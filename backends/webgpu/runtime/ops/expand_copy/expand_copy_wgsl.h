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

// @generated from expand_copy.wgsl - DO NOT EDIT.
// wgsl-sha256: ad996b7fd6eca5c5773715af3a9a117da83ef522042eb0ee918a623096417815
inline constexpr const char* kExpandCopyWGSL = R"(
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= out_meta.numel) {
        return;
    }
    var rem = idx;
    var l: u32 = 0u;
    for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
        let coord = rem / out_meta.strides[d];
        rem = rem % out_meta.strides[d];
        l = l + min(coord, in_meta.sizes[d] - 1u) * in_meta.strides[d];
    }
    output[idx] = input[l];
}
)";

inline constexpr uint32_t kExpandCopyWorkgroupSizeX = 64;
inline constexpr uint32_t kExpandCopyWorkgroupSizeY = 1;
inline constexpr uint32_t kExpandCopyWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
