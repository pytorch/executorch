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

// @generated from index.wgsl - DO NOT EDIT.
// wgsl-sha256: daed48e60bfcf2b7420d277576d794137d3bff383aef4f68464c98c8a7235c8e
inline constexpr const char* kIndexWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<i32>;

struct Params {
  numel: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_bufi = gid.x;
    if (out_bufi >= params.numel) {
        return;
    }

    // 1D-self gather out[i]=self[index[i]] (mirrors Vulkan index_tensor_buffer.glsl).
    let i = index[out_bufi];
    output[out_bufi] = input[u32(i)];
}
)";

inline constexpr uint32_t kIndexWorkgroupSizeX = 64;
inline constexpr uint32_t kIndexWorkgroupSizeY = 1;
inline constexpr uint32_t kIndexWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
