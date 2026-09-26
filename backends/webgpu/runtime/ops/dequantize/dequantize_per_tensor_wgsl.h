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

// @generated from dequantize_per_tensor.wgsl - DO NOT EDIT.
// wgsl-sha256: b9e24b5f3b57f6eb842838399985ab2de20513319387fa3341624d310687f90e
inline constexpr const char* kDequantizePerTensorWGSL = R"(
@group(0) @binding(0) var<storage, read> t_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> t_out: array<f32>;

struct Params {
  scale: f32,
  zero_point: i32,
  numel: u32,
  pad0: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    // One thread per packed u32 word (4 int8 elems); 2D-fold lifts the 65535 cap.
    let widx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (widx >= params.numel / 4u) {
        return;
    }
    let word = t_in[widx];
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let b = (word >> (j * 8u)) & 0xFFu;
        let s = i32(b ^ 0x80u) - 128;
        t_out[widx * 4u + j] = f32(s - params.zero_point) * params.scale;
    }
}
)";

inline constexpr uint32_t kDequantizePerTensorWorkgroupSizeX = 64;
inline constexpr uint32_t kDequantizePerTensorWorkgroupSizeY = 1;
inline constexpr uint32_t kDequantizePerTensorWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
