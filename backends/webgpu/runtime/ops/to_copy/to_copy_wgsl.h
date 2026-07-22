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

// @generated from to_copy.wgsl - DO NOT EDIT.
// wgsl-sha256: a368bcd110feb74b4a7cb51297386fa1abc85b5dc7f5815bb1c4ff0e6c0bf240
inline constexpr const char* kToCopyWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;

struct Params {
  numel: u32,
  convert_mode: u32, // 0 = same-dtype raw copy, 1 = int->float, 2 = float->int
  pad0: u32,
  pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.numel) {
        return;
    }
    // Both buffers are bound as raw i32 words so no f32 load/store ever touches
    // int32 bytes: an f32 load/store of an int NaN bit-pattern (e.g. -1 =
    // 0xFFFFFFFF) can be canonicalized and corrupt the value. int<->float
    // _to_copy must CONVERT, not byte-reinterpret; mode 0 is a bit-exact copy.
    if (params.convert_mode == 1u) {
        // int -> float: convert, store the f32 bits into the (float) output.
        output[idx] = bitcast<i32>(f32(input[idx]));
    } else if (params.convert_mode == 2u) {
        // float -> int: reinterpret input bits as f32, convert to int.
        output[idx] = i32(bitcast<f32>(input[idx]));
    } else {
        output[idx] = input[idx];
    }
}
)";

inline constexpr uint32_t kToCopyWorkgroupSizeX = 256;
inline constexpr uint32_t kToCopyWorkgroupSizeY = 1;
inline constexpr uint32_t kToCopyWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
