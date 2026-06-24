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

// @generated from sigmoid.wgsl - DO NOT EDIT.
// wgsl-sha256: 73a26ddce78d1cbd6cbb0c586791b338153cea9af13790dc1400516128a4c278
inline constexpr const char* kSigmoidWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: vec4<u32>,
  strides: vec4<u32>,
}
@group(0) @binding(2) var<uniform> out_meta: TensorMeta;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= out_meta.numel) {
        return;
    }
    output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}
)";

inline constexpr uint32_t kSigmoidWorkgroupSizeX = 64;
inline constexpr uint32_t kSigmoidWorkgroupSizeY = 1;
inline constexpr uint32_t kSigmoidWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
