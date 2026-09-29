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

// @generated from embedding.wgsl - DO NOT EDIT.
// wgsl-sha256: e7c94da588a9dd40c0e1be7d3c6ef33310c71390c57f6f330ae23a18866d0ea7
inline constexpr const char* kEmbeddingWGSL = R"(
@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
  dim: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
    if (idx >= params.num_elements) {
        return;
    }
    let row = idx / params.dim;
    let col = idx % params.dim;
    let row_id = u32(indices[row]);
    output[idx] = weight[row_id * params.dim + col];
}
)";

inline constexpr uint32_t kEmbeddingWorkgroupSizeX = 256;
inline constexpr uint32_t kEmbeddingWorkgroupSizeY = 1;
inline constexpr uint32_t kEmbeddingWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
