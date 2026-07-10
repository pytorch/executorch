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
// wgsl-sha256: 7ccb183b182070ea17138a4c06ec0e3419893e8312a6769bd945acd7223f9e9c
inline constexpr const char* kEmbeddingWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_indices: array<i32>;
@group(0) @binding(2) var<storage, read> t_weight: array<f32>;

struct Params {
  embed_dim: u32,
  num_elements: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 64u;

// fp32 embedding row-gather: out[row, col] = weight[indices[row], col].
// One thread per output element. int32 indices (the backend's index convention,
// see index.wgsl / embedding_q4gsw).
@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.num_elements) {
    return;
  }
  let row = idx / params.embed_dim;
  let col = idx % params.embed_dim;
  let token = u32(t_indices[row]);
  t_out[idx] = t_weight[token * params.embed_dim + col];
}
)";

inline constexpr uint32_t kEmbeddingWorkgroupSizeX = 64;
inline constexpr uint32_t kEmbeddingWorkgroupSizeY = 1;
inline constexpr uint32_t kEmbeddingWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
