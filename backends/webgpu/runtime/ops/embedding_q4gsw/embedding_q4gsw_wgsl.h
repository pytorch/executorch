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

// @generated from embedding_q4gsw.wgsl - DO NOT EDIT.
// wgsl-sha256: 94da1061b49b62556a79020182a4989439a7c51f919e83d577536c5b6d25f487
inline constexpr const char* kEmbeddingQ4gswWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_indices: array<i32>;
@group(0) @binding(2) var<storage, read> t_weight: array<u32>;
@group(0) @binding(3) var<storage, read> t_scales: array<f32>;

struct Params {
  embed_dim: u32,
  blocks_per_row: u32,
  num_indices: u32,
  group_size: u32,
  groups_per_row: u32,
  bytes_per_row: u32,
  total_blocks: u32,
  is_linear_weight: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

// One thread per 32-dim block of one gathered row (flat-buffer weight path).
@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let block = gid.x;
  if (block >= params.total_blocks) {
    return;
  }
  let indices_idx = block / params.blocks_per_row;
  let base_dim = (block % params.blocks_per_row) * 32u;

  // token assumed in-range (mirrors Vulkan; no vocab clamp).
  let token = u32(t_indices[indices_idx]);
  let row_byte_base = token * params.bytes_per_row;
  let out_base = indices_idx * params.embed_dim + base_dim;

  for (var t: u32 = 0u; t < 32u; t = t + 1u) {
    let dim = base_dim + t;
    let byte_idx = row_byte_base + (dim >> 1u);
    let word = t_weight[byte_idx >> 2u];
    let b = (word >> ((byte_idx & 3u) * 8u)) & 0xFFu;
    // Nibble packing depends on is_linear_weight: non-linear maps even dim ->
    // high nibble / odd -> low; linear maps even -> low / odd -> high.
    var nib: u32;
    if (((dim & 1u) == 0u) != (params.is_linear_weight != 0u)) {
      nib = (b >> 4u) & 0x0Fu;  // high nibble
    } else {
      nib = b & 0x0Fu;          // low nibble
    }
    let q = f32(i32(nib) - 8); // +8-shifted on pack; recover signed [-8,7]
    let scale = t_scales[token * params.groups_per_row + dim / params.group_size];
    t_out[out_base + t] = q * scale;
  }
}
)";

inline constexpr uint32_t kEmbeddingQ4gswWorkgroupSizeX = 64;
inline constexpr uint32_t kEmbeddingQ4gswWorkgroupSizeY = 1;
inline constexpr uint32_t kEmbeddingQ4gswWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
