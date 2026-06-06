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

// @generated from update_cache.wgsl - DO NOT EDIT.
// wgsl-sha256: 4a8cf6bf45bab3f3718dc3505e8f16bfea7130da2749010e3d0dffa631f84d8e
inline constexpr const char* kUpdateCacheWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_cache: array<f32>;
@group(0) @binding(1) var<storage, read> t_value: array<f32>;

struct Params {
  numel: u32,
  dst_offset: u32,
  cache_numel: u32,
  _pad0: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numel) {
    return;
  }
  if (params.dst_offset + i >= params.cache_numel) {
    return;
  }
  t_cache[params.dst_offset + i] = t_value[i];
}
)";

inline constexpr uint32_t kUpdateCacheWorkgroupSizeX = 64;
inline constexpr uint32_t kUpdateCacheWorkgroupSizeY = 1;
inline constexpr uint32_t kUpdateCacheWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
