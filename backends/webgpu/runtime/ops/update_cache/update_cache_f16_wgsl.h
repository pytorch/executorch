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

// @generated from update_cache_f16.wgsl - DO NOT EDIT.
// wgsl-sha256: e115bcc12d0b0d09e9f72e1f63fee21c0ce56f787dfbd728b9f4adebb1bf5630
inline constexpr const char* kUpdateCacheF16WGSL = R"(
enable f16;

// f16 KV-cache variant of update_cache.wgsl: cache stored as f16, source f32.
@group(0) @binding(0) var<storage, read_write> t_cache: array<f16>;
@group(0) @binding(1) var<storage, read> t_value: array<f32>;

struct Params {
  numel: u32,
  dst_offset: u32,
  cache_numel: u32,
  _pad0: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numel) {
    return;
  }
  if (params.dst_offset + i >= params.cache_numel) {
    return;
  }
  t_cache[params.dst_offset + i] = f16(t_value[i]);
}
)";

inline constexpr uint32_t kUpdateCacheF16WorkgroupSizeX = 256;
inline constexpr uint32_t kUpdateCacheF16WorkgroupSizeY = 1;
inline constexpr uint32_t kUpdateCacheF16WorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
