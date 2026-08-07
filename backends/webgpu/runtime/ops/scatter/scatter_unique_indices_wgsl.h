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

// @generated from scatter_unique_indices.wgsl - DO NOT EDIT.
// wgsl-sha256: 1db5230fb370bd30c52e857ca7134a5dfb6e04c381e3a2670f4fdcfe1d625d39
inline constexpr const char* kScatterUniqueIndicesWGSL = R"(
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read> source: array<u32>;

// One invocation per SOURCE element, replacing the shipping scatter.wgsl's
// single one-thread workgroup that walked all 4096 writes serially
// (measured 393.216 us = 6 timestamp quanta per dispatch).
//
// The shipping kernel is last-write-wins over ascending i, mirroring portable
// ExecuTorch op_scatter.cpp. This kernel is arbitrary-write-wins. The two are
// bit-identical iff the destinations are pairwise DISTINCT, which they are by
// construction here: `indices` is `token_ordering[topk_indices]`, a gather of
// 32 DISTINCT rows out of a [2048,128] permutation of [0,262144), so no two
// source elements can name the same destination. The out-of-range guard is
// kept verbatim, and is per-invocation, so negative / >= vocab entries are
// dropped exactly as before.
const WG: u32 = 64u;

@compute @workgroup_size(WG, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= 4096u) {
        return;
    }
    let destination = indices[i];
    if (destination >= 0 && destination < 262144) {
        output[u32(destination)] = source[i];
    }
}
)";

inline constexpr uint32_t kScatterUniqueIndicesWorkgroupSizeX = 64;
inline constexpr uint32_t kScatterUniqueIndicesWorkgroupSizeY = 1;
inline constexpr uint32_t kScatterUniqueIndicesWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
