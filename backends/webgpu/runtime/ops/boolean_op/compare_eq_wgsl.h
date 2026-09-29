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

// @generated from boolean_op.wgsl - DO NOT EDIT.
// wgsl-sha256: 2a3c203d0255086a6e67a9e7cb08538858e8e6cb6a5542f6acebf04b8ccca6b7
inline constexpr const char* kCompareEqWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
  num_elements: u32,
  scalar: f32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

// Per-variant predicate substituted from boolean_op.yaml (compare / logical_not).
fn elem_bool(i: u32) -> bool {
  return input[i] == params.scalar;
}

// One thread per output u32 word packs 4 bool bytes -> no inter-thread race.
@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_idx = gid.x;
    let n_words = (params.num_elements + 3u) / 4u;
    if (word_idx >= n_words) {
        return;
    }
    var packed: u32 = 0u;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let i = word_idx * 4u + j;
        if (i < params.num_elements) {
            if (elem_bool(i)) {
                packed = packed | (1u << (j * 8u));
            }
        }
    }
    output[word_idx] = packed;
}
)";

inline constexpr uint32_t kCompareEqWorkgroupSizeX = 64;
inline constexpr uint32_t kCompareEqWorkgroupSizeY = 1;
inline constexpr uint32_t kCompareEqWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
