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
// wgsl-sha256: fa1afd9596cc785e7db906d323caf84aaa97f4b850c343cc4e73be232455fe56
inline constexpr const char* kCompareGeIntWGSL = R"(
@group(0) @binding(0) var<storage, read> input: array<i32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

// int_scalar / float_compare are read by the *_int variants only: an Int
// scalar compares in int32, a Double one in fp32 (PyTorch promotes an integer
// tensor against a float scalar to the default float dtype).
struct Params {
  num_elements: u32,
  scalar: f32,
  int_scalar: i32,
  float_compare: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 64u;

// Per-variant predicate substituted from boolean_op.yaml (compare / logical_not).
fn elem_bool(i: u32) -> bool {
  return select(input[i] >= params.int_scalar, f32(input[i]) >= params.scalar, params.float_compare != 0u);
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

inline constexpr uint32_t kCompareGeIntWorkgroupSizeX = 64;
inline constexpr uint32_t kCompareGeIntWorkgroupSizeY = 1;
inline constexpr uint32_t kCompareGeIntWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
