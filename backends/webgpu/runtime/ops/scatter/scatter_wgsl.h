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

// @generated from scatter.wgsl - DO NOT EDIT.
// wgsl-sha256: ac43cd32373a4c93649c8174169b87d0c2c4eaf7be9f318a3c97273f5a1682bb
inline constexpr const char* kScatterWGSL = R"(
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read> source: array<u32>;

@compute @workgroup_size(1)
fn main() {
    for (var i = 0u; i < 4096u; i += 1u) {
        let destination = indices[i];
        if (destination >= 0 && destination < 262144) {
            output[u32(destination)] = source[i];
        }
    }
}
)";

inline constexpr uint32_t kScatterWorkgroupSizeX = 1;
inline constexpr uint32_t kScatterWorkgroupSizeY = 1;
inline constexpr uint32_t kScatterWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
