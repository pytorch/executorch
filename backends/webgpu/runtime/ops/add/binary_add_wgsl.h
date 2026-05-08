/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace executorch {
namespace backends {
namespace webgpu {

// WGSL shader source for element-wise add: output = input1 + alpha * input2
inline constexpr const char* kBinaryAddWGSL = R"(
@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
  alpha: f32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.num_elements) {
        return;
    }
    output[idx] = input1[idx] + params.alpha * input2[idx];
}
)";

inline constexpr uint32_t kBinaryAddWorkgroupSize = 256;

} // namespace webgpu
} // namespace backends
} // namespace executorch
