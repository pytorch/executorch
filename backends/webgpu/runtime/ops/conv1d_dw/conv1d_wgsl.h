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

// @generated from conv1d.wgsl - DO NOT EDIT.
// wgsl-sha256: 7bc955b7f43473aab96222e7a1228973b65e38e2e6a19e2b9793e0ed7f3768d9
inline constexpr const char* kConv1dWGSL = R"(
override wg_size: u32 = 64u;

struct Params {
  in_channels: u32,
  out_channels: u32,
  in_len: u32,
  out_len: u32,
  kernel_size: u32,
  stride: u32,
  padding: u32,
  dilation: u32,
  numel: u32,
  has_bias: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.numel) {
    return;
  }

  let out_t = idx % params.out_len;
  let out_c = (idx / params.out_len) % params.out_channels;
  let batch = idx / (params.out_channels * params.out_len);
  var sum = 0.0;

  for (var in_c = 0u; in_c < params.in_channels; in_c = in_c + 1u) {
    for (var k = 0u; k < params.kernel_size; k = k + 1u) {
      let in_t = i32(out_t * params.stride + k * params.dilation) -
          i32(params.padding);
      if (in_t >= 0 && in_t < i32(params.in_len)) {
        let input_idx =
            (batch * params.in_channels + in_c) * params.in_len + u32(in_t);
        let weight_idx =
            (out_c * params.in_channels + in_c) * params.kernel_size + k;
        sum = fma(input[input_idx], weight[weight_idx], sum);
      }
    }
  }
  if (params.has_bias != 0u) {
    sum = sum + bias[out_c];
  }
  output[idx] = sum;
}
)";

inline constexpr uint32_t kConv1dWorkgroupSizeX = 64;
inline constexpr uint32_t kConv1dWorkgroupSizeY = 1;
inline constexpr uint32_t kConv1dWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
