// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

struct Params {
  num_elements: u32,
  num_rows: u32,
  row_width: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> zero_points: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

fn load_zero_point(row: u32) -> i32 {
  let word = zero_points[row / 4u];
  let byte = (word >> ((row % 4u) * 8u)) & 0xffu;
  return select(i32(byte), i32(byte) - 256, byte >= 128u);
}

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>) {
  let index = gid.x + gid.y * nwg.x * wg_size;
  if (index >= params.num_elements) {
    return;
  }
  let row = index / params.row_width;
  if (row >= params.num_rows) {
    return;
  }
  let scale = scales[row];
  let zero_point = load_zero_point(row);
  let quantized = clamp(
      round(input[index] * (1.0 / scale)) + f32(zero_point), -128.0, 127.0);
  output[index] = (quantized - f32(zero_point)) * scale;
}
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
