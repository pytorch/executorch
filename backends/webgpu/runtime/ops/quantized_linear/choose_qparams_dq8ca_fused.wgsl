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
@group(0) @binding(2) var<storage, read_write> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> zero_points: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 256u;
const QUANT_MIN: i32 = -128;
const QUANT_MAX: i32 = 127;
const SMALL_SCALE_THRESHOLD: f32 = 6.1e-5;

var<workgroup> min_values: array<f32, WG>;
var<workgroup> max_values: array<f32, WG>;

fn reciprocal_is_infinite(value: f32) -> bool {
  // WGSL has no portable isinf builtin. This exponent/mantissa check is the
  // exact f32 equivalent used for Vulkan's isinf(1.0 / scale) condition.
  return (bitcast<u32>(1.0 / value) & 0x7fffffffu) == 0x7f800000u;
}

fn calculate_scale_and_zero_point(
    input_min: f32,
    input_max: f32) -> vec2<f32> {
  var min_value = min(input_min, 0.0);
  var max_value = max(input_max, 0.0);
  let qmin = f32(QUANT_MIN);
  let qmax = f32(QUANT_MAX);
  var scale = (max_value - min_value) / (qmax - qmin);
  if (scale == 0.0 || reciprocal_is_infinite(scale)) {
    scale = 0.1;
  }
  if (scale < SMALL_SCALE_THRESHOLD) {
    let original_scale = scale;
    scale = SMALL_SCALE_THRESHOLD;
    if (min_value == 0.0) {
      max_value = SMALL_SCALE_THRESHOLD * (qmax - qmin);
    } else if (max_value == 0.0) {
      min_value = -SMALL_SCALE_THRESHOLD * (qmax - qmin);
    } else {
      let amplifier = SMALL_SCALE_THRESHOLD / original_scale;
      min_value *= amplifier;
      max_value *= amplifier;
    }
  }

  let zero_point_from_min = qmin - min_value / scale;
  let zero_point_from_max = qmax - max_value / scale;
  let zero_point_from_min_error = abs(qmin) - abs(min_value / scale);
  let zero_point_from_max_error = abs(qmax) - abs(max_value / scale);
  var initial_zero_point = zero_point_from_max;
  if (zero_point_from_min_error < zero_point_from_max_error) {
    initial_zero_point = zero_point_from_min;
  }
  var nudged_zero_point: i32;
  if (initial_zero_point < qmin) {
    nudged_zero_point = QUANT_MIN;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = QUANT_MAX;
  } else {
    nudged_zero_point = i32(round(initial_zero_point));
  }
  return vec2<f32>(scale, f32(nudged_zero_point));
}

fn load_zero_point(row: u32) -> i32 {
  let word = atomicLoad(&zero_points[row / 4u]);
  let byte = (word >> ((row % 4u) * 8u)) & 0xffu;
  return select(i32(byte), i32(byte) - 256, byte >= 128u);
}

@compute @workgroup_size(WG, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ngrp: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>) {
  let lane = lid.x;
  var row = wid.x;
  loop {
    if (row >= params.num_rows) {
      break;
    }
    let row_start = row * params.row_width;

    var local_min = input[row_start];
    var local_max = input[row_start];
    var col = lane;
    while (col < params.row_width) {
      let value = input[row_start + col];
      local_min = min(local_min, value);
      local_max = max(local_max, value);
      col += WG;
    }

    min_values[lane] = local_min;
    max_values[lane] = local_max;
    workgroupBarrier();

    var stride = WG / 2u;
    while (stride > 0u) {
      if (lane < stride) {
        min_values[lane] = min(min_values[lane], min_values[lane + stride]);
        max_values[lane] = max(max_values[lane], max_values[lane + stride]);
      }
      workgroupBarrier();
      stride /= 2u;
    }

    if (lane == 0u) {
      let qparams =
          calculate_scale_and_zero_point(min_values[0], max_values[0]);
      scales[row] = qparams.x;
      let zero_point_byte = u32(i32(qparams.y)) & 0xffu;
      let pack = row / 4u;
      let shift = (row % 4u) * 8u;
      var clear_mask = ~(0xffu << shift);
      if (row % 4u == 0u) {
        for (var tail = 1u; tail < 4u; tail++) {
          if (row + tail >= params.num_rows) {
            clear_mask &= ~(0xffu << (tail * 8u));
          }
        }
      }
      atomicAnd(&zero_points[pack], clear_mask);
      atomicOr(&zero_points[pack], zero_point_byte << shift);
    }

    storageBarrier();
    workgroupBarrier();

    let scale = scales[row];
    let zero_point = load_zero_point(row);
    var elem = lane;
    while (elem < params.row_width) {
      let index = row_start + elem;
      let quantized = clamp(
          round(input[index] * (1.0 / scale)) + f32(zero_point),
          -128.0,
          127.0);
      output[index] = (quantized - f32(zero_point)) * scale;
      elem += WG;
    }

    workgroupBarrier();
    row += ngrp.x;
  }
}
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
