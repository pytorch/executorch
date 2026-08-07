// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_input: array<f32>;
@group(0) @binding(2) var<storage, read> t_weight: array<u32>;
@group(0) @binding(3) var<storage, read> t_scales: array<f32>;
@group(0) @binding(4) var<storage, read> t_bias: array<f32>;

struct Params {
  M: u32,
  N: u32,
  K: u32,
  K_packed: u32,
  group_size: u32,
  padded_N: u32,
  has_bias: u32,
  _pad: u32,
}
@group(0) @binding(5) var<uniform> params: Params;

const WG: u32 = 64u;
var<workgroup> partial0: array<f32, WG>;
var<workgroup> partial1: array<f32, WG>;
var<workgroup> partial2: array<f32, WG>;
var<workgroup> partial3: array<f32, WG>;
var<workgroup> partial4: array<f32, WG>;
var<workgroup> partial5: array<f32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ngrp: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>) {
  if (params.M != 3u) {
    return;
  }
  let num_pairs = (params.N + 1u) >> 1u;
  let num_words = params.K >> 3u;
  let row_words = params.K_packed >> 2u;
  var pair = wid.x;
  loop {
    if (pair >= num_pairs) {
      break;
    }
    let col0 = pair << 1u;
    let col1 = col0 + 1u;
    let has1 = col1 < params.N;
    let wbase0 = col0 * row_words;
    let wbase1 = col1 * row_words;
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;
    var acc4: f32 = 0.0;
    var acc5: f32 = 0.0;
    var w: u32 = lid.x;
    loop {
      if (w >= num_words) {
        break;
      }
      let k0 = w << 3u;
      let scale_row = (k0 / params.group_size) * params.padded_N;
      let word0 = t_weight[wbase0 + w];
      let scale0 = t_scales[scale_row + col0];
      var word1: u32 = 0u;
      var scale1: f32 = 0.0;
      if (has1) {
        word1 = t_weight[wbase1 + w];
        scale1 = t_scales[scale_row + col1];
      }
      for (var bi: u32 = 0u; bi < 4u; bi = bi + 1u) {
        let kk = bi << 1u;
        let input00 = t_input[k0 + kk];
        let input01 = t_input[k0 + kk + 1u];
        let input10 = t_input[params.K + k0 + kk];
        let input11 = t_input[params.K + k0 + kk + 1u];
        let input20 = t_input[2u * params.K + k0 + kk];
        let input21 = t_input[2u * params.K + k0 + kk + 1u];
        let byte0 = (word0 >> (bi * 8u)) & 0xFFu;
        let lo0 = f32(i32(byte0 & 0x0Fu) - 8);
        let hi0 = f32(i32((byte0 >> 4u) & 0x0Fu) - 8);
        acc0 = acc0 + input00 * lo0 * scale0;
        acc0 = acc0 + input01 * hi0 * scale0;
        acc2 = acc2 + input10 * lo0 * scale0;
        acc2 = acc2 + input11 * hi0 * scale0;
        acc4 = acc4 + input20 * lo0 * scale0;
        acc4 = acc4 + input21 * hi0 * scale0;
        let byte1 = (word1 >> (bi * 8u)) & 0xFFu;
        let lo1 = f32(i32(byte1 & 0x0Fu) - 8);
        let hi1 = f32(i32((byte1 >> 4u) & 0x0Fu) - 8);
        acc1 = acc1 + input00 * lo1 * scale1;
        acc1 = acc1 + input01 * hi1 * scale1;
        acc3 = acc3 + input10 * lo1 * scale1;
        acc3 = acc3 + input11 * hi1 * scale1;
        acc5 = acc5 + input20 * lo1 * scale1;
        acc5 = acc5 + input21 * hi1 * scale1;
      }
      w = w + WG;
    }

    partial0[lid.x] = acc0;
    partial1[lid.x] = acc1;
    partial2[lid.x] = acc2;
    partial3[lid.x] = acc3;
    partial4[lid.x] = acc4;
    partial5[lid.x] = acc5;
    workgroupBarrier();
    var stride: u32 = WG >> 1u;
    loop {
      if (stride == 0u) {
        break;
      }
      if (lid.x < stride) {
        partial0[lid.x] = partial0[lid.x] + partial0[lid.x + stride];
        partial1[lid.x] = partial1[lid.x] + partial1[lid.x + stride];
        partial2[lid.x] = partial2[lid.x] + partial2[lid.x + stride];
        partial3[lid.x] = partial3[lid.x] + partial3[lid.x + stride];
        partial4[lid.x] = partial4[lid.x] + partial4[lid.x + stride];
        partial5[lid.x] = partial5[lid.x] + partial5[lid.x + stride];
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    if (lid.x == 0u) {
      var out0 = partial0[0];
      var out1 = partial1[0];
      var out2 = partial2[0];
      var out3 = partial3[0];
      var out4 = partial4[0];
      var out5 = partial5[0];
      if (params.has_bias != 0u) {
        let bias0 = t_bias[col0];
        out0 = out0 + bias0;
        out2 = out2 + bias0;
        out4 = out4 + bias0;
        if (has1) {
          let bias1 = t_bias[col1];
          out1 = out1 + bias1;
          out3 = out3 + bias1;
          out5 = out5 + bias1;
        }
      }
      t_out[col0] = out0;
      t_out[params.N + col0] = out2;
      t_out[2u * params.N + col0] = out4;
      if (has1) {
        t_out[col1] = out1;
        t_out[params.N + col1] = out3;
        t_out[2u * params.N + col1] = out5;
      }
    }
    workgroupBarrier();
    pair = pair + ngrp.x;
  }
}
