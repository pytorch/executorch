// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_part_o: array<f32>;
@group(0) @binding(2) var<storage, read> t_part_ml: array<f32>;

struct Params {
  D: u32,
  num_splits: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

const WG_SIZE: u32 = 64u;
const MAX_SPLITS: u32 = 128u;
const MAX_D_PER_LANE: u32 = 2u;
const NEG_INF: f32 = -1.0e30;

// w_i = exp(m_i - M) per split, computed once and reused for the L-sum and every output dim.
var<workgroup> sh_w: array<f32, MAX_SPLITS>;
var<workgroup> sh_reduce: array<f32, WG_SIZE>;

// FlashDecoding pass 2: online-softmax merge of the per-split partials, then normalize.
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wid.x;
  let t = lid.x;
  let D = params.D;
  let ns = params.num_splits;
  let head_base = h * MAX_SPLITS;

  var local_m: f32 = NEG_INF;
  for (var i: u32 = t; i < ns; i = i + WG_SIZE) {
    local_m = max(local_m, t_part_ml[(head_base + i) * 2u + 0u]);
  }
  sh_reduce[t] = local_m;
  workgroupBarrier();
  for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (t < stride) {
      sh_reduce[t] = max(sh_reduce[t], sh_reduce[t + stride]);
    }
    workgroupBarrier();
  }
  let M = sh_reduce[0];
  // Compute w_i = exp(m_i - M) once per split into shared memory (was recomputed per output dim).
  for (var i: u32 = t; i < ns; i = i + WG_SIZE) {
    sh_w[i] = exp(t_part_ml[(head_base + i) * 2u + 0u] - M);
  }
  workgroupBarrier();

  var local_l: f32 = 0.0;
  for (var i: u32 = t; i < ns; i = i + WG_SIZE) {
    local_l = local_l + sh_w[i] * t_part_ml[(head_base + i) * 2u + 1u];
  }
  sh_reduce[t] = local_l;
  workgroupBarrier();
  for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (t < stride) {
      sh_reduce[t] = sh_reduce[t] + sh_reduce[t + stride];
    }
    workgroupBarrier();
  }
  let L = sh_reduce[0];
  let inv = select(0.0, 1.0 / L, L > 0.0);

  for (var nd: u32 = 0u; nd < MAX_D_PER_LANE; nd = nd + 1u) {
    let d = t + nd * WG_SIZE;
    if (d < D) {
      var acc: f32 = 0.0;
      for (var i: u32 = 0u; i < ns; i = i + 1u) {
        acc = acc + sh_w[i] * t_part_o[(head_base + i) * D + d];
      }
      t_out[h * D + d] = acc * inv;
    }
  }
}
