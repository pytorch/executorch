// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

@group(0) @binding(0) var<storage, read_write> attn: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;

struct Params {
  B: u32,
  Hq: u32,
  Hkv: u32,
  S_q: u32,
  S_kv: u32,
  D: u32,
  g: u32,
  has_mask: u32,
  mask_mode: u32,
  tensor_layout: u32,
  _pad0: u32,
  scale: f32,
  bh_lo: u32,
  bh_count: u32,
  elide_masked_qk: u32,
  _pad2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64;

fn q_row4(b: u32, h: u32, s: u32, d4_count: u32) -> u32 {
  if (params.tensor_layout == 0u) {
    return ((b * params.Hq + h) * params.S_q + s) * d4_count;
  }
  return ((b * params.S_q + s) * params.Hq + h) * d4_count;
}

fn k_row4(b: u32, h: u32, s: u32, d4_count: u32) -> u32 {
  if (params.tensor_layout == 0u) {
    return ((b * params.Hkv + h) * params.S_kv + s) * d4_count;
  }
  return ((b * params.S_kv + s) * params.Hkv + h) * d4_count;
}

fn all_finite(value: vec4<f32>) -> bool {
  let exponent = bitcast<vec4<u32>>(value) & vec4<u32>(0x7f800000u);
  return all(exponent != vec4<u32>(0x7f800000u));
}

@compute @workgroup_size(wg_size)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>) {
  let chunk_numel = params.bh_count * params.S_q * params.S_kv;
  let tid = gid.x + gid.y * (nwg.x * wg_size);
  if (tid >= chunk_numel) {
    return;
  }
  let idx = params.bh_lo * params.S_q * params.S_kv + tid;
  let c = idx % params.S_kv;
  let row = idx / params.S_kv;
  let s = row % params.S_q;
  let h = (row / params.S_q) % params.Hq;
  let b = row / (params.S_q * params.Hq);
  let kv_h = h / params.g;
  let d4_count = params.D / 4u;
  let qbase4 = q_row4(b, h, s, d4_count);
  let kbase4 = k_row4(b, kv_h, c, d4_count);

  var mask_value: f32 = 0.0;
  if (params.has_mask != 0u) {
    if (params.mask_mode == 1u) {
      mask_value = mask[params.S_kv * s + c];
    } else {
      mask_value = mask[idx];
    }
  }
  if (params.elide_masked_qk != 0u && params.mask_mode == 1u &&
      bitcast<u32>(mask_value) == 0xff800000u) {
    var finite_inputs = true;
    for (var d4: u32 = 0u; d4 < d4_count; d4 = d4 + 1u) {
      finite_inputs = finite_inputs && all_finite(q[qbase4 + d4]) &&
          all_finite(k[kbase4 + d4]);
    }
    if (finite_inputs) {
      attn[tid] = mask_value;
      return;
    }
  }

  var acc: f32 = 0.0;
  for (var d4: u32 = 0u; d4 < d4_count; d4 = d4 + 1u) {
    acc = acc + dot(q[qbase4 + d4], k[kbase4 + d4]);
  }
  acc = acc * params.scale;
  if (params.has_mask != 0u) {
    acc = acc + mask_value;
  }
  attn[tid] = acc;
}
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
