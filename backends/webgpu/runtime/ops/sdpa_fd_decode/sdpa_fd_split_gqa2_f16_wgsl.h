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

// @generated from sdpa_fd_split_gqa2_f16.wgsl - DO NOT EDIT.
// wgsl-sha256: 9571dc5a5b05c4f9d1f39c37e18100c3de3fe65573ae501f9b3442c8c21d9632
inline constexpr const char* kSdpaFdSplitGqa2F16WGSL = R"(
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

enable f16;

@group(0) @binding(0) var<storage, read_write> t_part_o: array<f32>;
@group(0) @binding(1) var<storage, read_write> t_part_ml: array<f32>;
@group(0) @binding(2) var<storage, read> t_q: array<f32>;
@group(0) @binding(3) var<storage, read> t_k_cache: array<f16>;
@group(0) @binding(4) var<storage, read> t_v_cache: array<f16>;

struct Params {
  _pad0: u32,
  Hkv: u32,
  D: u32,
  context_len: u32,
  g: u32,
  num_splits: u32,
  split_len: u32,
  scale: f32,
}
@group(0) @binding(5) var<uniform> params: Params;

const WG_SIZE: u32 = 64u;
const MAX_SPLITS: u32 = 128u;
const MAX_D_PER_LANE: u32 = 2u;
const G: u32 = 2u;
const NEG_INF: f32 = -1.0e30;

var<workgroup> sh_p: array<f32, G * WG_SIZE>;
var<workgroup> sh_red: array<f32, WG_SIZE>;

// Qwen3 GQA=2 f16 FlashDecoding split. A workgroup covers both query heads
// associated with one KV head. QK remains independent for each query head;
// the V row is loaded once and applied to both heads' softmax weights.
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>) {
  let h_kv = wid.x / params.num_splits;
  let split_i = wid.x % params.num_splits;
  let t = lid.x;
  let D = params.D;
  let D4 = D / 4u;
  let ctx = params.context_len;
  let kv_row_stride = params.Hkv * D;

  let c0 = split_i * params.split_len;
  var c1 = c0 + params.split_len;
  if (c1 > ctx) { c1 = ctx; }

  var m: array<f32, G>;
  var l: array<f32, G>;
  var o_acc: array<array<f32, MAX_D_PER_LANE>, G>;
  for (var group_head: u32 = 0u; group_head < G; group_head = group_head + 1u) {
    m[group_head] = NEG_INF;
    l[group_head] = 0.0;
    for (var nd: u32 = 0u; nd < MAX_D_PER_LANE; nd = nd + 1u) {
      o_acc[group_head][nd] = 0.0;
    }
  }

  var block: u32 = c0;
  loop {
    if (block >= c1) { break; }
    var n: u32 = c1 - block;
    if (n > WG_SIZE) { n = WG_SIZE; }

    if (t < n) {
      let kv_base = (block + t) * kv_row_stride + h_kv * D;
      for (var group_head: u32 = 0u; group_head < G; group_head = group_head + 1u) {
        let q_base = (h_kv * G + group_head) * D;
        var acc4 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        for (var i4: u32 = 0u; i4 < D4; i4 = i4 + 1u) {
          let qi = q_base + i4 * 4u;
          let ki = kv_base + i4 * 4u;
          let qv = vec4<f32>(
              t_q[qi], t_q[qi + 1u], t_q[qi + 2u], t_q[qi + 3u]);
          let kv = vec4<f32>(
              f32(t_k_cache[ki]), f32(t_k_cache[ki + 1u]),
              f32(t_k_cache[ki + 2u]), f32(t_k_cache[ki + 3u]));
          acc4 = acc4 + qv * kv;
        }
        sh_p[group_head * WG_SIZE + t] =
            (acc4.x + acc4.y + acc4.z + acc4.w) * params.scale;
      }
    } else {
      for (var group_head: u32 = 0u; group_head < G; group_head = group_head + 1u) {
        sh_p[group_head * WG_SIZE + t] = NEG_INF;
      }
    }
    workgroupBarrier();

    var rescale: array<f32, G>;
    for (var group_head: u32 = 0u; group_head < G; group_head = group_head + 1u) {
      sh_red[t] = sh_p[group_head * WG_SIZE + t];
      workgroupBarrier();
      for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (t < stride) {
          sh_red[t] = max(sh_red[t], sh_red[t + stride]);
        }
        workgroupBarrier();
      }
      let m_new = max(m[group_head], sh_red[0]);
      rescale[group_head] = exp(m[group_head] - m_new);

      var p_t: f32 = 0.0;
      if (t < n) {
        p_t = exp(sh_p[group_head * WG_SIZE + t] - m_new);
      }
      workgroupBarrier();
      sh_p[group_head * WG_SIZE + t] = p_t;
      sh_red[t] = p_t;
      workgroupBarrier();
      for (var stride: u32 = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (t < stride) {
          sh_red[t] = sh_red[t] + sh_red[t + stride];
        }
        workgroupBarrier();
      }
      l[group_head] = rescale[group_head] * l[group_head] + sh_red[0];
      m[group_head] = m_new;
      workgroupBarrier();
    }

    for (var nd: u32 = 0u; nd < MAX_D_PER_LANE; nd = nd + 1u) {
      let d = t + nd * WG_SIZE;
      if (d < D) {
        for (var group_head: u32 = 0u; group_head < G; group_head = group_head + 1u) {
          o_acc[group_head][nd] =
              rescale[group_head] * o_acc[group_head][nd];
        }
        for (var j: u32 = 0u; j < n; j = j + 1u) {
          let v_base = (block + j) * kv_row_stride + h_kv * D;
          let v_value = f32(t_v_cache[v_base + d]);
          for (var group_head: u32 = 0u; group_head < G; group_head = group_head + 1u) {
            o_acc[group_head][nd] = o_acc[group_head][nd] +
                sh_p[group_head * WG_SIZE + j] * v_value;
          }
        }
      }
    }
    workgroupBarrier();
    block = block + WG_SIZE;
  }

  for (var group_head: u32 = 0u; group_head < G; group_head = group_head + 1u) {
    let h = h_kv * G + group_head;
    let part = h * MAX_SPLITS + split_i;
    for (var nd: u32 = 0u; nd < MAX_D_PER_LANE; nd = nd + 1u) {
      let d = t + nd * WG_SIZE;
      if (d < D) {
        t_part_o[part * D + d] = o_acc[group_head][nd];
      }
    }
    if (t == 0u) {
      t_part_ml[part * 2u] = m[group_head];
      t_part_ml[part * 2u + 1u] = l[group_head];
    }
  }
}
)";

inline constexpr uint32_t kSdpaFdSplitGqa2F16WorkgroupSizeX = 64;
inline constexpr uint32_t kSdpaFdSplitGqa2F16WorkgroupSizeY = 1;
inline constexpr uint32_t kSdpaFdSplitGqa2F16WorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
