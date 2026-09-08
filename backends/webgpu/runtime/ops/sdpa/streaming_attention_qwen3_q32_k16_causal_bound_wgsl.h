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

// @generated from streaming_attention_qwen3_q32_k16_causal_bound.wgsl
// DO NOT EDIT.
// wgsl-sha256: 6d4396945b86cc3445698b1182fc4535ca786ad509ba128a6b6b08977e6a55e0
inline constexpr const char* kStreamingAttentionQwen3Q32K16CausalBoundWGSL = R"(
enable f16;

@group(0) @binding(0) var<storage, read_write> t_out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> t_q: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> t_k_cache: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> t_v_cache: array<vec4<f16>>;

struct Params {
  S: u32,
  context_len: u32,
  input_pos: u32,
  q_token_stride4: u32,
  q_head_stride4: u32,
  kv_token_stride4: u32,
  kv_head_stride4: u32,
  o_token_stride4: u32,
  o_head_stride4: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

const HQ: u32 = 16u;
const HKV: u32 = 8u;
const G: u32 = 2u;
const D: u32 = 128u;
const D4: u32 = 32u;
const Q_TILE: u32 = 32u;
const K_TILE: u32 = 16u;
const SCALE: f32 = 0.08838834764831845;
const NEG_INF: f32 = -1.0e30;

var<workgroup> t_q_tile: array<vec4<f32>, 1024>;
var<workgroup> t_kv_tile: array<vec4<f16>, 512>;
var<workgroup> t_scores: array<vec2<f32>, 256>;
var<workgroup> t_m: array<f32, 32>;
var<workgroup> t_d: array<f32, 32>;
var<workgroup> t_alpha: array<f32, 32>;

fn dot_qk(row: u32, key: u32) -> f32 {
  let q_base = row * D4;
  let k_base = key * D4;
  var sum = 0.0;
  var d4 = 0u;
  loop {
    if (d4 >= D4) {
      break;
    }
    sum += dot(t_q_tile[q_base + d4], vec4<f32>(t_kv_tile[k_base + d4]));
    d4 += 1u;
  }
  return sum * SCALE;
}

fn score_for(
  row: u32,
  key_in_tile: u32,
  key: u32,
  row_valid: bool,
  key_valid: bool,
  token: u32,
) -> f32 {
  if (row_valid && key_valid && key <= params.input_pos + token) {
    return dot_qk(row, key_in_tile);
  }
  return NEG_INF;
}

fn max2(v: vec2<f32>) -> f32 {
  return max(v.x, v.y);
}

fn exp_sum(v: vec2<f32>, maximum: f32) -> f32 {
  let p = exp(v - vec2<f32>(maximum));
  return p.x + p.y;
}

@compute @workgroup_size(32, 8, 1)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let groups_per_kv: u32 = (params.S * G + 31u) / 32u;
  let kv_head = wid.x / groups_per_kv;
  let row_group = wid.x % groups_per_kv;
  if (kv_head >= HKV) {
    return;
  }

  let row: u32 = lid.x;
  let dim_vec4_base: u32 = lid.y * 4u;
  let logical_row: u32 = row_group * Q_TILE + row;
  let row_valid = logical_row < params.S * G;
  let token: u32 = logical_row / G;
  let q_head: u32 = kv_head * G + logical_row % G;
  let local_linear = lid.y * 32u + lid.x;
  let group_max_logical_row =
    min(params.S * G - 1u, row_group * Q_TILE + Q_TILE - 1u);
  let group_max_token = group_max_logical_row / G;
  let group_context_len =
    min(params.context_len, params.input_pos + group_max_token + 1u);

  var load_slot = 0u;
  loop {
    if (load_slot >= 4u) {
      break;
    }
    let tile_index = local_linear + load_slot * 256u;
    let load_row = tile_index / D4;
    let load_dim4 = tile_index % D4;
    let load_logical_row = row_group * Q_TILE + load_row;
    if (load_logical_row < params.S * G) {
      let load_token = load_logical_row / G;
      let load_q_head = kv_head * G + load_logical_row % G;
      let q_index =
        load_token * params.q_token_stride4 +
        load_q_head * params.q_head_stride4 +
        load_dim4;
      t_q_tile[tile_index] = t_q[q_index];
    } else {
      t_q_tile[tile_index] = vec4<f32>(0.0);
    }
    load_slot += 1u;
  }

  if (lid.y == 0u) {
    t_m[row] = NEG_INF;
    t_d[row] = 0.0;
    t_alpha[row] = 0.0;
  }
  workgroupBarrier();

  var score_acc = vec2<f32>(0.0);
  var output_acc: array<vec4<f32>, 4>;
  output_acc[0] = vec4<f32>(0.0);
  output_acc[1] = vec4<f32>(0.0);
  output_acc[2] = vec4<f32>(0.0);
  output_acc[3] = vec4<f32>(0.0);

  var key_tile_start = 0u;
  loop {
    if (key_tile_start >= group_context_len) {
      break;
    }

    load_slot = 0u;
    loop {
      if (load_slot >= 2u) {
        break;
      }
      let tile_index = local_linear + load_slot * 256u;
      let key_in_tile = tile_index / D4;
      let load_dim4 = tile_index % D4;
      let key = key_tile_start + key_in_tile;
      if (key < params.context_len) {
        let cache_index =
          key * params.kv_token_stride4 +
          kv_head * params.kv_head_stride4 +
          load_dim4;
        t_kv_tile[tile_index] = t_k_cache[cache_index];
      } else {
        t_kv_tile[tile_index] = vec4<f16>(0.0h);
      }
      load_slot += 1u;
    }
    workgroupBarrier();

    let score_key_base = lid.y * 2u;
    let key0 = key_tile_start + score_key_base;
    let key1 = key0 + 1u;
    score_acc = vec2<f32>(
      score_for(row, score_key_base, key0, row_valid, key0 < params.context_len, token),
      score_for(row, score_key_base + 1u, key1, row_valid, key1 < params.context_len, token),
    );
    let score_store = row * 8u + lid.y;
    t_scores[score_store] = score_acc;
    workgroupBarrier();

    if (lid.y == 0u) {
      let row_score_base = row * 8u;
      let s0 = t_scores[row_score_base];
      let s1 = t_scores[row_score_base + 1u];
      let s2 = t_scores[row_score_base + 2u];
      let s3 = t_scores[row_score_base + 3u];
      let s4 = t_scores[row_score_base + 4u];
      let s5 = t_scores[row_score_base + 5u];
      let s6 = t_scores[row_score_base + 6u];
      let s7 = t_scores[row_score_base + 7u];
      let tile_max = max(
        max(max(max2(s0), max2(s1)), max(max2(s2), max2(s3))),
        max(max(max2(s4), max2(s5)), max(max2(s6), max2(s7))),
      );
      let old_m = t_m[row];
      let old_d = t_d[row];
      let new_m = max(old_m, tile_max);
      if (row_valid) {
        t_alpha[row] = exp(old_m - new_m);
        let tile_sum =
          exp_sum(s0, new_m) + exp_sum(s1, new_m) +
          exp_sum(s2, new_m) + exp_sum(s3, new_m) +
          exp_sum(s4, new_m) + exp_sum(s5, new_m) +
          exp_sum(s6, new_m) + exp_sum(s7, new_m);
        t_d[row] = old_d * t_alpha[row] + tile_sum;
        t_m[row] = new_m;
      } else {
        t_alpha[row] = 0.0;
        t_d[row] = 1.0;
        t_m[row] = 0.0;
      }
    }

    load_slot = 0u;
    loop {
      if (load_slot >= 2u) {
        break;
      }
      let tile_index = local_linear + load_slot * 256u;
      let key_in_tile = tile_index / D4;
      let load_dim4 = tile_index % D4;
      let key = key_tile_start + key_in_tile;
      if (key < params.context_len) {
        let cache_index =
          key * params.kv_token_stride4 +
          kv_head * params.kv_head_stride4 +
          load_dim4;
        t_kv_tile[tile_index] = t_v_cache[cache_index];
      } else {
        t_kv_tile[tile_index] = vec4<f16>(0.0h);
      }
      load_slot += 1u;
    }
    workgroupBarrier();

    let alpha = t_alpha[row];
    output_acc[0] = output_acc[0] * alpha;
    output_acc[1] = output_acc[1] * alpha;
    output_acc[2] = output_acc[2] * alpha;
    output_acc[3] = output_acc[3] * alpha;
    let new_m = t_m[row];
    let row_score_base = row * 8u;
    var score_block = 0u;
    loop {
      if (score_block >= 8u) {
        break;
      }
      let probabilities =
        exp(t_scores[row_score_base + score_block] - vec2<f32>(new_m));
      let value_key_base = score_block * 2u;
      let value_dim0 = dim_vec4_base;
      let value_dim1 = dim_vec4_base + 1u;
      let value_dim2 = dim_vec4_base + 2u;
      let value_dim3 = dim_vec4_base + 3u;
      output_acc[0] +=
        vec4<f32>(t_kv_tile[(value_key_base + 0u) * D4 + value_dim0]) * probabilities.x +
        vec4<f32>(t_kv_tile[(value_key_base + 1u) * D4 + value_dim0]) * probabilities.y;
      output_acc[1] +=
        vec4<f32>(t_kv_tile[(value_key_base + 0u) * D4 + value_dim1]) * probabilities.x +
        vec4<f32>(t_kv_tile[(value_key_base + 1u) * D4 + value_dim1]) * probabilities.y;
      output_acc[2] +=
        vec4<f32>(t_kv_tile[(value_key_base + 0u) * D4 + value_dim2]) * probabilities.x +
        vec4<f32>(t_kv_tile[(value_key_base + 1u) * D4 + value_dim2]) * probabilities.y;
      output_acc[3] +=
        vec4<f32>(t_kv_tile[(value_key_base + 0u) * D4 + value_dim3]) * probabilities.x +
        vec4<f32>(t_kv_tile[(value_key_base + 1u) * D4 + value_dim3]) * probabilities.y;
      score_block += 1u;
    }
    workgroupBarrier();
    key_tile_start += K_TILE;
  }

  if (row_valid) {
    let denominator = t_d[row];
    let output_base =
      token * params.o_token_stride4 +
      q_head * params.o_head_stride4 +
      dim_vec4_base;
    t_out[output_base] = output_acc[0] / denominator;
    t_out[output_base + 1u] = output_acc[1] / denominator;
    t_out[output_base + 2u] = output_acc[2] / denominator;
    t_out[output_base + 3u] = output_acc[3] / denominator;
  }
}
)";

inline constexpr uint32_t
    kStreamingAttentionQwen3Q32K16CausalBoundWorkgroupSizeX = 32;
inline constexpr uint32_t
    kStreamingAttentionQwen3Q32K16CausalBoundWorkgroupSizeY = 8;
inline constexpr uint32_t
    kStreamingAttentionQwen3Q32K16CausalBoundWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
