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

// @generated from rotary_embedding_hf.wgsl - DO NOT EDIT.
// wgsl-sha256: 4f081ed4c8165f021cbb722d379e437f30b8dfb08bf03bfcbaa406ed7799c7b6
inline constexpr const char* kRotaryEmbeddingHfWGSL = R"(
@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_in: array<f32>;
@group(0) @binding(2) var<storage, read> t_freqs_cos: array<f32>;
@group(0) @binding(3) var<storage, read> t_freqs_sin: array<f32>;

struct Params {
  n_heads: u32,
  seq: u32,
  head_dim: u32,
  half_dim: u32,
  num_pairs: u32,
  rotary_dim: u32,
  start_pos: u32,
  _pad0: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

// One thread per (i, i+half_dim) pair; HuggingFace rotate-half RoPE, shared
// xq/xk shader. freqs is the FULL [max_seq, rotary_dim] table indexed at row
// (start_pos + s); each output half uses its corresponding frequency column.
@compute @workgroup_size(wg_size, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let pair = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (pair >= params.num_pairs) {
    return;
  }
  let half_dim = params.half_dim;
  let pair_i = pair % half_dim;
  let t1 = pair / half_dim;
  let head = t1 % params.n_heads;
  let t2 = t1 / params.n_heads;
  let s = t2 % params.seq;
  let b = t2 / params.seq;

  let head_base =
      ((b * params.seq + s) * params.n_heads + head) * params.head_dim;
  let a_idx = head_base + pair_i;
  let b_idx = head_base + pair_i + half_dim;
  let freqs_base = (s + params.start_pos) * params.rotary_dim;
  let freqs_a_idx = freqs_base + pair_i;
  let freqs_b_idx = freqs_a_idx + half_dim;

  let c_a = t_freqs_cos[freqs_a_idx];
  let si_a = t_freqs_sin[freqs_a_idx];
  let c_b = t_freqs_cos[freqs_b_idx];
  let si_b = t_freqs_sin[freqs_b_idx];
  let x_a = t_in[a_idx];
  let x_b = t_in[b_idx];
  t_out[a_idx] = x_a * c_a - x_b * si_a;
  t_out[b_idx] = x_b * c_b + x_a * si_b;
}
)";

inline constexpr uint32_t kRotaryEmbeddingHfWorkgroupSizeX = 64;
inline constexpr uint32_t kRotaryEmbeddingHfWorkgroupSizeY = 1;
inline constexpr uint32_t kRotaryEmbeddingHfWorkgroupSizeZ = 1;

} // namespace executorch::backends::webgpu
