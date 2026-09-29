/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>

#include <cstdint>

namespace executorch::backends::webgpu {

// FlashDecoding's lane-owns-D layout covers head dims up to WG_SIZE(64) *
// MAX_D_PER_LANE(2). Decode shapes above this fall through to the materialized
// SDPA path (the FD selection predicate in Sdpa.cpp checks this).
constexpr int64_t kSdpaFdMaxHeadDim = 128;

struct SdpaFdDecodeState {
  uint32_t Hq;
  uint32_t Hkv;
  uint32_t D;
  uint32_t context_len;
  uint32_t g;
  uint32_t num_splits;
  uint32_t split_len;
  float scale;
  utils::WgCount split_grid;
  utils::WgCount reduce_grid;
};

struct SdpaFdDecodeResources {
  WGPUBuffer split_uniform;
  WGPUBuffer reduce_uniform;
  utils::DispatchRange dispatch_range;
};

SdpaFdDecodeState make_sdpa_fd_decode_state(
    WGPUDevice device,
    int64_t Hq,
    int64_t Hkv,
    int64_t D,
    int64_t context_len,
    int64_t g,
    float scale);

// Records split + reduce with retained UBOs. Route selection is owned by the
// caller so this helper never mutates recorded dispatch counts.
SdpaFdDecodeResources record_sdpa_fd_decode_dispatches(
    WebGPUGraph& graph,
    const WebGPUTensor& q,
    const WebGPUTensor& k_cache,
    const WebGPUTensor& v_cache,
    const WebGPUTensor& out,
    const SdpaFdDecodeState& state);

void write_sdpa_fd_decode_uniforms(
    WGPUQueue queue,
    const SdpaFdDecodeResources& resources,
    const SdpaFdDecodeState& state);

} // namespace executorch::backends::webgpu
