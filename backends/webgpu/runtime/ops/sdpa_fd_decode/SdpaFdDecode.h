/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>

#include <cstdint>

namespace executorch::backends::webgpu {

// FlashDecoding's lane-owns-D layout covers head dims up to WG_SIZE(64) *
// MAX_D_PER_LANE(2). Decode shapes above this fall through to the materialized
// SDPA path (the FD selection predicate in Sdpa.cpp checks this).
constexpr int64_t kSdpaFdMaxHeadDim = 128;

// Split-KV FlashDecoding decode dispatch (S==1): a split pass over
// Hq*num_splits workgroups + a reduce pass over Hq workgroups. Called from the
// Sdpa.cpp WEBGPU_SDPA_FD branch.
void sdpa_fd_decode_dispatch(
    WebGPUGraph& graph,
    const WebGPUTensor& q,
    const WebGPUTensor& k_cache,
    const WebGPUTensor& v_cache,
    const WebGPUTensor& out,
    int64_t Hq,
    int64_t Hkv,
    int64_t D,
    int64_t context_len,
    int64_t g,
    float scale);

} // namespace executorch::backends::webgpu
