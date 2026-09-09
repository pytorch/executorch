/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CUDA counterparts of the host sampling primitives in sampling.h.

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace muse_glimmer::cuda {

// Computes one argmax per contiguous row of `values`.
//
// `values` and `indices` must point to CUDA memory. Equal maxima select the
// lowest token index, matching muse_glimmer::argmax_index. The launch is asynchronous
// with respect to `stream`.
cudaError_t argmax_index(
    const float* values,
    int64_t row_count,
    int64_t row_size,
    uint64_t* indices,
    cudaStream_t stream);

} // namespace muse_glimmer::cuda
