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

class SamplingWorkspace {
 public:
  SamplingWorkspace();
  ~SamplingWorkspace();

  SamplingWorkspace(const SamplingWorkspace&) = delete;
  SamplingWorkspace& operator=(const SamplingWorkspace&) = delete;

  // Allocates all scratch buffers needed for this fixed batch shape. Call
  // before CUDA graph capture; repeated calls for the same shape are no-ops.
  cudaError_t reserve(
      int64_t row_count,
      int64_t row_size,
      cudaStream_t stream);

  // Resets the graph-safe Philox state used by stochastic primitives.
  cudaError_t set_seed(uint64_t seed, cudaStream_t stream);

 private:
  struct Impl;
  Impl* impl_;

  friend cudaError_t fill_sampling_probabilities(
      const float*,
      int64_t,
      int64_t,
      double,
      int32_t,
      double,
      float*,
      SamplingWorkspace&,
      cudaStream_t);
  friend cudaError_t categorical_sample(
      const float*,
      int64_t,
      int64_t,
      uint64_t*,
      SamplingWorkspace&,
      cudaStream_t);
};

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

// CUDA counterpart of muse_glimmer::fill_sampling_probabilities for contiguous rows.
// Applies temperature, then top-k, then top-p, and writes normalized dense
// probabilities in original token order. All tensor pointers are device
// pointers and the launch sequence is asynchronous with respect to `stream`.
cudaError_t fill_sampling_probabilities(
    const float* logits,
    int64_t row_count,
    int64_t row_size,
    double temperature,
    int32_t top_k,
    double top_p,
    float* probabilities,
    SamplingWorkspace& workspace,
    cudaStream_t stream);

// CUDA counterpart of muse_glimmer::categorical_sample for normalized contiguous
// probability rows. Produces one device-resident token id per row.
cudaError_t categorical_sample(
    const float* probabilities,
    int64_t row_count,
    int64_t row_size,
    uint64_t* tokens,
    SamplingWorkspace& workspace,
    cudaStream_t stream);

} // namespace muse_glimmer::cuda
