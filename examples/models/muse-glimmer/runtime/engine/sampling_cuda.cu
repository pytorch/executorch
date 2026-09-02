/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/muse-glimmer/runtime/engine/sampling_cuda.h>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace muse_glimmer::cuda {
namespace {

constexpr int kArgmaxThreads = 256;

struct ArgmaxCandidate {
  float value;
  uint64_t index;
};

__device__ ArgmaxCandidate better_candidate(
    ArgmaxCandidate lhs,
    ArgmaxCandidate rhs) {
  if (rhs.value > lhs.value ||
      (rhs.value == lhs.value && rhs.index < lhs.index)) {
    return rhs;
  }
  return lhs;
}

__global__ void argmax_index_kernel(
    const float* __restrict__ values,
    int64_t row_size,
    uint64_t* __restrict__ indices) {
  const int64_t row = blockIdx.x;
  const float* row_values = values + row * row_size;

  ArgmaxCandidate candidate{-CUDART_INF_F, uint64_t{0}};
  for (int64_t token = threadIdx.x; token < row_size;
       token += blockDim.x) {
    candidate = better_candidate(
        candidate,
        ArgmaxCandidate{row_values[token], static_cast<uint64_t>(token)});
  }

  __shared__ float shared_values[kArgmaxThreads];
  __shared__ uint64_t shared_indices[kArgmaxThreads];
  shared_values[threadIdx.x] = candidate.value;
  shared_indices[threadIdx.x] = candidate.index;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const ArgmaxCandidate reduced = better_candidate(
          ArgmaxCandidate{
              shared_values[threadIdx.x], shared_indices[threadIdx.x]},
          ArgmaxCandidate{
              shared_values[threadIdx.x + stride],
              shared_indices[threadIdx.x + stride]});
      shared_values[threadIdx.x] = reduced.value;
      shared_indices[threadIdx.x] = reduced.index;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    indices[row] = shared_indices[0];
  }
}

} // namespace

cudaError_t argmax_index(
    const float* values,
    int64_t row_count,
    int64_t row_size,
    uint64_t* indices,
    cudaStream_t stream) {
  if (values == nullptr || indices == nullptr || row_count <= 0 ||
      row_size <= 0) {
    return cudaErrorInvalidValue;
  }
  argmax_index_kernel<<<
      static_cast<unsigned int>(row_count), kArgmaxThreads, 0, stream>>>(
      values, row_size, indices);
  return cudaGetLastError();
}

} // namespace muse_glimmer::cuda
