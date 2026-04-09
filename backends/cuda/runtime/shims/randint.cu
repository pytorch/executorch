/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <curand.h>

#include <executorch/backends/cuda/runtime/shims/randint.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

#include <cstdint>
#include <ctime>

namespace executorch::backends::cuda {

using executorch::runtime::Error;

namespace {

// Transform cuRAND uniform doubles (0, 1] to int64 values in [low, high).
__global__ void uniform_to_randint_kernel(
    int64_t* out,
    const double* uniform,
    int64_t numel,
    int64_t low,
    int64_t range) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    // uniform is in (0, 1], so (uniform * range) is in (0, range].
    // Subtract 1 and clamp to get [0, range-1], then add low for [low, high-1].
    int64_t val = static_cast<int64_t>(uniform[idx] * range);
    out[idx] = low + (val >= range ? range - 1 : val);
  }
}

curandGenerator_t get_or_create_generator() {
  static curandGenerator_t gen = nullptr;
  if (gen == nullptr) {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(
        gen, static_cast<unsigned long long>(time(nullptr)));
  }
  return gen;
}

} // anonymous namespace

extern "C" {

AOTITorchError aoti_torch_cuda_randint_low_out(
    SlimTensor* out,
    int64_t low,
    int64_t high,
    const int64_t* size,
    int64_t size_len_) {
  ET_CHECK_OR_RETURN_ERROR(
      out != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_randint_low_out: out tensor is null");

  ET_CHECK_OR_RETURN_ERROR(
      high > low,
      InvalidArgument,
      "aoti_torch_cuda_randint_low_out: requires high > low");

  int64_t numel = 1;
  for (int64_t i = 0; i < size_len_; i++) {
    numel *= size[i];
  }
  if (numel == 0) {
    return Error::Ok;
  }

  int64_t range = high - low;
  int64_t* out_data = static_cast<int64_t*>(out->data_ptr());

  // Allocate temporary buffer for uniform doubles on device.
  double* d_uniform = nullptr;
  auto alloc_err = cudaMalloc(&d_uniform, numel * sizeof(double));
  ET_CHECK_OR_RETURN_ERROR(
      alloc_err == cudaSuccess,
      Internal,
      "aoti_torch_cuda_randint_low_out: cudaMalloc failed (%d)",
      static_cast<int>(alloc_err));

  // Generate uniform doubles in (0, 1].
  auto gen = get_or_create_generator();
  curandGenerateUniformDouble(gen, d_uniform, numel);

  // Transform to integers in [low, high).
  constexpr int kThreads = 256;
  int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
  uniform_to_randint_kernel<<<blocks, kThreads>>>(
      out_data, d_uniform, numel, low, range);

  cudaFree(d_uniform);

  return Error::Ok;
}

} // extern "C"

} // namespace executorch::backends::cuda
