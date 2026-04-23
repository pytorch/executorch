/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/shims/rand.h>

#include <executorch/backends/aoti/slim/cuda/guard.h>
#include <executorch/backends/aoti/slim/factory/empty.h>
#include <executorch/backends/aoti/slim/util/size_util.h>
#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>
#include <ctime>
#include <vector>

namespace executorch::backends::cuda {

namespace c10 = executorch::backends::aoti::slim::c10;
using c10::Device;
using c10::DeviceIndex;
using c10::DeviceType;
using c10::ScalarType;
using executorch::backends::aoti::slim::empty_strided;
using executorch::backends::aoti::slim::IntArrayRef;
using executorch::backends::aoti::slim::makeArrayRef;

namespace {

// ---- GPU-resident RNG state ----
// Seed and counter live in device memory allocated during the first call
// (warmup phase, before CUDA graph capture). The counter is atomically
// advanced by each kernel invocation on-device, so it automatically
// produces different random sequences on every CUDA graph replay.

struct RngState {
  unsigned long long seed;
  unsigned long long counter;
};

static RngState* d_rng = nullptr;
static bool g_rng_init_done = false;

// Initialize RNG state on the given stream.
// Must be called during warmup (before graph capture).
void ensure_rng_init(cudaStream_t stream) {
  if (!g_rng_init_done) {
    cudaMallocAsync(&d_rng, sizeof(RngState), stream);
    RngState h;
    h.seed = static_cast<unsigned long long>(time(nullptr));
    h.counter = 0;
    cudaMemcpyAsync(
        d_rng, &h, sizeof(RngState), cudaMemcpyHostToDevice, stream);
    // Synchronize to ensure the copy completes before we return
    // (the host-side RngState `h` is on the stack).
    cudaStreamSynchronize(stream);
    g_rng_init_done = true;
  }
}

// Philox-based randint kernel that reads seed from device-resident state
// and atomically advances the counter. The counter pointer survives CUDA
// graph replay, so each replay produces different values.
__global__ void philox_randint_graph_kernel(
    int64_t* __restrict__ out,
    int64_t numel,
    int64_t low,
    int64_t range,
    RngState* __restrict__ rng) {
  // Each thread reads the seed and computes its unique offset.
  // The "base offset" is read from rng->counter. We can't atomicAdd per
  // thread, so we use a two-pass approach: first a single-thread kernel
  // advances the counter, then the main kernel uses the old value.
  // But that requires two kernel launches...
  //
  // Simpler: since numel=1 for randint seed generation, just one thread.
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    // Each invocation atomically grabs `numel` slots from the counter.
    // For numel=1, this is just one atomicAdd.
    unsigned long long my_offset = atomicAdd(&rng->counter, 1ULL);
    curandStatePhilox4_32_10_t state;
    curand_init(rng->seed, idx, my_offset, &state);
    double val = curand_uniform_double(&state);
    int64_t ival = static_cast<int64_t>(val * range);
    out[idx] = low + (ival >= range ? range - 1 : ival);
  }
}

// Philox-based uniform float32 generator (graph-safe version).
__global__ void philox_rand_float_graph_kernel(
    float* __restrict__ out,
    int64_t numel,
    RngState* __restrict__ rng) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    unsigned long long my_offset = atomicAdd(&rng->counter, 1ULL);
    curandStatePhilox4_32_10_t state;
    curand_init(rng->seed, idx, my_offset, &state);
    out[idx] = curand_uniform(&state);
  }
}

// Philox-based uniform bfloat16 generator (graph-safe version).
__global__ void philox_rand_bf16_graph_kernel(
    uint16_t* __restrict__ out,
    int64_t numel,
    RngState* __restrict__ rng) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    unsigned long long my_offset = atomicAdd(&rng->counter, 1ULL);
    curandStatePhilox4_32_10_t state;
    curand_init(rng->seed, idx, my_offset, &state);
    float val = curand_uniform(&state);
    uint32_t bits;
    memcpy(&bits, &val, sizeof(uint32_t));
    uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7FFFu + lsb;
    out[idx] = static_cast<uint16_t>(bits >> 16);
  }
}

} // anonymous namespace

extern "C" {

AOTITorchError aoti_torch_cuda_rand(
    const int64_t* size,
    int64_t size_len_,
    int32_t* dtype,
    int32_t* layout,
    int32_t* device,
    int32_t device_index_,
    int32_t* pin_memory,
    SlimTensor** ret0) {
  (void)layout;
  (void)device;
  (void)pin_memory;

  ET_CHECK_OR_RETURN_ERROR(
      ret0 != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_rand: ret0 is null");

  // Default to float32 if dtype not specified.
  ScalarType scalar_type = ScalarType::Float;
  if (dtype != nullptr) {
    scalar_type = static_cast<ScalarType>(*dtype);
  }

  // Compute contiguous strides and total elements.
  std::vector<int64_t> strides(size_len_);
  int64_t numel = 1;
  for (int64_t i = size_len_ - 1; i >= 0; i--) {
    strides[i] = numel;
    numel *= size[i];
  }

  // Allocate output tensor.
  IntArrayRef sizes_ref(size, static_cast<size_t>(size_len_));
  *ret0 = new SlimTensor(empty_strided(
      sizes_ref,
      makeArrayRef(strides),
      scalar_type,
      Device(DeviceType::CUDA, static_cast<DeviceIndex>(device_index_))));

  if (numel == 0) {
    return Error::Ok;
  }

  // Get the current CUDA stream.
  auto stream_result = getCurrentCUDAStream(0);
  ET_CHECK_OR_RETURN_ERROR(
      stream_result.ok(),
      Internal,
      "aoti_torch_cuda_rand: failed to get CUDA stream");
  cudaStream_t stream = stream_result.get();

  ensure_rng_init(stream);

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);

  if (scalar_type == ScalarType::Float) {
    philox_rand_float_graph_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<float*>((*ret0)->data_ptr()), numel, d_rng);
  } else if (scalar_type == ScalarType::BFloat16) {
    philox_rand_bf16_graph_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<uint16_t*>((*ret0)->data_ptr()), numel, d_rng);
  } else {
    ET_LOG(
        Error,
        "aoti_torch_cuda_rand: unsupported dtype %d",
        static_cast<int>(scalar_type));
    return Error::NotSupported;
  }

  return Error::Ok;
}

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

  // Get the current CUDA stream.
  auto stream_result = getCurrentCUDAStream(0);
  ET_CHECK_OR_RETURN_ERROR(
      stream_result.ok(),
      Internal,
      "aoti_torch_cuda_randint_low_out: failed to get CUDA stream");
  cudaStream_t stream = stream_result.get();

  ensure_rng_init(stream);

  int64_t range = high - low;
  int64_t* out_data = static_cast<int64_t*>(out->data_ptr());

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
  philox_randint_graph_kernel<<<blocks, kThreads, 0, stream>>>(
      out_data, numel, low, range, d_rng);

  return Error::Ok;
}

} // extern "C"

} // namespace executorch::backends::cuda
