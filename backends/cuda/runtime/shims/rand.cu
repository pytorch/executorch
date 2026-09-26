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
#include <mutex>
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
  // Per-launch scratch — written by advance_counter_kernel and read by
  // the main RNG kernels. Single-threaded host driver is assumed
  // (typical inference / CUDA-graph replay use case).
  unsigned long long base_scratch;
};

static RngState* d_rng = nullptr;
// std::call_once guarantees one-shot initialization even when shims are
// invoked from multiple host threads (e.g. concurrent models / streams).
static std::once_flag g_rng_init_flag;

// Initialize RNG state on the given stream.
// Must be called during warmup (before graph capture). Subsequent calls
// from any thread are no-ops thanks to std::call_once.
void ensure_rng_init(cudaStream_t stream) {
  std::call_once(g_rng_init_flag, [&]() {
    cudaMallocAsync(&d_rng, sizeof(RngState), stream);
    RngState h;
    h.seed = static_cast<unsigned long long>(time(nullptr));
    h.counter = 0;
    h.base_scratch = 0;
    cudaMemcpyAsync(
        d_rng, &h, sizeof(RngState), cudaMemcpyHostToDevice, stream);
    // Synchronize to ensure the copy completes before we return
    // (the host-side RngState `h` is on the stack).
    cudaStreamSynchronize(stream);
  });
}

// Philox-based randint kernel. Reads its base offset from `rng->base_scratch`
// (populated by `advance_counter_kernel` immediately before this launch).
// This replaces the previous per-element atomicAdd contention with a single
// atomic per kernel launch.
//
// Matches PyTorch's `transformation::uniform_int_from_to` semantics: builds
// a 64-bit random value from two 32-bit curand draws, then takes
// `val % range + low` so the output lies in [low, high).
__global__ void philox_randint_graph_kernel(
    int64_t* __restrict__ out,
    int64_t numel,
    int64_t low,
    int64_t range,
    RngState* __restrict__ rng) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    curandStatePhilox4_32_10_t state;
    curand_init(rng->seed, idx, rng->base_scratch, &state);
    uint32_t hi = curand(&state);
    uint32_t lo = curand(&state);
    uint64_t rval = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    uint64_t urange = static_cast<uint64_t>(range);
    out[idx] = low + static_cast<int64_t>(rval % urange);
  }
}

// Maps a uniformly distributed uint32 to a float32 in [0, 1) following the
// pattern used by PyTorch's `transformation::uniform_real` in
// aten/src/ATen/native/cuda/DistributionTemplates.h: keep the low 24 mantissa
// bits and divide by 2^24.
__device__ inline float uniform_real_from_uint32(uint32_t val) {
  // std::numeric_limits<float>::digits == 24
  constexpr uint32_t kMantissaMask = (1u << 24) - 1;
  constexpr float kDivisor = 1.0f / static_cast<float>(1u << 24);
  return static_cast<float>(val & kMantissaMask) * kDivisor;
}

// Philox-based uniform float32 generator (graph-safe version). Produces
// values in [0, 1) to match torch.rand semantics.
__global__ void philox_rand_float_graph_kernel(
    float* __restrict__ out,
    int64_t numel,
    RngState* __restrict__ rng) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    curandStatePhilox4_32_10_t state;
    curand_init(rng->seed, idx, rng->base_scratch, &state);
    out[idx] = uniform_real_from_uint32(curand(&state));
  }
}

// Philox-based uniform bfloat16 generator (graph-safe version). Produces a
// float in [0, 1) and rounds to bfloat16 with round-to-nearest-even.
__global__ void philox_rand_bf16_graph_kernel(
    uint16_t* __restrict__ out,
    int64_t numel,
    RngState* __restrict__ rng) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numel) {
    curandStatePhilox4_32_10_t state;
    curand_init(rng->seed, idx, rng->base_scratch, &state);
    float val = uniform_real_from_uint32(curand(&state));
    uint32_t bits;
    memcpy(&bits, &val, sizeof(uint32_t));
    uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7FFFu + lsb;
    out[idx] = static_cast<uint16_t>(bits >> 16);
  }
}

// Single-thread helper that grabs a contiguous range of `numel` offsets
// from the on-device counter and writes the base into `rng->base_scratch`.
// Replaces `numel` per-element atomics with a single atomic per launch
// while staying graph-capturable.
__global__ void advance_counter_kernel(
    RngState* __restrict__ rng,
    unsigned long long numel) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    rng->base_scratch = atomicAdd(&rng->counter, numel);
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

  // Single atomicAdd per launch — grabs `numel` consecutive counter slots
  // for the kernel below, eliminating per-element contention on the GPU
  // counter.
  advance_counter_kernel<<<1, 1, 0, stream>>>(
      d_rng, static_cast<unsigned long long>(numel));

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
  // One atomicAdd per launch; subsequent kernel reads `rng->base_scratch`.
  advance_counter_kernel<<<1, 1, 0, stream>>>(
      d_rng, static_cast<unsigned long long>(numel));
  philox_randint_graph_kernel<<<blocks, kThreads, 0, stream>>>(
      out_data, numel, low, range, d_rng);

  return Error::Ok;
}

} // extern "C"

} // namespace executorch::backends::cuda
