/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/muse-glimmer/runtime/engine/sampling_cuda.h>

#include <cuda_runtime.h>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <curand_kernel.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace muse_glimmer::cuda {
namespace {

constexpr int kArgmaxThreads = 256;
constexpr int kSamplingThreads = 256;

struct ArgmaxCandidate {
  float value;
  uint64_t index;
};

struct DeviceRngState {
  uint64_t seed;
  unsigned long long counter;
  unsigned long long base;
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

__global__ void initialize_offsets_kernel(
    int64_t* offsets,
    int64_t row_count,
    int64_t row_size) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (row <= row_count) {
    offsets[row] = row * row_size;
  }
}

__global__ void initialize_token_indices_kernel(
    int32_t* indices,
    int64_t total_size,
    int64_t row_size) {
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (offset < total_size) {
    indices[offset] = static_cast<int32_t>(offset % row_size);
  }
}

__global__ void compute_sorted_weights_kernel(
    const float* sorted_logits,
    int64_t total_size,
    int64_t row_size,
    int32_t retained_count,
    float inverse_temperature,
    double* weights) {
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (offset >= total_size) {
    return;
  }
  const int64_t rank = offset % row_size;
  if (rank >= retained_count) {
    weights[offset] = 0.0;
    return;
  }
  const int64_t row_start = offset - rank;
  const float weight = expf(
      (sorted_logits[offset] - sorted_logits[row_start]) * inverse_temperature);
  weights[offset] = static_cast<double>(weight);
}

__global__ void find_nucleus_kernel(
    const double* cumulative,
    int64_t row_count,
    int64_t row_size,
    int32_t retained_count,
    double top_p,
    int32_t* cutoffs,
    float* denominators) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (row >= row_count) {
    return;
  }
  const int64_t row_start = row * row_size;
  const int32_t last_retained = retained_count - 1;
  const double total = cumulative[row_start + last_retained];
  int32_t cutoff = last_retained;
  if (top_p > 0.0 && top_p < 1.0) {
    const double target = top_p * total;
    int32_t low = 0;
    int32_t high = last_retained;
    while (low < high) {
      const int32_t middle = low + (high - low) / 2;
      if (cumulative[row_start + middle] >= target) {
        high = middle;
      } else {
        low = middle + 1;
      }
    }
    cutoff = low;
  }
  cutoffs[row] = cutoff;
  denominators[row] =
      static_cast<float>(cumulative[row_start + cutoff]);
}

__global__ void scatter_probabilities_kernel(
    const double* sorted_weights,
    const int32_t* sorted_indices,
    const int32_t* cutoffs,
    const float* denominators,
    int64_t total_size,
    int64_t row_size,
    float* probabilities) {
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (offset >= total_size) {
    return;
  }
  const int64_t row = offset / row_size;
  const int64_t rank = offset - row * row_size;
  const int32_t token = sorted_indices[offset];
  probabilities[row * row_size + token] = rank <= cutoffs[row]
      ? static_cast<float>(sorted_weights[offset]) / denominators[row]
      : 0.0f;
}

__global__ void initialize_rng_kernel(DeviceRngState* state, uint64_t seed) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    state->seed = seed;
    state->counter = 0;
    state->base = 0;
  }
}

__global__ void advance_rng_kernel(
    DeviceRngState* state,
    unsigned long long count) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    state->base = atomicAdd(&state->counter, count);
  }
}

__global__ void probabilities_to_double_kernel(
    const float* probabilities,
    int64_t total_size,
    double* values) {
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (offset < total_size) {
    values[offset] = static_cast<double>(probabilities[offset]);
  }
}

__device__ float uniform_from_uint32(uint32_t value) {
  constexpr uint32_t kMantissaMask = (uint32_t{1} << 24) - 1;
  constexpr float kScale = 1.0f / static_cast<float>(uint32_t{1} << 24);
  return static_cast<float>(value & kMantissaMask) * kScale;
}

__global__ void categorical_sample_kernel(
    const double* cumulative,
    int64_t row_count,
    int64_t row_size,
    DeviceRngState* rng,
    uint64_t* tokens) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (row >= row_count) {
    return;
  }
  curandStatePhilox4_32_10_t local_rng;
  curand_init(rng->seed, 0, rng->base + row, &local_rng);
  const double coin = static_cast<double>(uniform_from_uint32(curand(&local_rng)));
  const int64_t row_start = row * row_size;
  int64_t low = 0;
  int64_t high = row_size - 1;
  if (coin > cumulative[row_start + high]) {
    tokens[row] = static_cast<uint64_t>(high);
    return;
  }
  while (low < high) {
    const int64_t middle = low + (high - low) / 2;
    if (cumulative[row_start + middle] >= coin) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  tokens[row] = static_cast<uint64_t>(low);
}

__global__ void accept_with_probability_kernel(
    const float* probabilities,
    int64_t count,
    DeviceRngState* rng,
    uint8_t* accepted) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (index >= count) {
    return;
  }
  curandStatePhilox4_32_10_t local_rng;
  curand_init(rng->seed, 0, rng->base + index, &local_rng);
  accepted[index] =
      uniform_from_uint32(curand(&local_rng)) < probabilities[index];
}

__global__ void exclude_tokens_kernel(
    float* probabilities,
    int64_t row_count,
    int64_t row_size,
    const uint64_t* excluded_tokens) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (row < row_count && excluded_tokens[row] < row_size) {
    probabilities[row * row_size + excluded_tokens[row]] = 0.0f;
  }
}

__global__ void normalize_probability_rows_kernel(
    float* probabilities,
    const double* cumulative,
    int64_t total_size,
    int64_t row_size) {
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (offset < total_size) {
    const int64_t row = offset / row_size;
    const float denominator = static_cast<float>(
        cumulative[row * row_size + row_size - 1]);
    probabilities[offset] /= denominator;
  }
}

__global__ void compute_residual_kernel(
    const float* target,
    const float* draft,
    int64_t total_size,
    float* residual,
    double* residual_double) {
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (offset < total_size) {
    const float value = fmaxf(0.0f, target[offset] - draft[offset]);
    residual[offset] = value;
    residual_double[offset] = static_cast<double>(value);
  }
}

__global__ void normalize_residual_rows_kernel(
    float* target,
    const float* residual,
    const double* cumulative,
    int64_t total_size,
    int64_t row_size) {
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (offset < total_size) {
    const int64_t row = offset / row_size;
    const float denominator = static_cast<float>(
        cumulative[row * row_size + row_size - 1]);
    if (denominator > 0.0f) {
      target[offset] = residual[offset] / denominator;
    }
  }
}

__global__ void greedy_speculative_sample_kernel(
    const uint64_t* target_tokens,
    const uint64_t* candidates,
    int64_t verify_length,
    int64_t* accepted_count,
    uint64_t* correction_token) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  *accepted_count = verify_length;
  *correction_token = target_tokens[verify_length - 1];
  for (int64_t row = 0; row < verify_length - 1; ++row) {
    if (target_tokens[row] != candidates[row + 1]) {
      *accepted_count = row + 1;
      *correction_token = target_tokens[row];
      return;
    }
  }
}

enum CorrectionMode : int32_t {
  kTargetDistribution = 0,
  kResidualDistribution = 1,
  kExcludeDraftToken = 2,
};

__global__ void select_speculative_correction_kernel(
    const float* target_probabilities,
    const float* draft_probabilities,
    const uint64_t* candidates,
    int64_t verify_length,
    int64_t row_size,
    bool draft_argmax,
    DeviceRngState* rng,
    int32_t* selection,
    int64_t* accepted_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  *accepted_count = verify_length;
  selection[0] = static_cast<int32_t>(verify_length - 1);
  selection[1] = kTargetDistribution;
  for (int64_t row = 0; row < verify_length - 1; ++row) {
    const uint64_t token = candidates[row + 1];
    const float p = target_probabilities[row * row_size + token];
    const float q = draft_argmax
        ? 1.0f
        : draft_probabilities[(row + 1) * row_size + token];
    const float acceptance = q > 0.0f ? fminf(1.0f, p / q) : 1.0f;
    curandStatePhilox4_32_10_t local_rng;
    curand_init(rng->seed, 0, rng->base + row, &local_rng);
    if (uniform_from_uint32(curand(&local_rng)) >= acceptance) {
      *accepted_count = row + 1;
      selection[0] = static_cast<int32_t>(row);
      selection[1] = draft_argmax ? kExcludeDraftToken : kResidualDistribution;
      return;
    }
  }
}

__global__ void build_correction_distribution_kernel(
    const float* target_probabilities,
    const float* draft_probabilities,
    const uint64_t* candidates,
    const int32_t* selection,
    int64_t row_size,
    float* correction,
    double* correction_double) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (token >= row_size) {
    return;
  }
  const int64_t row = selection[0];
  const int32_t mode = selection[1];
  const float p = target_probabilities[row * row_size + token];
  float value = p;
  if (mode == kResidualDistribution) {
    value = fmaxf(0.0f, p - draft_probabilities[(row + 1) * row_size + token]);
  } else if (mode == kExcludeDraftToken &&
             token == static_cast<int64_t>(candidates[row + 1])) {
    value = 0.0f;
  }
  correction[token] = value;
  correction_double[token] = static_cast<double>(value);
}

__global__ void normalize_correction_distribution_kernel(
    const float* target_probabilities,
    const int32_t* selection,
    const double* cumulative,
    int64_t row_size,
    float* correction) {
  const int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x +
      threadIdx.x;
  if (token >= row_size) {
    return;
  }
  const float denominator = static_cast<float>(cumulative[row_size - 1]);
  if (denominator > 0.0f) {
    correction[token] /= denominator;
  } else {
    correction[token] =
        target_probabilities[static_cast<int64_t>(selection[0]) * row_size + token];
  }
}

} // namespace

struct SamplingWorkspace::Impl {
  int64_t row_count{0};
  int64_t row_size{0};
  int64_t total_size{0};
  float* sort_keys_in{nullptr};
  float* sort_keys_out{nullptr};
  int32_t* sort_indices_in{nullptr};
  int32_t* sort_indices_out{nullptr};
  double* weights{nullptr};
  double* cumulative{nullptr};
  int64_t* offsets{nullptr};
  int32_t* cutoffs{nullptr};
  float* denominators{nullptr};
  DeviceRngState* rng{nullptr};
  void* temporary_storage{nullptr};
  size_t temporary_storage_bytes{0};

  void release() {
    cudaFree(rng);
    cudaFree(temporary_storage);
    cudaFree(denominators);
    cudaFree(cutoffs);
    cudaFree(offsets);
    cudaFree(cumulative);
    cudaFree(weights);
    cudaFree(sort_indices_out);
    cudaFree(sort_indices_in);
    cudaFree(sort_keys_out);
    cudaFree(sort_keys_in);
    *this = Impl{};
  }
};

SamplingWorkspace::SamplingWorkspace() : impl_(new Impl()) {}

SamplingWorkspace::~SamplingWorkspace() {
  impl_->release();
  delete impl_;
}

cudaError_t SamplingWorkspace::reserve(
    int64_t row_count,
    int64_t row_size,
    cudaStream_t stream) {
  if (row_count <= 0 || row_size <= 0 ||
      row_size > std::numeric_limits<int32_t>::max() ||
      row_count > std::numeric_limits<int64_t>::max() / row_size) {
    return cudaErrorInvalidValue;
  }
  if (impl_->row_count == row_count && impl_->row_size == row_size) {
    return cudaSuccess;
  }

  impl_->release();
  impl_->row_count = row_count;
  impl_->row_size = row_size;
  impl_->total_size = row_count * row_size;
  const size_t total_size = static_cast<size_t>(impl_->total_size);

#define MUSE_GLIMMER_CUDA_ALLOCATE(member, count)                                  \
  do {                                                                     \
    const cudaError_t error = cudaMalloc(                                  \
        reinterpret_cast<void**>(&impl_->member),                          \
        static_cast<size_t>(count) * sizeof(*impl_->member));              \
    if (error != cudaSuccess) {                                            \
      impl_->release();                                                    \
      return error;                                                        \
    }                                                                      \
  } while (false)

  MUSE_GLIMMER_CUDA_ALLOCATE(sort_keys_in, total_size);
  MUSE_GLIMMER_CUDA_ALLOCATE(sort_keys_out, total_size);
  MUSE_GLIMMER_CUDA_ALLOCATE(sort_indices_in, total_size);
  MUSE_GLIMMER_CUDA_ALLOCATE(sort_indices_out, total_size);
  MUSE_GLIMMER_CUDA_ALLOCATE(weights, total_size);
  MUSE_GLIMMER_CUDA_ALLOCATE(cumulative, total_size);
  MUSE_GLIMMER_CUDA_ALLOCATE(offsets, row_count + 1);
  MUSE_GLIMMER_CUDA_ALLOCATE(cutoffs, row_count);
  MUSE_GLIMMER_CUDA_ALLOCATE(denominators, row_count);
  MUSE_GLIMMER_CUDA_ALLOCATE(rng, 1);

#undef MUSE_GLIMMER_CUDA_ALLOCATE

  const int offset_blocks = static_cast<int>(
      (row_count + 1 + kSamplingThreads - 1) / kSamplingThreads);
  initialize_offsets_kernel<<<offset_blocks, kSamplingThreads, 0, stream>>>(
      impl_->offsets, row_count, row_size);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    impl_->release();
    return error;
  }
  initialize_rng_kernel<<<1, 1, 0, stream>>>(impl_->rng, 0);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    impl_->release();
    return error;
  }

  size_t sort_bytes = 0;
  error = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      sort_bytes,
      impl_->sort_keys_in,
      impl_->sort_keys_out,
      impl_->sort_indices_in,
      impl_->sort_indices_out,
      static_cast<int>(impl_->total_size),
      static_cast<int>(row_count),
      impl_->offsets,
      impl_->offsets + 1,
      0,
      sizeof(float) * 8,
      stream);
  if (error != cudaSuccess) {
    impl_->release();
    return error;
  }

  size_t scan_bytes = 0;
  error = cub::DeviceScan::InclusiveSum(
      nullptr,
      scan_bytes,
      impl_->weights,
      impl_->cumulative,
      static_cast<int>(row_size),
      stream);
  if (error != cudaSuccess) {
    impl_->release();
    return error;
  }

  impl_->temporary_storage_bytes = std::max(sort_bytes, scan_bytes);
  error = cudaMalloc(
      &impl_->temporary_storage, impl_->temporary_storage_bytes);
  if (error != cudaSuccess) {
    impl_->release();
  }
  return error;
}

cudaError_t SamplingWorkspace::set_seed(uint64_t seed, cudaStream_t stream) {
  if (impl_->rng == nullptr) {
    return cudaErrorInvalidValue;
  }
  initialize_rng_kernel<<<1, 1, 0, stream>>>(impl_->rng, seed);
  return cudaGetLastError();
}

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

cudaError_t fill_sampling_probabilities(
    const float* logits,
    int64_t row_count,
    int64_t row_size,
    double temperature,
    int32_t top_k,
    double top_p,
    float* probabilities,
    SamplingWorkspace& workspace,
    cudaStream_t stream) {
  if (logits == nullptr || probabilities == nullptr || temperature <= 0.0 ||
      row_count <= 0 || row_size <= 0 ||
      row_count > std::numeric_limits<int>::max() ||
      row_size > std::numeric_limits<int>::max() / row_count) {
    return cudaErrorInvalidValue;
  }
  cudaError_t error = workspace.reserve(row_count, row_size, stream);
  if (error != cudaSuccess) {
    return error;
  }
  auto& state = *workspace.impl_;
  const int64_t total_size = state.total_size;
  const size_t logits_bytes = static_cast<size_t>(total_size) * sizeof(float);
  error = cudaMemcpyAsync(
      state.sort_keys_in,
      logits,
      logits_bytes,
      cudaMemcpyDeviceToDevice,
      stream);
  if (error != cudaSuccess) {
    return error;
  }

  const int item_blocks = static_cast<int>(
      (total_size + kSamplingThreads - 1) / kSamplingThreads);
  initialize_token_indices_kernel<<<
      item_blocks, kSamplingThreads, 0, stream>>>(
      state.sort_indices_in, total_size, row_size);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }

  error = cub::DeviceSegmentedRadixSort::SortPairsDescending(
      state.temporary_storage,
      state.temporary_storage_bytes,
      state.sort_keys_in,
      state.sort_keys_out,
      state.sort_indices_in,
      state.sort_indices_out,
      static_cast<int>(total_size),
      static_cast<int>(row_count),
      state.offsets,
      state.offsets + 1,
      0,
      sizeof(float) * 8,
      stream);
  if (error != cudaSuccess) {
    return error;
  }

  const int32_t retained_count =
      top_k > 0 && top_k < row_size ? top_k : static_cast<int32_t>(row_size);
  compute_sorted_weights_kernel<<<
      item_blocks, kSamplingThreads, 0, stream>>>(
      state.sort_keys_out,
      total_size,
      row_size,
      retained_count,
      1.0f / static_cast<float>(temperature),
      state.weights);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }

  for (int64_t row = 0; row < row_count; ++row) {
    error = cub::DeviceScan::InclusiveSum(
        state.temporary_storage,
        state.temporary_storage_bytes,
        state.weights + row * row_size,
        state.cumulative + row * row_size,
        static_cast<int>(row_size),
        stream);
    if (error != cudaSuccess) {
      return error;
    }
  }

  const int row_blocks = static_cast<int>(
      (row_count + kSamplingThreads - 1) / kSamplingThreads);
  find_nucleus_kernel<<<row_blocks, kSamplingThreads, 0, stream>>>(
      state.cumulative,
      row_count,
      row_size,
      retained_count,
      top_p,
      state.cutoffs,
      state.denominators);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }

  scatter_probabilities_kernel<<<
      item_blocks, kSamplingThreads, 0, stream>>>(
      state.weights,
      state.sort_indices_out,
      state.cutoffs,
      state.denominators,
      total_size,
      row_size,
      probabilities);
  return cudaGetLastError();
}

cudaError_t categorical_sample(
    const float* probabilities,
    int64_t row_count,
    int64_t row_size,
    uint64_t* tokens,
    SamplingWorkspace& workspace,
    cudaStream_t stream) {
  if (probabilities == nullptr || tokens == nullptr || row_count <= 0 ||
      row_size <= 0 || row_count > std::numeric_limits<int>::max() ||
      row_size > std::numeric_limits<int>::max() / row_count) {
    return cudaErrorInvalidValue;
  }
  cudaError_t error = workspace.reserve(row_count, row_size, stream);
  if (error != cudaSuccess) {
    return error;
  }
  auto& state = *workspace.impl_;
  const int64_t total_size = state.total_size;
  const int item_blocks = static_cast<int>(
      (total_size + kSamplingThreads - 1) / kSamplingThreads);
  probabilities_to_double_kernel<<<
      item_blocks, kSamplingThreads, 0, stream>>>(
      probabilities, total_size, state.weights);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }

  for (int64_t row = 0; row < row_count; ++row) {
    error = cub::DeviceScan::InclusiveSum(
        state.temporary_storage,
        state.temporary_storage_bytes,
        state.weights + row * row_size,
        state.cumulative + row * row_size,
        static_cast<int>(row_size),
        stream);
    if (error != cudaSuccess) {
      return error;
    }
  }
  advance_rng_kernel<<<1, 1, 0, stream>>>(state.rng, row_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  const int row_blocks = static_cast<int>(
      (row_count + kSamplingThreads - 1) / kSamplingThreads);
  categorical_sample_kernel<<<row_blocks, kSamplingThreads, 0, stream>>>(
      state.cumulative, row_count, row_size, state.rng, tokens);
  return cudaGetLastError();
}

cudaError_t accept_with_probability(
    const float* probabilities,
    int64_t count,
    uint8_t* accepted,
    SamplingWorkspace& workspace,
    cudaStream_t stream) {
  if (probabilities == nullptr || accepted == nullptr || count <= 0 ||
      workspace.impl_->rng == nullptr) {
    return cudaErrorInvalidValue;
  }
  auto& state = *workspace.impl_;
  advance_rng_kernel<<<1, 1, 0, stream>>>(state.rng, count);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  const int blocks =
      static_cast<int>((count + kSamplingThreads - 1) / kSamplingThreads);
  accept_with_probability_kernel<<<blocks, kSamplingThreads, 0, stream>>>(
      probabilities, count, state.rng, accepted);
  return cudaGetLastError();
}

cudaError_t sample_excluding_token_in_place(
    float* probabilities,
    int64_t row_count,
    int64_t row_size,
    const uint64_t* excluded_tokens,
    uint64_t* sampled_tokens,
    SamplingWorkspace& workspace,
    cudaStream_t stream) {
  if (probabilities == nullptr || excluded_tokens == nullptr ||
      sampled_tokens == nullptr || row_count <= 0 || row_size <= 0 ||
      row_count > std::numeric_limits<int>::max() ||
      row_size > std::numeric_limits<int>::max() / row_count) {
    return cudaErrorInvalidValue;
  }
  cudaError_t error = workspace.reserve(row_count, row_size, stream);
  if (error != cudaSuccess) {
    return error;
  }
  auto& state = *workspace.impl_;
  const int row_blocks = static_cast<int>(
      (row_count + kSamplingThreads - 1) / kSamplingThreads);
  exclude_tokens_kernel<<<row_blocks, kSamplingThreads, 0, stream>>>(
      probabilities, row_count, row_size, excluded_tokens);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }

  const int64_t total_size = state.total_size;
  const int item_blocks = static_cast<int>(
      (total_size + kSamplingThreads - 1) / kSamplingThreads);
  probabilities_to_double_kernel<<<
      item_blocks, kSamplingThreads, 0, stream>>>(
      probabilities, total_size, state.weights);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  for (int64_t row = 0; row < row_count; ++row) {
    error = cub::DeviceScan::InclusiveSum(
        state.temporary_storage,
        state.temporary_storage_bytes,
        state.weights + row * row_size,
        state.cumulative + row * row_size,
        static_cast<int>(row_size),
        stream);
    if (error != cudaSuccess) {
      return error;
    }
  }
  normalize_probability_rows_kernel<<<
      item_blocks, kSamplingThreads, 0, stream>>>(
      probabilities, state.cumulative, total_size, row_size);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  return categorical_sample(
      probabilities,
      row_count,
      row_size,
      sampled_tokens,
      workspace,
      stream);
}

cudaError_t sample_from_residual_in_place(
    float* target_probabilities,
    const float* draft_probabilities,
    int64_t row_count,
    int64_t row_size,
    uint64_t* sampled_tokens,
    SamplingWorkspace& workspace,
    cudaStream_t stream) {
  if (target_probabilities == nullptr || draft_probabilities == nullptr ||
      sampled_tokens == nullptr || row_count <= 0 || row_size <= 0 ||
      row_count > std::numeric_limits<int>::max() ||
      row_size > std::numeric_limits<int>::max() / row_count) {
    return cudaErrorInvalidValue;
  }
  cudaError_t error = workspace.reserve(row_count, row_size, stream);
  if (error != cudaSuccess) {
    return error;
  }
  auto& state = *workspace.impl_;
  const int64_t total_size = state.total_size;
  const int item_blocks = static_cast<int>(
      (total_size + kSamplingThreads - 1) / kSamplingThreads);
  compute_residual_kernel<<<item_blocks, kSamplingThreads, 0, stream>>>(
      target_probabilities,
      draft_probabilities,
      total_size,
      state.sort_keys_in,
      state.weights);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  for (int64_t row = 0; row < row_count; ++row) {
    error = cub::DeviceScan::InclusiveSum(
        state.temporary_storage,
        state.temporary_storage_bytes,
        state.weights + row * row_size,
        state.cumulative + row * row_size,
        static_cast<int>(row_size),
        stream);
    if (error != cudaSuccess) {
      return error;
    }
  }
  normalize_residual_rows_kernel<<<
      item_blocks, kSamplingThreads, 0, stream>>>(
      target_probabilities,
      state.sort_keys_in,
      state.cumulative,
      total_size,
      row_size);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  return categorical_sample(
      target_probabilities,
      row_count,
      row_size,
      sampled_tokens,
      workspace,
      stream);
}

cudaError_t sample_token(
    const float* logits,
    int64_t row_count,
    int64_t row_size,
    double temperature,
    int32_t top_k,
    double top_p,
    uint64_t* sampled_tokens,
    float* out_probabilities,
    bool probabilities_only,
    SamplingWorkspace& workspace,
    cudaStream_t stream) {
  if (logits == nullptr || sampled_tokens == nullptr || row_count <= 0 ||
      row_size <= 0) {
    return cudaErrorInvalidValue;
  }
  if (temperature <= 0.0) {
    return argmax_index(
        logits, row_count, row_size, sampled_tokens, stream);
  }
  cudaError_t error = workspace.reserve(row_count, row_size, stream);
  if (error != cudaSuccess) {
    return error;
  }
  float* probabilities = out_probabilities != nullptr
      ? out_probabilities
      : workspace.impl_->sort_keys_in;
  error = fill_sampling_probabilities(
      logits,
      row_count,
      row_size,
      temperature,
      top_k,
      top_p,
      probabilities,
      workspace,
      stream);
  if (error != cudaSuccess ||
      (out_probabilities != nullptr && probabilities_only)) {
    return error;
  }
  return categorical_sample(
      probabilities,
      row_count,
      row_size,
      sampled_tokens,
      workspace,
      stream);
}

cudaError_t greedy_speculative_sample(
    const uint64_t* target_tokens,
    const uint64_t* candidates,
    int64_t verify_length,
    int64_t* accepted_count,
    uint64_t* correction_token,
    cudaStream_t stream) {
  if (target_tokens == nullptr || candidates == nullptr ||
      accepted_count == nullptr || correction_token == nullptr ||
      verify_length < 1) {
    return cudaErrorInvalidValue;
  }
  greedy_speculative_sample_kernel<<<1, 1, 0, stream>>>(
      target_tokens,
      candidates,
      verify_length,
      accepted_count,
      correction_token);
  return cudaGetLastError();
}

cudaError_t stochastic_speculative_sample(
    const float* target_probabilities,
    const float* draft_probabilities,
    const uint64_t* candidates,
    int64_t verify_length,
    int64_t row_size,
    bool draft_argmax,
    int64_t* accepted_count,
    uint64_t* correction_token,
    SamplingWorkspace& workspace,
    cudaStream_t stream) {
  if (target_probabilities == nullptr || draft_probabilities == nullptr ||
      candidates == nullptr || accepted_count == nullptr ||
      correction_token == nullptr || verify_length < 2 || row_size <= 0) {
    return cudaErrorInvalidValue;
  }
  cudaError_t error = workspace.reserve(verify_length, row_size, stream);
  if (error != cudaSuccess) {
    return error;
  }
  auto& state = *workspace.impl_;
  if (verify_length > 1) {
    advance_rng_kernel<<<1, 1, 0, stream>>>(state.rng, verify_length - 1);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
  }
  select_speculative_correction_kernel<<<1, 1, 0, stream>>>(
      target_probabilities,
      draft_probabilities,
      candidates,
      verify_length,
      row_size,
      draft_argmax,
      state.rng,
      state.cutoffs,
      accepted_count);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }

  const int blocks =
      static_cast<int>((row_size + kSamplingThreads - 1) / kSamplingThreads);
  build_correction_distribution_kernel<<<blocks, kSamplingThreads, 0, stream>>>(
      target_probabilities,
      draft_probabilities,
      candidates,
      state.cutoffs,
      row_size,
      state.sort_keys_in,
      state.weights);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  error = cub::DeviceScan::InclusiveSum(
      state.temporary_storage,
      state.temporary_storage_bytes,
      state.weights,
      state.cumulative,
      static_cast<int>(row_size),
      stream);
  if (error != cudaSuccess) {
    return error;
  }
  normalize_correction_distribution_kernel<<<
      blocks, kSamplingThreads, 0, stream>>>(
      target_probabilities,
      state.cutoffs,
      state.cumulative,
      row_size,
      state.sort_keys_in);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    return error;
  }
  probabilities_to_double_kernel<<<blocks, kSamplingThreads, 0, stream>>>(
      state.sort_keys_in, row_size, state.weights);
  error = cub::DeviceScan::InclusiveSum(
      state.temporary_storage,
      state.temporary_storage_bytes,
      state.weights,
      state.cumulative,
      static_cast<int>(row_size),
      stream);
  if (error != cudaSuccess) {
    return error;
  }
  advance_rng_kernel<<<1, 1, 0, stream>>>(state.rng, 1);
  categorical_sample_kernel<<<1, 1, 0, stream>>>(
      state.cumulative, 1, row_size, state.rng, correction_token);
  return cudaGetLastError();
}

} // namespace muse_glimmer::cuda
