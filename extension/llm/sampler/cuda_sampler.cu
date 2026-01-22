/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/sampler/cuda_sampler.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace extension {
namespace llm {

// CudaSampler implementation
//
// IMPORTANT: Stream synchronization considerations
// ------------------------------------------------
// CudaSampler uses the default CUDA stream (nullptr) rather than creating its
// own stream. This is a deliberate design choice for the following reasons:
//
// 1. The CUDA backend (cuda_backend.cpp) creates its own stream internally for
//    running the model (encoder/decoder). This stream is encapsulated inside
//    CudaDelegateHandle and is not exposed through any public API.
//
// 2. When the decoder produces logits on the backend's stream, we need those
//    logits to be fully written before we run argmax on them. Using different
//    streams without explicit synchronization could cause race conditions.
//
// 3. The legacy default stream (stream 0 / nullptr) has special synchronization
//    semantics: operations on the default stream implicitly synchronize with
//    operations on other streams in the same CUDA context. This means:
//    - The argmax kernel will wait for the decoder to finish writing logits
//    - No explicit cudaDeviceSynchronize() or cross-stream synchronization needed
//
// 4. Trade-off: Using the default stream prevents concurrent execution between
//    the sampler and other CUDA operations. However, for single-token argmax
//    on a vocabulary-sized tensor, this overhead is negligible compared to the
//    complexity of managing cross-stream synchronization.
//
// If in the future the CUDA backend exposes its stream, we could pass it here
// for tighter integration and potential pipelining opportunities.
//
CudaSampler::CudaSampler() : out_token_gpu_(nullptr) {
  // Allocate GPU memory for output token
  cudaError_t err = cudaMalloc(&out_token_gpu_, sizeof(int));
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "Failed to allocate GPU memory for CudaSampler: %s",
        cudaGetErrorString(err));
    out_token_gpu_ = nullptr;
    return;
  }
  // Note: We intentionally do NOT create a CUDA stream here.
  // We use the default stream (nullptr) for synchronization with the backend.
  // See the detailed comment above for rationale.
}

CudaSampler::~CudaSampler() {
  // Note: No stream to destroy since we use the default stream (nullptr)
  if (out_token_gpu_ != nullptr) {
    cudaFree(out_token_gpu_);
  }
}

CudaSampler::CudaSampler(CudaSampler&& other) noexcept
    : out_token_gpu_(other.out_token_gpu_) {
  other.out_token_gpu_ = nullptr;
}

CudaSampler& CudaSampler::operator=(CudaSampler&& other) noexcept {
  if (this != &other) {
    // Clean up existing resources
    if (out_token_gpu_ != nullptr) {
      cudaFree(out_token_gpu_);
    }
    // Take ownership of other's resources
    out_token_gpu_ = other.out_token_gpu_;
    other.out_token_gpu_ = nullptr;
  }
  return *this;
}

int32_t CudaSampler::sample_argmax(
    const void* logits_ptr,
    int vocab_size,
    ::executorch::aten::ScalarType scalar_type) {
  if (out_token_gpu_ == nullptr) {
    ET_LOG(Error, "CudaSampler not properly initialized");
    return -1;
  }

  // Use default stream (nullptr) for implicit synchronization with backend
  return cuda::argmax_cuda(
      logits_ptr, vocab_size, scalar_type, nullptr, out_token_gpu_);
}

} // namespace llm
} // namespace extension
} // namespace executorch

