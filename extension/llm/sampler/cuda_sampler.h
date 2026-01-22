/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef CUDA_AVAILABLE

#include <cuda_runtime.h>
#include <cstdint>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace executorch {
namespace extension {
namespace llm {

/**
 * CUDA-based sampler for performing argmax on GPU.
 * This class avoids memory allocation in the hot path by pre-allocating
 * scratch space on initialization.
 *
 * NOTE: This sampler uses the default CUDA stream (nullptr) rather than
 * creating its own stream. This provides implicit synchronization with
 * the CUDA backend's stream, ensuring that logits are fully written before
 * argmax reads them. See argmax.cu for detailed rationale.
 */
class CudaSampler {
 public:
  CudaSampler();
  ~CudaSampler();

  // Non-copyable
  CudaSampler(const CudaSampler&) = delete;
  CudaSampler& operator=(const CudaSampler&) = delete;

  // Movable
  CudaSampler(CudaSampler&& other) noexcept;
  CudaSampler& operator=(CudaSampler&& other) noexcept;

  /**
   * Perform argmax sampling on GPU logits.
   *
   * @param logits_ptr Pointer to GPU memory containing logits
   * @param vocab_size Vocabulary size (number of logits)
   * @param scalar_type Data type of the logits tensor
   * @return The token index with the highest logit value, or -1 on error
   */
  int32_t sample_argmax(
      const void* logits_ptr,
      int vocab_size,
      ::executorch::aten::ScalarType scalar_type);

 private:
  // Pre-allocated GPU memory for output token
  int* out_token_gpu_;
};

namespace cuda {

/**
 * Perform argmax on GPU logits tensor.
 * This is a lower-level function that requires pre-allocated GPU memory.
 *
 * @param logits_ptr Pointer to GPU memory containing logits
 * @param vocab_size Vocabulary size
 * @param scalar_type Data type of the logits
 * @param cuda_stream CUDA stream for async execution
 * @param out_token_gpu Pre-allocated GPU memory for output token
 * @return The token index with highest logit, or -1 on error
 */
int32_t argmax_cuda(
    const void* logits_ptr,
    int vocab_size,
    ::executorch::aten::ScalarType scalar_type,
    cudaStream_t cuda_stream,
    int* out_token_gpu);

} // namespace cuda

} // namespace llm
} // namespace extension
} // namespace executorch

#endif // CUDA_AVAILABLE

