/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/sampler/argmax.cuh>
#include <executorch/extension/llm/sampler/cuda_sampler.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace extension {
namespace llm {
namespace cuda {

// Wrapper function that performs argmax on GPU logits tensor
// Returns the token index with the highest logit value
// logits_ptr: pointer to GPU memory containing logits
// vocab_size: vocabulary size
// scalar_type: data type of the logits tensor
// cuda_stream: CUDA stream for async execution (nullptr for default stream)
// out_token_gpu: pre-allocated GPU memory for output token (int*)
int32_t argmax_cuda(
    const void* logits_ptr,
    int vocab_size,
    ::executorch::aten::ScalarType scalar_type,
    cudaStream_t cuda_stream,
    int* out_token_gpu) {
  // Launch kernel for single row (batch size 1)
  launch_argmax_vocab_rows(
      logits_ptr,
      scalar_type,
      1, // rows = 1
      vocab_size,
      out_token_gpu,
      nullptr, // don't need max logit value
      cuda_stream);

  // Copy result back to host
  int32_t token;
  cudaError_t err = cudaMemcpyAsync(
      &token, out_token_gpu, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "Failed to copy argmax result from GPU: %s",
        cudaGetErrorString(err));
    return -1;
  }

  // Synchronize to ensure result is ready
  err = cudaStreamSynchronize(cuda_stream);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "Failed to synchronize CUDA stream: %s",
        cudaGetErrorString(err));
    return -1;
  }

  return token;
}

} // namespace cuda
} // namespace llm
} // namespace extension
} // namespace executorch
