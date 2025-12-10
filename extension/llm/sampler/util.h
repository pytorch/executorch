/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/extension/llm/sampler/sampler.h>
#ifdef CUDA_AVAILABLE
#include <executorch/extension/llm/sampler/sampler_cuda.h>
#endif

namespace executorch {
namespace extension {
namespace llm {

/**
 * Sample the next token from the logits tensor.
 * @param logits_tensor The logits tensor.
 * @param temperature The temperature parameter used to control randomness in
 * sampling.
 * @param cuda_stream Optional CUDA stream to use for device-side sampling.
 * @param workspace Optional workspace for CUDA sampling to reduce allocations.
 * @return The next token.
 */
inline int32_t logits_to_token(
    const executorch::aten::Tensor& logits_tensor,
    const float temperature = 0.0f,
    void* cuda_stream = nullptr
#ifdef CUDA_AVAILABLE
    ,
    SamplerWorkspace* workspace = nullptr
#endif
) {
  int32_t result = 0;

  // Create a minimal context for error handling in ET_SWITCH
  struct {
    [[noreturn]] void fail(torch::executor::Error /* error */) {
      ET_CHECK_MSG(false, "Unsupported dtype in logits_to_token");
    }
  } ctx;

  ET_SWITCH_FOUR_TYPES(
      Float,
      Half,
      BFloat16,
      UInt16,
      logits_tensor.scalar_type(),
      ctx,
      "logits_to_token",
      CTYPE,
      [&]() {
        // If the logit_tensor rank is 3, the shape is [batch, seq_length,
        // vocab_size], get the last logits, sample and return. Else the model
        // outputs the last logit, directly sample and return.
        auto* logits = logits_tensor.mutable_data_ptr<CTYPE>();
        ssize_t vocab_size = logits_tensor.size(logits_tensor.dim() - 1);
        if (logits_tensor.dim() == 3) {
          auto num_tokens = logits_tensor.size(1);
          logits += (num_tokens - 1) * vocab_size;
        }
#ifdef CUDA_AVAILABLE
        if (cuda_stream != nullptr) {
          CudaSampler sampler(vocab_size, temperature, cuda_stream, workspace);
          result = sampler.sample(logits);
          return;
        }
#endif
        Sampler sampler(vocab_size, temperature);
        result = sampler.sample(logits);
      });
  return result;
}

} // namespace llm
} // namespace extension
} // namespace executorch
