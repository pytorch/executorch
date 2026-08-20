/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace executorch {
namespace extension {
namespace llm {

/**
 * Sample the next token from the logits tensor using a pre-configured Sampler.
 * @param logits_tensor The logits tensor.
 * @param sampler The sampler to use for token selection.
 * @return The next token.
 * @note The sampler's vocab_size must match logits_tensor's last dimension.
 */
inline int32_t sample_from_logits(
    const executorch::aten::Tensor& logits_tensor,
    Sampler& sampler) {
  int32_t result = 0;

  struct {
    [[noreturn]] void fail(torch::executor::Error /* error */) {
      ET_CHECK_MSG(false, "Unsupported dtype in sample_from_logits");
    }
  } ctx;

  ET_SWITCH_FOUR_TYPES(
      Float,
      Half,
      BFloat16,
      UInt16,
      logits_tensor.scalar_type(),
      ctx,
      "sample_from_logits",
      CTYPE,
      [&]() {
        auto* logits = logits_tensor.mutable_data_ptr<CTYPE>();
        ssize_t vocab_size = logits_tensor.size(logits_tensor.dim() - 1);
        if (logits_tensor.dim() == 3) {
          auto num_tokens = logits_tensor.size(1);
          logits += (num_tokens - 1) * vocab_size;
        }
        result = sampler.sample(logits);
      });
  return result;
}

/**
 * Sample the next token from the logits tensor.
 * @param logits_tensor The logits tensor.
 * @param temperature The temperature parameter used to control randomness in
 * sampling.
 * @return The next token.
 */
inline int32_t logits_to_token(
    const executorch::aten::Tensor& logits_tensor,
    const float temperature = 0.0f) {
  ssize_t vocab_size = logits_tensor.size(logits_tensor.dim() - 1);
  // @lint-ignore CLANGTIDY facebook-hte-Deprecated
  Sampler sampler(vocab_size, temperature);
  return sample_from_logits(logits_tensor, sampler);
}

} // namespace llm
} // namespace extension
} // namespace executorch
