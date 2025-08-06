/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given a image tensor, prefill the KV cache of a multimodal LLM.

#pragma once

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace llm {

// Assuming kv cache and parallel prefill are enabled.
class ET_EXPERIMENTAL ImagePrefiller {
 public:
  explicit ImagePrefiller(::executorch::extension::Module* module)
      : module_(module) {}

  /**
   * Prefill an LLM Module with the given image input.
   * @param image The image input to the multimodal LLM.
   * @param start_pos The starting position in KV cache of the input in the LLM.
   * It's passed as reference and will be updated inside this function.
   * @return The next token of the LLM Module after prefill.
   */
  virtual ::executorch::runtime::Result<uint64_t> prefill(
      Image& image,
      int64_t& start_pos);

  virtual ::executorch::runtime::Error load();
  virtual bool is_method_loaded();

  virtual ~ImagePrefiller() = default;

 protected:
  /**
   * Sample the next token from the logits tensor.
   * @param logits_tensor The logits tensor.
   * @param temperature The temperature parameter used to control randomness in
   * sampling.
   * @return The next token.
   */
  inline uint64_t logits_to_token(
      const executorch::aten::Tensor& logits_tensor,
      const float temperature = 0.0f) {
    uint64_t result = 0;
    ET_SWITCH_THREE_TYPES(
        Float,
        Half,
        BFloat16,
        logits_tensor.scalar_type(),
        unused,
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
          // @lint-ignore CLANGTIDY facebook-hte-Deprecated
          Sampler sampler(vocab_size, temperature);
          result = sampler.sample(logits);
        });
    return result;
  }

  Module* module_;
};

} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::ImagePrefiller;
} // namespace executor
} // namespace torch
