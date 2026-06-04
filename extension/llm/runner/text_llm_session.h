/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Internal adapter implementing the model-agnostic LLMSession over a
// TextLLMRunner. It lives in `detail` (not a public API) so it can be named as
// a friend of TextLLMRunner: this is the *only* caller of the runner's internal
// token-step hooks (prefill_tokens/decode_one/seek/position). Server and engine
// code depend on LLMSession alone, never on this type or on TextLLMRunner.

#pragma once

#include <memory>
#include <vector>

#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>

namespace executorch::extension::llm::detail {

class TextLLMSession : public LLMSession {
 public:
  explicit TextLLMSession(std::unique_ptr<TextLLMRunner> runner);

  ::executorch::runtime::Error prefill_tokens(
      std::vector<uint64_t> tokens,
      const SamplingConfig* initial_sampling = nullptr) override;
  ::executorch::runtime::Result<DecodeResult> decode_one(
      const SamplingConfig& sampling) override;
  ::executorch::runtime::Error seek(int64_t pos) override;
  int64_t position() const override;
  ::executorch::runtime::Error reset() override;
  void stop() override;

 private:
  std::unique_ptr<TextLLMRunner> runner_;
};

} // namespace executorch::extension::llm::detail
