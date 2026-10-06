/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/serving/types.h>
#include <executorch/runtime/core/result.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace executorch::extension::llm::serving::detail {

struct PreparedPrompt {
  std::vector<batching::Token> tokens;
};

// Default text/token-ID preparation owns its result. Model-specific inputs use
// the serving PromptPreparation callback to return owned prepared backing.
inline runtime::Result<PreparedPrompt> prepare_prompt(
    const tokenizers::Tokenizer& tokenizer,
    const PromptInput& input,
    std::size_t max_tokens = std::numeric_limits<batching::Position>::max()) {
  const auto limit = std::min(
      max_tokens,
      static_cast<std::size_t>(std::numeric_limits<batching::Position>::max()));
  PreparedPrompt prepared;
  for (const auto& segment : input.segments) {
    std::vector<batching::Token> encoded;
    const std::vector<batching::Token>* ids;
    if (const auto* text = segment.try_get_text()) {
      auto result = tokenizer.encode(*text, /*bos=*/0, /*eos=*/0);
      if (!result.ok()) {
        return runtime::Error::InvalidArgument;
      }
      encoded = std::move(*result);
      ids = &encoded;
    } else if (const auto* tokens = segment.try_get_tokens()) {
      ids = tokens;
    } else {
      return runtime::Error::InvalidArgument;
    }
    if (ids->size() > limit - prepared.tokens.size()) {
      return runtime::Error::InvalidArgument;
    }
    prepared.tokens.insert(prepared.tokens.end(), ids->begin(), ids->end());
  }
  if (prepared.tokens.empty()) {
    return runtime::Error::InvalidArgument;
  }
  return prepared;
}

} // namespace executorch::extension::llm::serving::detail
