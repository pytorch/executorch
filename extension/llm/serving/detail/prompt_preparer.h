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
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace executorch::extension::llm::serving::detail {

inline runtime::Result<std::vector<batching::Token>> prepare_text_prompt(
    const tokenizers::Tokenizer& tokenizer,
    const PromptInput& input,
    std::size_t max_tokens = std::numeric_limits<batching::Position>::max()) {
  const auto limit = std::min(
      max_tokens,
      static_cast<std::size_t>(std::numeric_limits<batching::Position>::max()));
  std::vector<batching::Token> prepared;
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
    if (ids->size() > limit - prepared.size()) {
      return runtime::Error::InvalidArgument;
    }
    prepared.insert(prepared.end(), ids->begin(), ids->end());
  }
  if (prepared.empty()) {
    return runtime::Error::InvalidArgument;
  }
  return prepared;
}

// Owned normalized input, an error, or cancellation (monostate).
using PreparedPromptResult = std::variant<
    std::vector<batching::Token>,
    PreparedPromptInput,
    ServingError,
    std::monostate>;

// Run on control, outside runtime locks, after admission/options validation.
// Consumes invoked callbacks and opaque backing; preserves the input variant
// alternative so the runtime can distinguish opaque requests after preparation.
inline PreparedPromptResult prepare_prompt(
    const PromptPreparationContext& context,
    GenerationPrompt& input,
    std::optional<PromptPreparation>& prepare) {
  if (prepare) {
    if (context.cancelled && context.cancelled()) {
      return std::monostate{};
    }
    if (!*prepare) {
      return ServingError{
          ErrorCode::InvalidArgument, "empty prompt preparation callback"};
    }
    PromptPreparationResult result;
    {
      // Release captures before normalization, including on exception.
      PromptPreparation callback;
      callback.swap(*prepare);
      prepare.reset();
      result = callback(context);
    }
    if (context.cancelled && context.cancelled()) {
      return std::monostate{};
    }
    if (auto* error = std::get_if<ServingError>(&result)) {
      return std::move(*error);
    }
    input = std::get<GenerationPrompt>(std::move(result));
  }
  const auto limit = std::min(
      context.max_prompt_positions,
      static_cast<std::size_t>(std::numeric_limits<batching::Position>::max()));
  if (auto* opaque = std::get_if<PreparedPromptInput>(&input)) {
    const auto size = opaque->input ? opaque->input->size() : 0;
    if (size == 0 || size > limit) {
      return ServingError{
          ErrorCode::InvalidArgument, "invalid prepared prompt size"};
    }
    return std::move(*opaque);
  }
  auto result = prepare_text_prompt(
      context.tokenizer, std::get<PromptInput>(input), limit);
  if (!result.ok()) {
    return ServingError{
        ErrorCode::InvalidArgument, "prompt preparation failed"};
  }
  return std::move(*result);
}

} // namespace executorch::extension::llm::serving::detail
