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

class TokenPreparedInput final : public batching::PreparedInput {
 public:
  explicit TokenPreparedInput(
      batching::TokenInputPtr tokens,
      std::size_t offset = 0)
      : tokens_(std::move(tokens)), offset_(offset) {}
  static const void* type() {
    static char kind;
    return &kind;
  }
  const void* kind() const override {
    return type();
  }
  std::size_t size() const override {
    return tokens_->size() - offset_;
  }
  batching::Token last_prompt_token() const override {
    return tokens_->back();
  }
  batching::PreparedInputPtr suffix(std::size_t start) const override {
    return start < size()
        ? std::make_shared<TokenPreparedInput>(tokens_, offset_ + start)
        : nullptr;
  }
  batching::PrefixIdentityPtr prefix_identity() const override {
    return std::make_shared<const batching::PrefixIdentity>(
        batching::PrefixIdentity{
            {batching::TokenSpan{tokens_, offset_, size()}}});
  }
  std::vector<batching::Token> tokens() const {
    return {tokens_->begin() + offset_, tokens_->end()};
  }

 private:
  batching::TokenInputPtr tokens_;
  std::size_t offset_;
};

// Owned normalized input, an error, or cancellation (monostate).
using PreparedPromptResult =
    std::variant<batching::PreparedInputPtr, ServingError, std::monostate>;

// Run on control, outside runtime locks, after admission/options validation.
// Consumes invoked callbacks before model preparation.
inline PreparedPromptResult prepare_prompt(
    const PromptPreparationContext& context,
    PromptInput& input,
    std::optional<PromptPreparation>& prepare,
    const ModelPreparer& model_preparer = {}) {
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
    input = std::get<PromptInput>(std::move(result));
  }
  const auto limit = std::min(
      context.max_prompt_positions,
      static_cast<std::size_t>(std::numeric_limits<batching::Position>::max()));
  batching::PreparedInputPtr prepared;
  if (model_preparer) {
    auto result = model_preparer(context, input);
    if (auto* error = std::get_if<ServingError>(&result)) {
      return std::move(*error);
    }
    prepared = std::get<batching::PreparedInputPtr>(std::move(result));
  } else {
    auto result = prepare_text_prompt(context.tokenizer, input, limit);
    if (!result.ok()) {
      return ServingError{
          ErrorCode::InvalidArgument, "prompt preparation failed"};
    }
    prepared = std::make_shared<TokenPreparedInput>(
        std::make_shared<const std::vector<batching::Token>>(
            std::move(*result)));
  }
  if (context.cancelled && context.cancelled()) {
    return std::monostate{};
  }
  const auto size = prepared ? prepared->size() : 0;
  const auto identity = prepared ? prepared->prefix_identity() : nullptr;
  if (!size || size > limit || (identity && identity->size() != size)) {
    return ServingError{
        ErrorCode::InvalidArgument, "invalid prepared prompt size or identity"};
  }
  return prepared;
}

} // namespace executorch::extension::llm::serving::detail
