/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/batching/types.h>
#include <gtest/gtest.h>
#include <utility>

namespace executorch::extension::llm::serving::testing {

class TestPreparedInput : public batching::PreparedInput {
 public:
  explicit TestPreparedInput(std::size_t positions)
      : positions_(positions),
        tokens(std::make_shared<const std::vector<batching::Token>>(
            positions <= 64 ? positions : 0,
            42)) {}
  explicit TestPreparedInput(std::vector<batching::Token> values)
      : positions_(values.size()),
        tokens(std::make_shared<const std::vector<batching::Token>>(
            std::move(values))) {}
  const void* kind() const override {
    return &tag;
  }
  std::size_t size() const override {
    return positions_;
  }
  std::optional<batching::Token> initial_detokenization_token() const override {
    return previous;
  }
  batching::PreparedInputPtr suffix(std::size_t start) const override {
    if (start >= size() || (start && refuse_suffix)) {
      return nullptr;
    }
    auto view = std::make_shared<TestPreparedInput>(*this);
    view->offset += start;
    view->positions_ -= start;
    view->positions_ += invalid_suffix ? 1 : 0;
    if (identity) {
      auto metadata = std::make_shared<batching::PrefixIdentity>();
      for (auto span : identity->spans) {
        std::visit(
            [&](auto& s) {
              const auto skip = std::min(start, s.size);
              s.offset += skip;
              s.size -= skip;
              start -= skip;
              if (s.size) {
                metadata->spans.emplace_back(s);
              }
            },
            span);
      }
      view->identity = std::move(metadata);
    }
    return view;
  }
  batching::PrefixIdentityPtr prefix_identity() const override {
    return identity;
  }
  batching::PrefixIdentityPtr identity;
  std::optional<batching::Token> previous = 42;
  std::size_t offset = 0;
  bool refuse_suffix = false;
  bool invalid_suffix = false;
  inline static char tag;
  std::size_t positions_;
  // Oversized metadata-only test inputs need no corresponding allocation.
  batching::TokenInputPtr tokens;
};

inline batching::TokenInputPtr validated_backing(
    const batching::Input& slice,
    bool accept_prepared) {
  batching::TokenInputPtr source;
  if (const auto* raw = std::get_if<batching::TokenInputPtr>(&slice.payload)) {
    source = *raw;
  } else {
    const auto& prepared = std::get<batching::PreparedInputPtr>(slice.payload);
    if (!accept_prepared || !prepared ||
        prepared->kind() != &TestPreparedInput::tag) {
      ADD_FAILURE() << "unexpected opaque input reached execute";
      return {};
    }
    const auto& view = static_cast<const TestPreparedInput&>(*prepared);
    source = std::make_shared<const std::vector<batching::Token>>(
        view.tokens->begin() + view.offset, view.tokens->end());
  }
  if (!source || slice.offset > source->size() ||
      slice.size > source->size() - slice.offset) {
    ADD_FAILURE() << "input slice exceeds test backing";
    return {};
  }
  return source;
}

} // namespace executorch::extension::llm::serving::testing
