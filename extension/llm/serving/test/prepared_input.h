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
      : positions_(positions), tokens(positions <= 64 ? positions : 0, 42) {}
  explicit TestPreparedInput(std::vector<batching::Token> values)
      : positions_(values.size()), tokens(std::move(values)) {}
  const void* kind() const override {
    return &tag;
  }
  std::size_t size() const override {
    return positions_;
  }
  inline static char tag;
  const std::size_t positions_;
  // Oversized metadata-only test inputs need no corresponding allocation.
  const std::vector<batching::Token> tokens;
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
    source = batching::TokenInputPtr(
        prepared, &static_cast<const TestPreparedInput&>(*prepared).tokens);
  }
  if (!source || slice.offset > source->size() ||
      slice.size > source->size() - slice.offset) {
    ADD_FAILURE() << "input slice exceeds test backing";
    return {};
  }
  return source;
}

} // namespace executorch::extension::llm::serving::testing
