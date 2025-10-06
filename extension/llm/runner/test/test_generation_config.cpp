/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/irunner.h>
#include <gtest/gtest.h>

using namespace ::testing;
using executorch::extension::llm::GenerationConfig;

namespace {
class GenerationConfigTest : public Test {};

TEST_F(GenerationConfigTest, TestResolveMaxNewTokensBothDefault) {
  // Test when both seq_len and max_new_tokens are -1 (default)
  GenerationConfig config;
  // Default values: seq_len = -1, max_new_tokens = -1

  // max_context_len = 100, num_prompt_tokens = 20
  // Expected: max_context_len - num_prompt_tokens = 100 - 20 = 80
  EXPECT_EQ(config.resolve_max_new_tokens(100, 20), 80);

  // max_context_len = 50, num_prompt_tokens = 30
  // Expected: max_context_len - num_prompt_tokens = 50 - 30 = 20
  EXPECT_EQ(config.resolve_max_new_tokens(50, 30), 20);

  // Edge case: num_prompt_tokens equals max_context_len
  // Expected: 0 (no tokens left)
  EXPECT_EQ(config.resolve_max_new_tokens(40, 40), 0);

  // Edge case: num_prompt_tokens exceeds max_context_len
  // Expected: 0 (no tokens left, and we ensure non-negative result)
  EXPECT_EQ(config.resolve_max_new_tokens(30, 50), 0);
}

TEST_F(GenerationConfigTest, TestResolveMaxNewTokensOnlyMaxNewTokens) {
  // Test when only max_new_tokens is specified (seq_len = -1)
  GenerationConfig config;
  config.seq_len = -1;
  config.max_new_tokens = 25;

  // max_context_len = 100, num_prompt_tokens = 20
  // Available tokens: 100 - 20 = 80
  // Expected: min(max_new_tokens, available) = min(25, 80) = 25
  EXPECT_EQ(config.resolve_max_new_tokens(100, 20), 25);

  // max_context_len = 50, num_prompt_tokens = 40
  // Available tokens: 50 - 40 = 10
  // Expected: min(max_new_tokens, available) = min(25, 10) = 10
  EXPECT_EQ(config.resolve_max_new_tokens(50, 40), 10);

  // Edge case: num_prompt_tokens equals max_context_len
  // Available tokens: 0
  // Expected: 0 (no tokens left)
  EXPECT_EQ(config.resolve_max_new_tokens(40, 40), 0);
}

TEST_F(GenerationConfigTest, TestResolveMaxNewTokensOnlySeqLen) {
  // Test when only seq_len is specified (max_new_tokens = -1)
  GenerationConfig config;
  config.seq_len = 50;
  config.max_new_tokens = -1;

  // max_context_len = 100, num_prompt_tokens = 20
  // Effective seq_len: min(seq_len, max_context_len) = min(50, 100) = 50
  // Expected: effective_seq_len - num_prompt_tokens = 50 - 20 = 30
  EXPECT_EQ(config.resolve_max_new_tokens(100, 20), 30);

  // max_context_len = 40, num_prompt_tokens = 20
  // Effective seq_len: min(seq_len, max_context_len) = min(50, 40) = 40
  // Expected: effective_seq_len - num_prompt_tokens = 40 - 20 = 20
  EXPECT_EQ(config.resolve_max_new_tokens(40, 20), 20);

  // Edge case: num_prompt_tokens equals effective seq_len
  // Expected: 0 (no tokens left)
  EXPECT_EQ(config.resolve_max_new_tokens(100, 50), 0);

  // Edge case: num_prompt_tokens exceeds effective seq_len
  // Expected: 0 (no tokens left, and we ensure non-negative result)
  EXPECT_EQ(config.resolve_max_new_tokens(100, 60), 0);
}

TEST_F(GenerationConfigTest, TestResolveMaxNewTokensBothSpecified) {
  // Test when both seq_len and max_new_tokens are specified
  GenerationConfig config;
  config.seq_len = 50;
  config.max_new_tokens = 25;

  // max_context_len = 100, num_prompt_tokens = 20
  // Effective seq_len: min(seq_len, max_context_len) = min(50, 100) = 50
  // Available tokens: effective_seq_len - num_prompt_tokens = 50 - 20 = 30
  // Expected: min(max_new_tokens, available) = min(25, 30) = 25
  EXPECT_EQ(config.resolve_max_new_tokens(100, 20), 25);

  // max_context_len = 40, num_prompt_tokens = 20
  // Effective seq_len: min(seq_len, max_context_len) = min(50, 40) = 40
  // Available tokens: effective_seq_len - num_prompt_tokens = 40 - 20 = 20
  // Expected: min(max_new_tokens, available) = min(25, 20) = 20
  EXPECT_EQ(config.resolve_max_new_tokens(40, 20), 20);

  // Edge case: num_prompt_tokens equals effective seq_len
  // Available tokens: 0
  // Expected: 0 (no tokens left)
  EXPECT_EQ(config.resolve_max_new_tokens(40, 40), 0);

  // Edge case: max_new_tokens is very small
  config.max_new_tokens = 5;
  // Available tokens: 50 - 20 = 30
  // Expected: min(max_new_tokens, available) = min(5, 30) = 5
  EXPECT_EQ(config.resolve_max_new_tokens(100, 20), 5);
}
} // namespace
