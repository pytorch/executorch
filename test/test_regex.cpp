/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "pytorch/tokenizers/pcre2_regex.h"
#include "pytorch/tokenizers/re2_regex.h"
#include "pytorch/tokenizers/regex.h"

using namespace tokenizers;

class RegexTest : public ::testing::Test {};

// Test basic functionality
TEST_F(RegexTest, BasicMatching) {
  auto regex = TK_UNWRAP_THROW(create_regex("\\w+"));

  std::string text = "Hello world";
  auto matches = regex->find_all(text);
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0].start, 0);
  EXPECT_EQ(matches[0].end, 5);
  EXPECT_EQ(
      text.substr(matches[0].start, matches[0].end - matches[0].start),
      "Hello");
  EXPECT_EQ(matches[1].start, 6);
  EXPECT_EQ(matches[1].end, 11);
  EXPECT_EQ(
      text.substr(matches[1].start, matches[1].end - matches[1].start),
      "world");
}

// Test pattern that only PCRE2 supports (lookbehind)
TEST_F(RegexTest, Pcre2Specific) {
  const std::string pattern = "(?<=@)\\w+";

  // Verify that the factory function fallsback on a PCRE2 regex
  auto regex = TK_UNWRAP_THROW(create_regex(pattern));
  EXPECT_NE(dynamic_cast<Pcre2Regex*>(regex.get()), nullptr);

  std::string text = "user@example.com";
  auto matches = regex->find_all(text);
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0].start, 5);
  EXPECT_EQ(matches[0].end, 12);
  EXPECT_EQ(
      text.substr(matches[0].start, matches[0].end - matches[0].start),
      "example");
}

// Test complex pattern with negative lookahead that should fall back to PCRE2.
// This specific pattern is from the Qwen2.5 1.5B pretokenizer.
// https://huggingface.co/Qwen/Qwen2.5-1.5B/raw/main/tokenizer.json
TEST_F(RegexTest, ComplexPatternWithNegativeLookahead) {
  const std::string complex_pattern =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

  // Now verify that the factory function fallsback on a PCRE2 regex
  auto regex = TK_UNWRAP_THROW(create_regex(complex_pattern));
  EXPECT_NE(dynamic_cast<Pcre2Regex*>(regex.get()), nullptr);

  // Test the pattern with some sample text
  std::string text = "Hello's world\n  test";
  auto matches = regex->find_all(text);

  // We expect to match:
  // 1. "Hello" (word)
  // 2. "'s" (contraction)
  // 3. " world" (word with leading space)
  // 4. "\n" (newline)
  // 5. " " (whitespace)
  // 6. " test" (word with leading space)
  ASSERT_EQ(matches.size(), 6);

  EXPECT_EQ(matches[0].start, 0);
  EXPECT_EQ(matches[0].end, 5);
  EXPECT_EQ(
      text.substr(matches[0].start, matches[0].end - matches[0].start),
      "Hello");
  EXPECT_EQ(matches[1].start, 5);
  EXPECT_EQ(matches[1].end, 7);
  EXPECT_EQ(
      text.substr(matches[1].start, matches[1].end - matches[1].start), "'s");
  EXPECT_EQ(matches[2].start, 7);
  EXPECT_EQ(matches[2].end, 13);
  EXPECT_EQ(
      text.substr(matches[2].start, matches[2].end - matches[2].start),
      " world");
  EXPECT_EQ(matches[3].start, 13);
  EXPECT_EQ(matches[3].end, 14);
  EXPECT_EQ(
      text.substr(matches[3].start, matches[3].end - matches[3].start), "\n");
  EXPECT_EQ(matches[4].start, 14);
  EXPECT_EQ(matches[4].end, 15);
  EXPECT_EQ(
      text.substr(matches[4].start, matches[4].end - matches[4].start), " ");
  EXPECT_EQ(matches[5].start, 15);
  EXPECT_EQ(matches[5].end, 20);
  EXPECT_EQ(
      text.substr(matches[5].start, matches[5].end - matches[5].start),
      " test");
}
