/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/serving/detail/text_output.h>

#include <gtest/gtest.h>

#include <map>

using executorch::extension::llm::serving::detail::TextOutput;
using executorch::runtime::Error;

namespace {

class PieceTokenizer : public tokenizers::Tokenizer {
 public:
  std::map<uint64_t, std::string> pieces;
  mutable std::vector<std::pair<uint64_t, uint64_t>> decoded;

  tokenizers::Error load(const std::string&) override {
    return tokenizers::Error::Ok;
  }

  tokenizers::Result<std::string>
  decode(uint64_t previous, uint64_t token, bool) const override {
    decoded.emplace_back(previous, token);
    auto it = pieces.find(token);
    if (it == pieces.end()) {
      return tokenizers::Error::Internal;
    }
    return it->second;
  }

  tokenizers::Result<std::vector<uint64_t>>
  encode(const std::string&, int8_t, int8_t) const override {
    return tokenizers::Error::Internal;
  }

  tokenizers::Result<std::string> id_to_piece(uint64_t) const override {
    return tokenizers::Error::Internal;
  }

  tokenizers::Result<uint64_t> piece_to_id(const std::string&) const override {
    return tokenizers::Error::Internal;
  }
};

} // namespace

TEST(TextOutputTest, ExcludesTerminalTokenAndKeepsDecodeContext) {
  PieceTokenizer tokenizer;
  tokenizer.pieces = {{1, "hi"}, {2, "!"}};
  std::string text;
  TextOutput output(tokenizer, 8, {99}, {}, [&](const auto& s) { text += s; });
  EXPECT_EQ(output.append({1, 2, 99, 3}), Error::Ok);
  EXPECT_TRUE(output.stopped());
  output.finish();
  EXPECT_EQ(text, "hi!");
  EXPECT_EQ(output.completion_tokens(), 2u);
  ASSERT_TRUE(output.generated_token_ids().has_value());
  EXPECT_EQ(*output.generated_token_ids(), (std::vector<uint64_t>{1, 2}));
  EXPECT_EQ(
      tokenizer.decoded,
      (std::vector<std::pair<uint64_t, uint64_t>>{{8, 1}, {1, 2}}));
}

TEST(TextOutputTest, StringStopCanTrimInsideSpeculativeTokenOutput) {
  PieceTokenizer tokenizer;
  tokenizer.pieces = {{1, "hello EN"}, {2, "Dhidden"}, {3, "ignored"}};
  std::string text;
  TextOutput output(
      tokenizer, 0, {}, {"END"}, [&](const auto& s) { text += s; });
  EXPECT_EQ(output.append({1, 2, 3}), Error::Ok);
  EXPECT_TRUE(output.string_stopped());
  output.finish();
  EXPECT_EQ(text, "hello ");
  EXPECT_EQ(output.completion_tokens(), 2u);
  EXPECT_FALSE(output.generated_token_ids().has_value());
  EXPECT_EQ(tokenizer.decoded.size(), 2u);
}

TEST(TextOutputTest, HoldsUtf8AcrossUpdatesAndFlushesStopLookbehind) {
  PieceTokenizer tokenizer;
  tokenizer.pieces = {{1, "a\xE4"}, {2, "\xB8"}, {3, "\x96"}};
  std::string text;
  TextOutput output(
      tokenizer, 0, {}, {"END"}, [&](const auto& s) { text += s; });
  EXPECT_EQ(output.append({1}), Error::Ok);
  EXPECT_TRUE(text.empty());
  EXPECT_EQ(output.append({2, 3}), Error::Ok);
  EXPECT_EQ(text, "a");
  output.finish();
  output.finish();
  EXPECT_EQ(text, "a\xE4\xB8\x96");
  EXPECT_FALSE(output.stopped());
  EXPECT_EQ(output.completion_tokens(), 3u);
  EXPECT_TRUE(output.generated_token_ids().has_value());
  EXPECT_EQ(output.append({1}), Error::InvalidState);
}

TEST(TextOutputTest, EosStillFlushesOrdinaryHeldText) {
  PieceTokenizer tokenizer;
  tokenizer.pieces = {{1, "last"}};
  std::string text;
  TextOutput output(
      tokenizer, 0, {9}, {"END"}, [&](const auto& s) { text += s; });
  EXPECT_EQ(output.append({1, 9}), Error::Ok);
  output.finish();
  EXPECT_EQ(text, "last");
  EXPECT_EQ(output.completion_tokens(), 1u);
  EXPECT_TRUE(output.generated_token_ids().has_value());
}

TEST(TextOutputTest, DecodeFailureIsStickyAndInvalidatesReplay) {
  PieceTokenizer tokenizer;
  tokenizer.pieces = {{1, "ok"}};
  std::string text;
  TextOutput output(tokenizer, 0, {}, {}, [&](const auto& s) { text += s; });
  EXPECT_EQ(output.append({1, 2}), Error::InvalidArgument);
  EXPECT_EQ(output.append({1}), Error::InvalidArgument);
  output.finish();
  EXPECT_EQ(text, "ok");
  EXPECT_EQ(output.completion_tokens(), 1u);
  EXPECT_FALSE(output.generated_token_ids().has_value());
}

TEST(TextOutputTest, TerminalOnlyCompletionHasKnownEmptyReplay) {
  PieceTokenizer tokenizer;
  TextOutput output(tokenizer, 0, {9}, {}, nullptr);
  EXPECT_EQ(output.append({9}), Error::Ok);
  output.finish();
  EXPECT_EQ(output.completion_tokens(), 0u);
  ASSERT_TRUE(output.generated_token_ids().has_value());
  EXPECT_TRUE(output.generated_token_ids()->empty());
}
