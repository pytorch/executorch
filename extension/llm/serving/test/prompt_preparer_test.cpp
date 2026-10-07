/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/serving/detail/prompt_preparer.h>

#include <gtest/gtest.h>

#include <map>
#include <tuple>

using namespace executorch::extension::llm;
using executorch::runtime::Error;

namespace {

class RecordingTokenizer : public tokenizers::Tokenizer {
 public:
  std::map<std::string, std::vector<uint64_t>> encodings;
  mutable std::vector<std::tuple<std::string, int8_t, int8_t>> calls;

  tokenizers::Error load(const std::string&) override {
    return tokenizers::Error::Ok;
  }

  tokenizers::Result<std::vector<uint64_t>>
  encode(const std::string& text, int8_t bos, int8_t eos) const override {
    calls.emplace_back(text, bos, eos);
    auto it = encodings.find(text);
    if (it == encodings.end()) {
      return tokenizers::Error::Internal;
    }
    return it->second;
  }

  tokenizers::Result<std::string> decode(uint64_t, uint64_t, bool)
      const override {
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

TEST(PromptPreparerTest, PreservesSegmentBoundariesAndExactIds) {
  RecordingTokenizer tokenizer;
  tokenizer.encodings = {{"a", {1}}, {"b", {2}}, {"ab", {3}}};
  const serving::PromptInput input{{
      make_text_input("a"),
      make_text_input("b"),
      make_token_input({0, 42}),
  }};
  auto result = serving::detail::prepare_prompt(tokenizer, input);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->tokens, (std::vector<uint64_t>{1, 2, 0, 42}));
  ASSERT_EQ(tokenizer.calls.size(), 2u);
  EXPECT_EQ(
      tokenizer.calls[0],
      std::make_tuple(std::string("a"), int8_t{0}, int8_t{0}));
  EXPECT_EQ(
      tokenizer.calls[1],
      std::make_tuple(std::string("b"), int8_t{0}, int8_t{0}));
  EXPECT_EQ(input.segments[2].get_tokens(), (std::vector<uint64_t>{0, 42}));
}

TEST(PromptPreparerTest, RejectsEmptyPreparedPrompts) {
  RecordingTokenizer tokenizer;
  tokenizer.encodings = {{"", {}}};
  EXPECT_EQ(
      serving::detail::prepare_prompt(tokenizer, {}).error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      serving::detail::prepare_prompt(
          tokenizer, serving::PromptInput{{make_text_input("")}})
          .error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      serving::detail::prepare_prompt(
          tokenizer, serving::PromptInput{{make_token_input({})}})
          .error(),
      Error::InvalidArgument);
}

TEST(PromptPreparerTest, RejectsUnsupportedModalitiesWithoutSkippingSegment) {
  RecordingTokenizer tokenizer;
  for (const auto& segment : {
           make_image_input(Image{}),
           make_audio_input(Audio{}),
           make_raw_audio_input(RawAudio{}),
       }) {
    const serving::PromptInput input{{
        make_token_input({1}),
        segment,
        make_token_input({2}),
    }};
    EXPECT_EQ(
        serving::detail::prepare_prompt(tokenizer, input).error(),
        Error::InvalidArgument);
  }
  EXPECT_TRUE(tokenizer.calls.empty());
}

TEST(PromptPreparerTest, RejectsEncodingFailureWithoutSkippingSegment) {
  RecordingTokenizer tokenizer;
  tokenizer.encodings = {{"ok", {1}}};
  const serving::PromptInput input{
      {make_text_input("missing"), make_text_input("ok")}};
  EXPECT_EQ(
      serving::detail::prepare_prompt(tokenizer, input).error(),
      Error::InvalidArgument);
  EXPECT_EQ(tokenizer.calls.size(), 1u);
}

TEST(PromptPreparerTest, EnforcesTotalTokenLimitAcrossSegments) {
  RecordingTokenizer tokenizer;
  tokenizer.encodings = {{"a", {1, 2}}};
  const serving::PromptInput input{
      {make_text_input("a"), make_token_input({3, 4})}};
  EXPECT_TRUE(serving::detail::prepare_prompt(tokenizer, input, 4).ok());
  EXPECT_EQ(
      serving::detail::prepare_prompt(tokenizer, input, 3).error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      serving::detail::prepare_prompt(tokenizer, input, 0).error(),
      Error::InvalidArgument);
}

TEST(PromptPreparerTest, IdOnlyPromptsDoNotCallTokenizer) {
  RecordingTokenizer tokenizer;
  const serving::PromptInput input{{make_token_input({4, 2, 0})}};
  auto result = serving::detail::prepare_prompt(tokenizer, input);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->tokens, (std::vector<uint64_t>{4, 2, 0}));
  EXPECT_TRUE(tokenizer.calls.empty());
}
