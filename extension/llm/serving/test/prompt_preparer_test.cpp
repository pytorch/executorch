/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/serving/detail/prompt_preparer.h>
#include <executorch/extension/llm/serving/test/prepared_input.h>

#include <gtest/gtest.h>

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <tuple>

using namespace executorch::extension::llm;
using batching::Token;
using executorch::runtime::Error;
using serving::ErrorCode;
using serving::ModelPreparer;
using serving::PromptInput;
using serving::PromptPreparation;
using serving::PromptPreparationContext;
using serving::ServingError;
using serving::testing::TestPreparedInput;

namespace {

class RecordingTokenizer : public tokenizers::Tokenizer {
 public:
  std::map<std::string, std::vector<uint64_t>> encodings;
  mutable std::vector<std::tuple<std::string, int8_t, int8_t>> calls;
  std::function<void()> on_encode;

  tokenizers::Error load(const std::string&) override {
    return tokenizers::Error::Ok;
  }

  tokenizers::Result<std::vector<uint64_t>>
  encode(const std::string& text, int8_t bos, int8_t eos) const override {
    if (on_encode) {
      on_encode();
    }
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
  auto result = serving::detail::prepare_text_prompt(tokenizer, input);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(*result, (std::vector<batching::Token>{1, 2, 0, 42}));
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
      serving::detail::prepare_text_prompt(tokenizer, {}).error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      serving::detail::prepare_text_prompt(
          tokenizer, serving::PromptInput{{make_text_input("")}})
          .error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      serving::detail::prepare_text_prompt(
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
        serving::detail::prepare_text_prompt(tokenizer, input).error(),
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
      serving::detail::prepare_text_prompt(tokenizer, input).error(),
      Error::InvalidArgument);
  EXPECT_EQ(tokenizer.calls.size(), 1u);
}

TEST(PromptPreparerTest, EnforcesTotalTokenLimitAcrossSegments) {
  RecordingTokenizer tokenizer;
  tokenizer.encodings = {{"a", {1, 2}}};
  const serving::PromptInput input{
      {make_text_input("a"), make_token_input({3, 4})}};
  EXPECT_TRUE(serving::detail::prepare_text_prompt(tokenizer, input, 4).ok());
  EXPECT_EQ(
      serving::detail::prepare_text_prompt(tokenizer, input, 3).error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      serving::detail::prepare_text_prompt(tokenizer, input, 0).error(),
      Error::InvalidArgument);
}

TEST(PromptPreparerTest, IdOnlyPromptsDoNotCallTokenizer) {
  RecordingTokenizer tokenizer;
  const serving::PromptInput input{{make_token_input({4, 2, 0})}};
  auto result = serving::detail::prepare_text_prompt(tokenizer, input);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(*result, (std::vector<batching::Token>{4, 2, 0}));
  EXPECT_TRUE(tokenizer.calls.empty());
}

TEST(PromptPreparerTest, NormalizesAndValidatesDirectAndDeferredRawPrompts) {
  RecordingTokenizer tokenizer;
  tokenizer.encodings = {{"a", {1, 2}}};
  const PromptPreparationContext context{tokenizer, 3, {}};
  for (bool deferred : {false, true}) {
    for (const auto& [raw, expected] :
         std::vector<std::pair<PromptInput, std::vector<batching::Token>>>{
             {{{make_text_input("a"), make_token_input({0})}}, {1, 2, 0}},
             {{}, {}},
             {{{make_text_input("missing")}}, {}},
             {{{make_image_input(Image{})}}, {}},
             {{{make_text_input("a"), make_token_input({3, 4})}}, {}}}) {
      SCOPED_TRACE(deferred);
      PromptInput input = raw;
      std::optional<PromptPreparation> prepare;
      if (deferred) {
        input = PromptInput{};
        prepare = [prompt = raw](const auto&) { return prompt; };
      }
      auto result = serving::detail::prepare_prompt(context, input, prepare);
      EXPECT_FALSE(prepare.has_value());
      EXPECT_EQ(input.segments.size(), raw.segments.size());
      if (expected.empty()) {
        ASSERT_TRUE(std::holds_alternative<ServingError>(result));
        EXPECT_EQ(
            std::get<ServingError>(result).code, ErrorCode::InvalidArgument);
      } else {
        ASSERT_TRUE(std::holds_alternative<batching::PreparedInputPtr>(result));
        const auto& prepared = std::get<batching::PreparedInputPtr>(result);
        EXPECT_EQ(
            static_cast<const serving::detail::TokenPreparedInput&>(*prepared)
                .tokens(),
            expected);
      }
    }
  }
}

TEST(PromptPreparerTest, MovesOpaqueBackingPreservingTagAndPreviousToken) {
  RecordingTokenizer tokenizer;
  const auto position_limit =
      static_cast<std::size_t>(std::numeric_limits<batching::Position>::max());
  for (bool deferred : {false, true}) {
    for (auto size : {std::size_t{3}, position_limit}) {
      SCOPED_TRACE(deferred);
      const PromptPreparationContext context{tokenizer, size, {}};
      auto backing = std::make_shared<TestPreparedInput>(size);
      std::weak_ptr<const batching::PreparedInput> weak = backing;
      const auto previous = std::numeric_limits<batching::Token>::max();
      backing->previous = previous;
      PromptInput input;
      ModelPreparer model = [backing = std::move(backing)](
                                const auto&, const auto&) mutable {
        return batching::PreparedInputPtr{std::exchange(backing, nullptr)};
      };
      std::optional<PromptPreparation> prepare;
      if (deferred) {
        prepare = [prompt = std::move(input)](const auto&) mutable {
          return std::move(prompt);
        };
        input = PromptInput{};
      }
      {
        auto result =
            serving::detail::prepare_prompt(context, input, prepare, model);
        EXPECT_FALSE(prepare.has_value());
        ASSERT_TRUE(std::holds_alternative<batching::PreparedInputPtr>(result));
        const auto& opaque = std::get<batching::PreparedInputPtr>(result);
        ASSERT_NE(opaque, nullptr);
        EXPECT_EQ(opaque, weak.lock());
        EXPECT_EQ(opaque->size(), size);
        EXPECT_EQ(opaque->last_prompt_token(), previous);
        EXPECT_FALSE(weak.expired());
      }
      EXPECT_TRUE(weak.expired());
    }
  }
  EXPECT_TRUE(tokenizer.calls.empty());
}

TEST(PromptPreparerTest, RejectsInvalidDirectAndDeferredOpaquePrompts) {
  RecordingTokenizer tokenizer;
  auto invalid = std::make_shared<TestPreparedInput>(3);
  invalid->identity = std::make_shared<const batching::PrefixIdentity>(
      batching::PrefixIdentity{{batching::OpaqueSpan{{7}, 0, 2}}});
  const auto beyond_position_limit =
      static_cast<std::size_t>(std::numeric_limits<batching::Position>::max()) +
      1;
  for (bool deferred : {false, true}) {
    for (const auto& [backing, limit] :
         std::vector<std::pair<batching::PreparedInputPtr, std::size_t>>{
             {nullptr, 3},
             {invalid, 3},
             {std::make_shared<TestPreparedInput>(0), 3},
             {std::make_shared<TestPreparedInput>(4), 3},
             {std::make_shared<TestPreparedInput>(1), 0},
             {std::make_shared<TestPreparedInput>(beyond_position_limit),
              beyond_position_limit}}) {
      SCOPED_TRACE(deferred);
      const PromptPreparationContext context{tokenizer, limit, {}};
      PromptInput input;
      ModelPreparer model = [backing = backing](const auto&, const auto&) {
        return backing;
      };
      std::optional<PromptPreparation> prepare;
      if (deferred) {
        prepare = [input](const auto&) { return input; };
        input = PromptInput{};
      }
      auto result =
          serving::detail::prepare_prompt(context, input, prepare, model);
      EXPECT_FALSE(prepare.has_value());
      ASSERT_TRUE(std::holds_alternative<ServingError>(result));
      EXPECT_EQ(
          std::get<ServingError>(result).code, ErrorCode::InvalidArgument);
      EXPECT_TRUE(input.segments.empty());
    }
  }
  EXPECT_TRUE(tokenizer.calls.empty());
}

TEST(PromptPreparerTest, RejectsEmptyCallbackAndPreservesCallbackError) {
  RecordingTokenizer tokenizer;
  const PromptPreparationContext context{tokenizer, 3, {}};
  PromptInput input{{make_token_input({7})}};
  std::optional<PromptPreparation> prepare{std::in_place};
  auto result = serving::detail::prepare_prompt(context, input, prepare);
  ASSERT_TRUE(std::holds_alternative<ServingError>(result));
  EXPECT_EQ(std::get<ServingError>(result).code, ErrorCode::InvalidArgument);
  prepare = [](const auto&) {
    return ServingError{ErrorCode::CapacityExceeded, "preparation detail"};
  };
  result = serving::detail::prepare_prompt(context, input, prepare);
  EXPECT_FALSE(prepare.has_value());
  ASSERT_TRUE(std::holds_alternative<ServingError>(result));
  EXPECT_EQ(std::get<ServingError>(result).code, ErrorCode::CapacityExceeded);
  EXPECT_EQ(std::get<ServingError>(result).message, "preparation detail");
  EXPECT_EQ(input.segments[0].get_tokens(), (std::vector<batching::Token>{7}));
  EXPECT_TRUE(tokenizer.calls.empty());
}

TEST(PromptPreparerTest, CancellationBeforeOrAfterCaptureRelease) {
  RecordingTokenizer tokenizer;
  for (bool before : {false, true}) {
    SCOPED_TRACE(before);
    auto capture = std::make_shared<int>(0);
    std::weak_ptr<int> weak = capture;
    bool called = false;
    const PromptPreparationContext context{
        tokenizer, 3, [&] { return before || weak.expired(); }};
    PromptInput input{{make_text_input("unused")}};
    std::optional<PromptPreparation> prepare = [capture = std::move(capture),
                                                &called](const auto&) {
      called = true;
      return ServingError{serving::ErrorCode::Internal, "cancelled error"};
    };
    auto result = serving::detail::prepare_prompt(context, input, prepare);
    EXPECT_TRUE(std::holds_alternative<std::monostate>(result));
    EXPECT_EQ(called, !before);
    EXPECT_EQ(prepare.has_value(), before);
    EXPECT_EQ(weak.expired(), !before);
    prepare.reset();
    EXPECT_TRUE(weak.expired());
  }
  EXPECT_TRUE(tokenizer.calls.empty());
}

TEST(PromptPreparerTest, ReleasesCallbackCapturesBeforeTextEncoding) {
  RecordingTokenizer tokenizer;
  tokenizer.encodings = {{"a", {9}}};
  auto capture = std::make_shared<int>(0);
  std::weak_ptr<int> weak = capture;
  PromptInput input;
  const PromptPreparationContext context{tokenizer, 1, {}};
  std::optional<PromptPreparation> prepare = [capture = std::move(capture),
                                              &context](const auto& actual) {
    EXPECT_EQ(&actual, &context);
    return PromptInput{{make_text_input("a")}};
  };
  tokenizer.on_encode = [&] {
    EXPECT_TRUE(weak.expired());
    EXPECT_FALSE(prepare.has_value());
  };
  auto result = serving::detail::prepare_prompt(
      context,
      input,
      prepare,
      [&](const auto& actual,
          const auto& source) -> serving::ModelPreparationResult {
        EXPECT_TRUE(weak.expired());
        EXPECT_FALSE(prepare.has_value());
        auto tokens =
            serving::detail::prepare_text_prompt(actual.tokenizer, source);
        return batching::PreparedInputPtr{
            std::make_shared<serving::detail::TokenPreparedInput>(
                std::make_shared<const std::vector<Token>>(
                    std::move(*tokens)))};
      });
  ASSERT_TRUE(std::holds_alternative<batching::PreparedInputPtr>(result));
  EXPECT_EQ(
      std::get<batching::PreparedInputPtr>(result)->last_prompt_token(), 9u);
  EXPECT_EQ(tokenizer.calls.size(), 1u);
}
