/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */

#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/platform/runtime.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace ::testing;
using executorch::extension::llm::TextDecoderRunner;
using executorch::extension::llm::TextPrefiller;
using executorch::runtime::Error;
using executorch::runtime::Result;
using executorch::runtime::testing::TensorFactory;

namespace {
// Mock class for TextDecoderRunner
class MockTextDecoderRunner : public TextDecoderRunner {
 public:
  MockTextDecoderRunner() : TextDecoderRunner(nullptr, nullptr) {}
  MOCK_METHOD(
      Result<executorch::aten::Tensor>,
      step,
      (executorch::extension::TensorPtr&, int64_t),
      ());
  MOCK_METHOD(bool, is_method_loaded, (), ());
  MOCK_METHOD(Result<uint64_t>, prefill, (std::vector<uint64_t>&, int64_t), ());
  MOCK_METHOD(::executorch::runtime::Error, load, (), ());
};

// Test fixture for TextPrefiller tests
class TextPrefillerTest : public Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
    // Set up default behavior for the text decoder runner
    ON_CALL(text_decoder_runner_, is_method_loaded())
        .WillByDefault(Return(true));
    ON_CALL(text_decoder_runner_, step)
        .WillByDefault([&](executorch::extension::TensorPtr&, int64_t) {
          return Result<executorch::aten::Tensor>(tensor);
        });
  }

  // Helper method to create a TextPrefiller with specific parameters
  std::unique_ptr<TextPrefiller> createTextPrefiller(
      int64_t max_seq_len,
      bool use_kv_cache = true,
      bool enable_parallel_prefill = false) {
    return std::make_unique<TextPrefiller>(
        &text_decoder_runner_,
        use_kv_cache,
        enable_parallel_prefill,
        max_seq_len);
  }

  // Create a mock TextPrefiller that allows us to mock prefill_chunk calls
  class MockTextPrefiller : public TextPrefiller {
   public:
    MockTextPrefiller(
        TextDecoderRunner* text_decoder_runner,
        bool use_kv_cache,
        bool enable_parallel_prefill,
        int64_t max_seq_len)
        : TextPrefiller(
              text_decoder_runner,
              use_kv_cache,
              enable_parallel_prefill,
              max_seq_len) {}

    MOCK_METHOD(
        ::executorch::runtime::Result<uint64_t>,
        prefill_chunk,
        (std::vector<uint64_t>&, int64_t&),
        ());
  };

  // Create a mock TextPrefiller
  std::unique_ptr<MockTextPrefiller> createMockTextPrefiller(
      int64_t max_seq_len,
      bool use_kv_cache = true,
      bool enable_parallel_prefill = false) {
    return std::make_unique<MockTextPrefiller>(
        &text_decoder_runner_,
        use_kv_cache,
        enable_parallel_prefill,
        max_seq_len);
  }

  MockTextDecoderRunner text_decoder_runner_;
  std::vector<float> return_logits_ = {0.1f, 0.2f, 0.3f, 0.4f};
  TensorFactory<executorch::aten::ScalarType::Float> tf;
  executorch::aten::Tensor tensor = tf.make({1, 4}, return_logits_);
};

// Test that prefill() calls prefill_chunk() once when prompt tokens <=
// max_seq_len
TEST_F(TextPrefillerTest, PrefillCallsPrefillChunkOnceWhenPromptFits) {
  // Create a spy TextPrefiller with max_seq_len = 10
  auto prefiller = createMockTextPrefiller(10);

  // Create prompt tokens with size <= max_seq_len
  std::vector<uint64_t> prompt_tokens = {1, 2, 3, 4, 5};
  int64_t start_pos = 0;

  // Expect prefill_chunk to be called exactly once with the entire prompt
  EXPECT_CALL(*prefiller, prefill_chunk(_, _))
      .Times(1)
      .WillOnce([&](std::vector<uint64_t>& tokens, int64_t& pos) {
        // Verify the tokens passed to prefill_chunk
        EXPECT_EQ(tokens.size(), prompt_tokens.size());
        for (size_t i = 0; i < tokens.size(); i++) {
          EXPECT_EQ(tokens[i], prompt_tokens[i]);
        }
        // Verify the position
        EXPECT_EQ(pos, start_pos);
        return Result<uint64_t>(42);
      });

  // Call prefill
  auto result = prefiller->prefill(prompt_tokens, start_pos);

  // Verify the result
  EXPECT_EQ(result.error(), Error::Ok);
  EXPECT_EQ(result.get(), 42);
}

// Test that prefill() calls prefill_chunk() multiple times when prompt tokens >
// max_seq_len
TEST_F(
    TextPrefillerTest,
    PrefillCallsPrefillChunkMultipleTimesWhenPromptExceedsMaxLen) {
  // Create a spy TextPrefiller with max_seq_len = 3
  const int64_t max_seq_len = 3;
  auto prefiller = createMockTextPrefiller(max_seq_len);

  // Create prompt tokens with size > max_seq_len
  std::vector<uint64_t> prompt_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
  int64_t start_pos = 0;

  // Set up expectations for prefill_chunk calls
  {
    InSequence seq; // Ensure calls happen in the expected order

    // First chunk: tokens [1, 2, 3]
    EXPECT_CALL(*prefiller, prefill_chunk(_, _))
        .WillOnce([&](std::vector<uint64_t>& tokens, int64_t& pos) {
          EXPECT_EQ(tokens.size(), 3);
          EXPECT_EQ(tokens[0], 1);
          EXPECT_EQ(tokens[1], 2);
          EXPECT_EQ(tokens[2], 3);
          EXPECT_EQ(pos, 0);
          return Result<uint64_t>(10);
        });

    // Second chunk: tokens [4, 5, 6]
    EXPECT_CALL(*prefiller, prefill_chunk(_, _))
        .WillOnce([&](std::vector<uint64_t>& tokens, int64_t& pos) {
          EXPECT_EQ(tokens.size(), 3);
          EXPECT_EQ(tokens[0], 4);
          EXPECT_EQ(tokens[1], 5);
          EXPECT_EQ(tokens[2], 6);
          EXPECT_EQ(pos, 3);
          return Result<uint64_t>(20);
        });

    // Third chunk: tokens [7, 8]
    EXPECT_CALL(*prefiller, prefill_chunk(_, _))
        .WillOnce([&](std::vector<uint64_t>& tokens, int64_t& pos) {
          EXPECT_EQ(tokens.size(), 2);
          EXPECT_EQ(tokens[0], 7);
          EXPECT_EQ(tokens[1], 8);
          EXPECT_EQ(pos, 6);
          return Result<uint64_t>(30);
        });
  }

  // Call prefill
  auto result = prefiller->prefill(prompt_tokens, start_pos);

  // Verify the result
  EXPECT_EQ(result.error(), Error::Ok);
  EXPECT_EQ(result.get(), 30); // Should return the token from the last chunk

  // Verify that start_pos has been updated correctly
  EXPECT_EQ(start_pos, prompt_tokens.size());
}

// Test that prefill() handles edge cases correctly
TEST_F(TextPrefillerTest, PrefillHandlesEdgeCasesCorrectly) {
  // Create a spy TextPrefiller with max_seq_len = 1
  const int64_t max_seq_len = 1;
  auto prefiller = createMockTextPrefiller(max_seq_len);

  // Create prompt tokens with size > max_seq_len
  std::vector<uint64_t> prompt_tokens = {1, 2, 3};
  int64_t start_pos = 5; // Non-zero starting position

  // Set up expectations for prefill_chunk calls
  {
    InSequence seq;

    // First chunk: token [1]
    EXPECT_CALL(*prefiller, prefill_chunk(_, _))
        .WillOnce([&](std::vector<uint64_t>& tokens, int64_t& pos) {
          EXPECT_EQ(tokens.size(), 1);
          EXPECT_EQ(tokens[0], 1);
          EXPECT_EQ(pos, 5);
          return Result<uint64_t>(10);
        });

    // Second chunk: token [2]
    EXPECT_CALL(*prefiller, prefill_chunk(_, _))
        .WillOnce([&](std::vector<uint64_t>& tokens, int64_t& pos) {
          EXPECT_EQ(tokens.size(), 1);
          EXPECT_EQ(tokens[0], 2);
          EXPECT_EQ(pos, 6);
          return Result<uint64_t>(20);
        });

    // Third chunk: token [3]
    EXPECT_CALL(*prefiller, prefill_chunk(_, _))
        .WillOnce([&](std::vector<uint64_t>& tokens, int64_t& pos) {
          EXPECT_EQ(tokens.size(), 1);
          EXPECT_EQ(tokens[0], 3);
          EXPECT_EQ(pos, 7);
          return Result<uint64_t>(30);
        });
  }

  // Call prefill
  auto result = prefiller->prefill(prompt_tokens, start_pos);

  // Verify the result
  EXPECT_EQ(result.error(), Error::Ok);
  EXPECT_EQ(result.get(), 30);

  // Verify that start_pos has been updated correctly
  EXPECT_EQ(start_pos, 8); // 5 (initial) + 3 (tokens)
}

// Test that prefill() handles errors from prefill_chunk correctly
TEST_F(TextPrefillerTest, PrefillHandlesPrefillChunkErrorsCorrectly) {
  // Create a spy TextPrefiller with max_seq_len = 3
  const int64_t max_seq_len = 3;
  auto prefiller = createMockTextPrefiller(max_seq_len);

  // Create prompt tokens with size > max_seq_len
  std::vector<uint64_t> prompt_tokens = {1, 2, 3, 4, 5};
  int64_t start_pos = 0;

  // Set up expectations for prefill_chunk calls
  {
    InSequence seq;

    // First chunk: tokens [1, 2, 3] - succeeds
    EXPECT_CALL(*prefiller, prefill_chunk(_, _))
        .WillOnce([&](std::vector<uint64_t>& tokens, int64_t& pos) {
          return Result<uint64_t>(10);
        });

    // Second chunk: tokens [4, 5] - fails
    EXPECT_CALL(*prefiller, prefill_chunk(_, _))
        .WillOnce([&](std::vector<uint64_t>& tokens, int64_t& pos) {
          return Result<uint64_t>(Error::InvalidArgument);
        });
  }

  // Call prefill
  auto result = prefiller->prefill(prompt_tokens, start_pos);

  // Verify that the error is propagated
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

// Test that prefill_chunk() works correctly with parallel prefill enabled
TEST_F(TextPrefillerTest, PrefillChunkWorksWithParallelPrefill) {
  // Create a TextPrefiller with parallel prefill enabled
  auto prefiller = createTextPrefiller(10, true, true);

  // Set up expectations for the text decoder runner
  ON_CALL(text_decoder_runner_, step(_, _))
      .WillByDefault([&](executorch::extension::TensorPtr&, int64_t) {
        return Result<executorch::aten::Tensor>(tensor);
      });

  // Create prompt tokens
  std::vector<uint64_t> prompt_tokens = {1, 2, 3};
  int64_t start_pos = 0;

  // Call prefill
  auto result = prefiller->prefill(prompt_tokens, start_pos);

  // Verify the result
  EXPECT_EQ(result.error(), Error::Ok);

  // Verify that start_pos has been updated correctly
  EXPECT_EQ(start_pos, prompt_tokens.size());
}
} // namespace
