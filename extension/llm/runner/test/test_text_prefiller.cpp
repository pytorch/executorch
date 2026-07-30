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
        (override));
    MOCK_METHOD(
        ::executorch::runtime::Result<uint64_t>,
        prefill_chunk,
        (std::vector<uint64_t>&, int64_t&, float),
        (override));
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
  constexpr float temperature = 0.7f;
  EXPECT_CALL(*prefiller, prefill_chunk(_, _, FloatEq(temperature)))
      .Times(1)
      .WillOnce([&](std::vector<uint64_t>& tokens, int64_t& pos, float temp) {
        // Verify the tokens passed to prefill_chunk
        EXPECT_EQ(tokens.size(), prompt_tokens.size());
        for (size_t i = 0; i < tokens.size(); i++) {
          EXPECT_EQ(tokens[i], prompt_tokens[i]);
        }
        // Verify the position
        EXPECT_EQ(pos, start_pos);
        EXPECT_EQ(temp, temperature);
        return Result<uint64_t>(42);
      });

  // Call prefill
  auto result = prefiller->prefill(prompt_tokens, start_pos, temperature);

  // Verify the result
  EXPECT_EQ(result.error(), Error::Ok);
  EXPECT_EQ(result.get(), 42);
}

TEST_F(TextPrefillerTest, TwoArgumentPrefillUsesGreedyTemperature) {
  auto prefiller = createMockTextPrefiller(10);

  std::vector<uint64_t> prompt_tokens = {1, 2, 3};
  int64_t start_pos = 0;

  EXPECT_CALL(*prefiller, prefill_chunk(_, _, FloatEq(0.0f)))
      .Times(1)
      .WillOnce([](std::vector<uint64_t>&, int64_t&, float) {
        return Result<uint64_t>(42);
      });

  auto result = prefiller->prefill(prompt_tokens, start_pos);

  EXPECT_EQ(result.error(), Error::Ok);
  EXPECT_EQ(result.get(), 42);
}

TEST_F(TextPrefillerTest, PrefillAcceptsTemperatureBounds) {
  auto prefiller = createMockTextPrefiller(10);

  std::vector<uint64_t> prompt_tokens = {1, 2, 3};
  int64_t start_pos = 0;

  {
    InSequence seq;
    EXPECT_CALL(*prefiller, prefill_chunk(_, _, FloatEq(0.0f)))
        .WillOnce([](std::vector<uint64_t>&, int64_t&, float) {
          return Result<uint64_t>(41);
        });
    EXPECT_CALL(*prefiller, prefill_chunk(_, _, FloatEq(1.0f)))
        .WillOnce([](std::vector<uint64_t>&, int64_t&, float) {
          return Result<uint64_t>(42);
        });
  }

  auto greedy = prefiller->prefill(prompt_tokens, start_pos, 0.0f);
  auto max_temp = prefiller->prefill(prompt_tokens, start_pos, 1.0f);

  EXPECT_EQ(greedy.error(), Error::Ok);
  EXPECT_EQ(greedy.get(), 41);
  EXPECT_EQ(max_temp.error(), Error::Ok);
  EXPECT_EQ(max_temp.get(), 42);
}

TEST_F(TextPrefillerTest, PrefillRejectsTemperatureOutOfRange) {
  auto prefiller = createMockTextPrefiller(10);

  std::vector<uint64_t> prompt_tokens = {1, 2, 3};
  int64_t start_pos = 0;

  EXPECT_CALL(*prefiller, prefill_chunk(_, _, _)).Times(0);

  EXPECT_EQ(
      prefiller->prefill(prompt_tokens, start_pos, -0.1f).error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      prefiller->prefill(prompt_tokens, start_pos, 1.1f).error(),
      Error::InvalidArgument);
}

TEST_F(TextPrefillerTest, TwoArgumentPrefillChunkOverrideStillDispatches) {
  class LegacyPrefiller final : public TextPrefiller {
   public:
    explicit LegacyPrefiller(TextDecoderRunner* text_decoder_runner)
        : TextPrefiller(text_decoder_runner, true, true, 10) {}

    Result<uint64_t> prefill_chunk(std::vector<uint64_t>&, int64_t&) override {
      called = true;
      return Result<uint64_t>(42);
    }

    bool called = false;
  };

  LegacyPrefiller prefiller(&text_decoder_runner_);
  TextPrefiller* base = &prefiller;
  std::vector<uint64_t> prompt_tokens = {1, 2, 3};
  int64_t start_pos = 0;

  auto result = base->prefill_chunk(prompt_tokens, start_pos);

  EXPECT_EQ(result.error(), Error::Ok);
  EXPECT_EQ(result.get(), 42);
  EXPECT_TRUE(prefiller.called);
}

TEST_F(TextPrefillerTest, ChunkedPrefillSamplesOnlyLastChunkWithTemperature) {
  auto prefiller = createMockTextPrefiller(3);

  std::vector<uint64_t> prompt_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
  int64_t start_pos = 0;
  constexpr float temperature = 0.9f;

  {
    InSequence seq;
    EXPECT_CALL(*prefiller, prefill_chunk(_, _, FloatEq(0.0f)))
        .WillOnce([](std::vector<uint64_t>&, int64_t&, float) {
          return Result<uint64_t>(10);
        });
    EXPECT_CALL(*prefiller, prefill_chunk(_, _, FloatEq(0.0f)))
        .WillOnce([](std::vector<uint64_t>&, int64_t&, float) {
          return Result<uint64_t>(11);
        });
    EXPECT_CALL(*prefiller, prefill_chunk(_, _, FloatEq(temperature)))
        .WillOnce([](std::vector<uint64_t>&, int64_t&, float) {
          return Result<uint64_t>(12);
        });
  }

  auto result = prefiller->prefill(prompt_tokens, start_pos, temperature);

  EXPECT_EQ(result.error(), Error::Ok);
  EXPECT_EQ(result.get(), 12);
}

// Test that prefill() calls prefill_chunk() multiple times when prompt tokens >
// max_seq_len
TEST_F(
    TextPrefillerTest,
    PrefillCallsPrefillChunkMultipleTimesWhenPromptExceedsMaxLen) {
  // Create a real TextPrefiller with max_seq_len = 3 and parallel prefill
  const int64_t max_seq_len = 3;
  auto prefiller = createTextPrefiller(max_seq_len, true, true);

  // Create prompt tokens with size > max_seq_len
  std::vector<uint64_t> prompt_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
  int64_t start_pos = 0;

  // Track all tokens and positions passed to text_decoder_runner step
  struct StepCall {
    std::vector<uint64_t> tokens;
    int64_t pos;
  };
  std::vector<StepCall> step_calls;

  // Set up expectations for text_decoder_runner step calls
  EXPECT_CALL(text_decoder_runner_, step(_, _))
      .Times(3) // Should be called 3 times for 3 chunks
      .WillRepeatedly(
          [&](executorch::extension::TensorPtr& tokens, int64_t pos) {
            // Extract token values from tensor
            std::vector<uint64_t> token_values;
            int64_t num_tokens = tokens->size(1);
            auto* token_data = tokens->const_data_ptr<int64_t>();
            for (int64_t i = 0; i < num_tokens; i++) {
              token_values.push_back(static_cast<uint64_t>(token_data[i]));
            }
            step_calls.push_back({token_values, pos});
            return Result<executorch::aten::Tensor>(tensor);
          });

  // Call prefill
  auto result = prefiller->prefill(prompt_tokens, start_pos);

  // Verify the result
  EXPECT_EQ(result.error(), Error::Ok);

  // Verify that step was called 3 times with correct tokens and positions
  ASSERT_EQ(step_calls.size(), 3);

  // First chunk: tokens [1, 2, 3] at position 0
  EXPECT_EQ(step_calls[0].tokens.size(), 3);
  EXPECT_EQ(step_calls[0].tokens[0], 1);
  EXPECT_EQ(step_calls[0].tokens[1], 2);
  EXPECT_EQ(step_calls[0].tokens[2], 3);
  EXPECT_EQ(step_calls[0].pos, 0);

  // Second chunk: tokens [4, 5, 6] at position 3
  EXPECT_EQ(step_calls[1].tokens.size(), 3);
  EXPECT_EQ(step_calls[1].tokens[0], 4);
  EXPECT_EQ(step_calls[1].tokens[1], 5);
  EXPECT_EQ(step_calls[1].tokens[2], 6);
  EXPECT_EQ(step_calls[1].pos, 3);

  // Third chunk: tokens [7, 8] at position 6
  EXPECT_EQ(step_calls[2].tokens.size(), 2);
  EXPECT_EQ(step_calls[2].tokens[0], 7);
  EXPECT_EQ(step_calls[2].tokens[1], 8);
  EXPECT_EQ(step_calls[2].pos, 6);

  // Verify that start_pos has been updated correctly
  EXPECT_EQ(start_pos, prompt_tokens.size());
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
    EXPECT_CALL(*prefiller, prefill_chunk(_, _, _))
        .WillOnce([&](std::vector<uint64_t>& tokens, int64_t& pos, float) {
          return Result<uint64_t>(10);
        });

    // Second chunk: tokens [4, 5] - fails
    EXPECT_CALL(*prefiller, prefill_chunk(_, _, _))
        .WillOnce([&](std::vector<uint64_t>& tokens, int64_t& pos, float) {
          return Result<uint64_t>(Error::InvalidArgument);
        });
  }

  // Call prefill
  auto result = prefiller->prefill(prompt_tokens, start_pos);

  // Verify that the error is propagated
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

TEST_F(TextPrefillerTest, PrefillChunkRejectsTemperatureOutOfRange) {
  auto prefiller = createTextPrefiller(10, true, true);

  std::vector<uint64_t> prompt_tokens = {1, 2, 3};
  int64_t start_pos = 0;

  EXPECT_CALL(text_decoder_runner_, step(_, _)).Times(0);

  EXPECT_EQ(
      prefiller->prefill_chunk(prompt_tokens, start_pos, -0.1f).error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      prefiller->prefill_chunk(prompt_tokens, start_pos, 1.1f).error(),
      Error::InvalidArgument);
  EXPECT_EQ(start_pos, 0);
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
// Test that prefill_chunk updates start_pos correctly with parallel prefill
TEST_F(TextPrefillerTest, PrefillChunkUpdatesStartPosCorrectlyParallel) {
  // Create a TextPrefiller with parallel prefill enabled
  auto prefiller = createTextPrefiller(10, true, true);

  // Set up expectations for the text decoder runner
  int64_t captured_pos = -1;
  EXPECT_CALL(text_decoder_runner_, step(_, _))
      .WillOnce([&](executorch::extension::TensorPtr& tokens, int64_t pos) {
        captured_pos = pos;
        // Verify tokens shape is [1, num_tokens]
        EXPECT_EQ(tokens->dim(), 2);
        EXPECT_EQ(tokens->size(0), 1);
        EXPECT_EQ(tokens->size(1), 3);
        return Result<executorch::aten::Tensor>(tensor);
      });

  // Create prompt tokens
  std::vector<uint64_t> prompt_tokens = {1, 2, 3};
  int64_t start_pos = 5; // Non-zero starting position

  // Call prefill_chunk directly
  auto result = prefiller->prefill_chunk(prompt_tokens, start_pos);

  // Verify the result
  EXPECT_EQ(result.error(), Error::Ok);

  // Verify that step was called with the original start_pos
  EXPECT_EQ(captured_pos, 5);

  // Verify that start_pos has been updated by the number of tokens
  // This is the key test: start_pos should be updated exactly once
  EXPECT_EQ(start_pos, 8); // 5 + 3 tokens
}

// Test that prefill_chunk updates start_pos correctly with sequential prefill
TEST_F(TextPrefillerTest, PrefillChunkUpdatesStartPosCorrectlySequential) {
  // Create a TextPrefiller with sequential prefill (parallel disabled)
  auto prefiller = createTextPrefiller(10, true, false);

  // Track all positions passed to step
  std::vector<int64_t> captured_positions;
  EXPECT_CALL(text_decoder_runner_, step(_, _))
      .Times(3)
      .WillRepeatedly(
          [&](executorch::extension::TensorPtr& tokens, int64_t pos) {
            captured_positions.push_back(pos);
            // Verify tokens shape is [1, 1] for sequential prefill
            EXPECT_EQ(tokens->dim(), 2);
            EXPECT_EQ(tokens->size(0), 1);
            EXPECT_EQ(tokens->size(1), 1);
            return Result<executorch::aten::Tensor>(tensor);
          });

  // Create prompt tokens
  std::vector<uint64_t> prompt_tokens = {1, 2, 3};
  int64_t start_pos = 10; // Non-zero starting position

  // Call prefill_chunk directly
  auto result = prefiller->prefill_chunk(prompt_tokens, start_pos);

  // Verify the result
  EXPECT_EQ(result.error(), Error::Ok);

  // Verify that step was called with incrementing positions
  ASSERT_EQ(captured_positions.size(), 3);
  EXPECT_EQ(captured_positions[0], 10); // First token at initial start_pos
  EXPECT_EQ(captured_positions[1], 11); // Second token at start_pos + 1
  EXPECT_EQ(captured_positions[2], 12); // Third token at start_pos + 2

  // Verify that start_pos has been updated by the number of tokens
  // This is the key test: start_pos should be updated exactly once per token
  EXPECT_EQ(start_pos, 13); // 10 + 3 tokens
}

// Test that prefill with chunking updates start_pos correctly across chunks.
// This test would have caught the bug where start_pos was being updated twice.
TEST_F(
    TextPrefillerTest,
    PrefillWithChunkingUpdatesStartPosCorrectlyAcrossChunks) {
  // Create a TextPrefiller with max_seq_len = 3 and parallel prefill
  auto prefiller = createTextPrefiller(3, true, true);

  // Track all positions passed to step
  std::vector<int64_t> captured_positions;
  EXPECT_CALL(text_decoder_runner_, step(_, _))
      .Times(3) // Should be called 3 times: [1,2,3], [4,5,6], [7,8]
      .WillRepeatedly(
          [&](executorch::extension::TensorPtr& tokens, int64_t pos) {
            captured_positions.push_back(pos);
            return Result<executorch::aten::Tensor>(tensor);
          });

  // Create prompt tokens that exceed max_seq_len
  std::vector<uint64_t> prompt_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
  int64_t start_pos = 100; // Non-zero starting position

  // Call prefill (which will chunk internally)
  auto result = prefiller->prefill(prompt_tokens, start_pos);

  // Verify the result
  EXPECT_EQ(result.error(), Error::Ok);

  // Verify that step was called with correct positions for each chunk
  // If start_pos were updated twice (the bug), these would be wrong
  ASSERT_EQ(captured_positions.size(), 3);
  EXPECT_EQ(captured_positions[0], 100); // Chunk 1: tokens [1,2,3]
  EXPECT_EQ(captured_positions[1], 103); // Chunk 2: tokens [4,5,6]
  EXPECT_EQ(captured_positions[2], 106); // Chunk 3: tokens [7,8]

  // Verify that final start_pos is correct
  // This is the key test for the bug: start_pos should be exactly
  // initial_pos + num_tokens, not double-incremented
  EXPECT_EQ(start_pos, 108); // 100 + 8 tokens
}
} // namespace
