/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */

#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/extension/llm/sampler/logit_processor.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>

using namespace ::testing;
using executorch::extension::llm::GenerationConfig;
using executorch::extension::llm::LLMServingCapacity;
using executorch::extension::llm::LogitProcessor;
using executorch::extension::llm::Stats;
using executorch::extension::llm::TextDecoderRunner;
using executorch::extension::llm::TextLLMRunner;
using executorch::extension::llm::TextPrefiller;
using executorch::extension::llm::TextTokenGenerator;
using executorch::runtime::Error;
using executorch::runtime::Result;
using executorch::runtime::testing::TensorFactory;

namespace {
// Mock classes for dependencies
class MockTokenizer : public ::tokenizers::Tokenizer {
 public:
  MOCK_METHOD(::tokenizers::Error, load, (const std::string&), (override));
  MOCK_METHOD(bool, is_loaded, (), (const, override));
  MOCK_METHOD(
      ::tokenizers::Result<std::vector<uint64_t>>,
      encode,
      (const std::string&, int8_t, int8_t),
      (const, override));
  MOCK_METHOD(
      ::tokenizers::Result<std::string>,
      decode,
      (uint64_t, uint64_t, bool),
      (const, override));
  MOCK_METHOD(
      ::tokenizers::Result<std::string>,
      id_to_piece,
      (uint64_t),
      (const, override));
  MOCK_METHOD(
      ::tokenizers::Result<uint64_t>,
      piece_to_id,
      (const std::string&),
      (const, override));
};

class MockModule : public ::executorch::extension::Module {
 public:
  MockModule() : Module("") {}
  MOCK_METHOD(
      Error,
      load,
      (const executorch::runtime::Program::Verification),
      (override));
  MOCK_METHOD(bool, is_loaded, (), (const, override));
  MOCK_METHOD(
      Result<std::vector<executorch::runtime::EValue>>,
      execute,
      (const std::string&, const std::vector<executorch::runtime::EValue>&),
      (override));
};

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

class MockTextPrefiller : public TextPrefiller {
 public:
  explicit MockTextPrefiller(TextDecoderRunner* text_decoder_runner)
      : TextPrefiller(text_decoder_runner, false, false, 0) {}
  MOCK_METHOD(
      Result<uint64_t>,
      prefill,
      (std::vector<uint64_t>&, int64_t&),
      ());
  MOCK_METHOD(::executorch::runtime::Error, load, (), ());
  MOCK_METHOD(bool, is_loaded, (), ());
};

class MaskTokenProcessor : public LogitProcessor {
 public:
  explicit MaskTokenProcessor(int32_t banned_token)
      : banned_token_(banned_token) {}

  ::executorch::runtime::Error process(
      ::executorch::aten::Tensor logits) override {
    const int32_t vocab_size = logits.size(logits.dim() - 1);
    int32_t offset = 0;
    if (logits.dim() == 3) {
      offset = (logits.size(1) - 1) * vocab_size;
    }
    float* data = logits.mutable_data_ptr<float>();
    if (banned_token_ >= 0 && banned_token_ < vocab_size) {
      data[offset + banned_token_] = -std::numeric_limits<float>::infinity();
    }
    return ::executorch::runtime::Error::Ok;
  }

 private:
  int32_t banned_token_;
};

// Callback counter class for tests
class CallbackCounter {
 public:
  CallbackCounter() : count_(0) {}

  void callback(const std::string& token) {
    (void)token;
    count_++;
  }

  int getCount() const {
    return count_;
  }

 private:
  int count_;
};

// Test fixture for Runner tests - minimal setup
class RunnerTest : public Test {
 protected:
  // Helper functions to create and set up mock objects
  std::unique_ptr<MockTokenizer> createMockTokenizer() {
    auto tokenizer = std::make_unique<MockTokenizer>();

    // Set up default behavior for the tokenizer
    ON_CALL(*tokenizer, is_loaded).WillByDefault(Return(true));
    ON_CALL(*tokenizer, encode)
        .WillByDefault([](const std::string&, int8_t, int8_t) {
          return ::tokenizers::Result<std::vector<uint64_t>>(
              std::vector<uint64_t>{1, 2, 3});
        });

    ON_CALL(*tokenizer, decode).WillByDefault([](uint64_t, uint64_t, bool) {
      return ::tokenizers::Result<std::string>("token");
    });

    ON_CALL(*tokenizer, id_to_piece).WillByDefault([](uint64_t) {
      return ::tokenizers::Result<std::string>("piece");
    });

    ON_CALL(*tokenizer, piece_to_id).WillByDefault([](const std::string&) {
      return ::tokenizers::Result<uint64_t>(0);
    });

    return tokenizer;
  }

  std::unique_ptr<MockTextDecoderRunner> createMockTextDecoderRunner() {
    auto text_decoder_runner = std::make_unique<MockTextDecoderRunner>();
    ON_CALL(*text_decoder_runner, step)
        .WillByDefault([&](executorch::extension::TensorPtr&, int64_t) {
          return Result<executorch::aten::Tensor>(tensor);
        });
    ON_CALL(*text_decoder_runner, is_method_loaded())
        .WillByDefault(Return(true));
    return text_decoder_runner;
  }

  std::unique_ptr<MockTextPrefiller> createMockTextPrefiller(
      TextDecoderRunner* text_decoder_runner) {
    auto text_prefiller =
        std::make_unique<MockTextPrefiller>(text_decoder_runner);
    ON_CALL(*text_prefiller, is_loaded()).WillByDefault(Return(true));
    // Set up default behavior for the text prefiller
    ON_CALL(*text_prefiller, prefill)
        .WillByDefault([](const std::vector<uint64_t>&, int64_t) {
          return Result<uint64_t>(4);
        });

    return text_prefiller;
  }

  std::unique_ptr<TextTokenGenerator> createTextTokenGenerator(
      ::tokenizers::Tokenizer* tokenizer,
      TextDecoderRunner* text_decoder_runner,
      Stats* stats) {
    auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
        std::unordered_set<uint64_t>{100});
    return std::make_unique<TextTokenGenerator>(
        tokenizer,
        text_decoder_runner,
        true, // use_kv_cache
        std::move(eos_ids),
        stats);
  }

  std::unordered_map<std::string, int64_t> createDefaultMetadata() {
    return {
        {"enable_dynamic_shape", false},
        {"get_max_seq_len", 128},
        {"get_max_context_len", 128},
        {"use_kv_cache", true},
    };
  }

  // Builds a loaded TextLLMRunner with default mocks whose prefiller advances
  // the position cursor by the number of tokens (so position()/seek() and the
  // capacity bound can be exercised directly).
  std::unique_ptr<TextLLMRunner> makeRunner(
      std::unordered_map<std::string, int64_t> metadata,
      std::shared_ptr<LogitProcessor> logit_processor = nullptr) {
    auto tokenizer = createMockTokenizer();
    auto text_decoder_runner = createMockTextDecoderRunner();
    auto text_prefiller = createMockTextPrefiller(text_decoder_runner.get());
    ON_CALL(*text_prefiller, prefill(_, _))
        .WillByDefault([](std::vector<uint64_t>& tokens, int64_t& pos) {
          pos += tokens.size();
          return Result<uint64_t>(42);
        });
    ON_CALL(*text_prefiller, is_loaded()).WillByDefault(Return(true));
    auto stats = std::make_unique<executorch::llm::Stats>();
    auto text_token_generator = createTextTokenGenerator(
        tokenizer.get(), text_decoder_runner.get(), stats.get());
    if (logit_processor) {
      text_token_generator->add_logit_processor(std::move(logit_processor));
    }
    auto module = std::make_unique<MockModule>();
    auto io_manager =
        std::make_unique<executorch::extension::llm::IOManager>(*module);
    auto runner = std::make_unique<TextLLMRunner>(
        std::move(metadata),
        std::unique_ptr<::tokenizers::Tokenizer>(tokenizer.release()),
        std::move(module),
        std::move(text_decoder_runner),
        std::unique_ptr<::executorch::extension::llm::TextPrefiller>(
            text_prefiller.release()),
        std::move(io_manager),
        std::move(text_token_generator),
        std::move(stats));
    runner->load();
    return runner;
  }

 protected:
  Stats stats_;
  std::vector<float> return_logits_ = {0.1f, 0.2f, 0.3f, 0.4f};
  TensorFactory<executorch::aten::ScalarType::Float> tf;
  executorch::aten::Tensor tensor = tf.make({1, 4}, return_logits_);
};

// Test that generate() calls the token callback exactly max_new_tokens times
TEST_F(RunnerTest, GenerateCallsCallbackExactlyMaxNewTokensTimes) {
  // Create mock instances using helper functions
  auto tokenizer = createMockTokenizer();
  auto text_decoder_runner = createMockTextDecoderRunner();
  auto text_prefiller = createMockTextPrefiller(text_decoder_runner.get());

  // Set up expectations for the tokenizer encode method
  ON_CALL(*tokenizer, encode(_, _, _))
      .WillByDefault([&](const std::string&, int8_t, int8_t) {
        return ::tokenizers::Result<std::vector<uint64_t>>(
            std::vector<uint64_t>{1, 2, 3});
      });

  // Set up expectations for the text prefiller
  ON_CALL(*text_prefiller, prefill(_, _))
      .WillByDefault([&](std::vector<uint64_t>&, int64_t&) {
        return (Result<uint64_t>(4));
      });

  // Set up expectations for load methods
  ON_CALL(*text_prefiller, is_loaded()).WillByDefault(Return(true));

  std::unique_ptr<executorch::llm::Stats> stats =
      std::make_unique<executorch::llm::Stats>();
  // Create a real TextTokenGenerator
  auto text_token_generator = createTextTokenGenerator(
      tokenizer.get(), text_decoder_runner.get(), stats.get());

  // Create a Runner with our mocked components
  auto module = std::make_unique<MockModule>();
  auto io_manager =
      std::make_unique<executorch::extension::llm::IOManager>(*module);
  TextLLMRunner runner(
      createDefaultMetadata(),
      std::unique_ptr<::tokenizers::Tokenizer>(tokenizer.release()),
      std::move(module),
      std::move(text_decoder_runner),
      std::unique_ptr<::executorch::extension::llm::TextPrefiller>(
          text_prefiller.release()),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats));

  // Load
  runner.load();

  // Set up the generation config with a specific max_new_tokens value
  GenerationConfig config;
  config.max_new_tokens = 10;
  config.echo = false;

  // Create a callback counter
  CallbackCounter counter;

  // Call generate with our callback
  Error err = runner.generate(
      "test prompt", config, [&counter](const std::string& token) {
        counter.callback(token);
      });

  // Verify the callback was called exactly max_new_tokens times
  // The first token is generated by prefill, and the rest by the token
  // generator
  EXPECT_EQ(counter.getCount(), config.max_new_tokens);
  EXPECT_EQ(err, Error::Ok);
}

// Test that warmup() calls generate with the warming flag set
TEST_F(RunnerTest, WarmupCallsGenerateWithWarmingFlag) {
  // Create mock instances using helper functions
  auto tokenizer = createMockTokenizer();
  auto text_decoder_runner = createMockTextDecoderRunner();
  auto text_prefiller = createMockTextPrefiller(text_decoder_runner.get());

  // Set up expectations for the tokenizer encode method
  ON_CALL(*tokenizer, encode(_, _, _))
      .WillByDefault([&](const std::string&, int8_t, int8_t) {
        return ::tokenizers::Result<std::vector<uint64_t>>(
            std::vector<uint64_t>{1, 2, 3});
      });

  // Set up expectations for the text prefiller
  ON_CALL(*text_prefiller, prefill(_, _))
      .WillByDefault([&](std::vector<uint64_t>&, int64_t&) {
        return (Result<uint64_t>(4));
      });

  // Set up expectations for load methods
  ON_CALL(*text_prefiller, is_loaded()).WillByDefault(Return(true));

  std::unique_ptr<executorch::llm::Stats> stats =
      std::make_unique<executorch::llm::Stats>();
  // Create a TextTokenGenerator
  auto text_token_generator = createTextTokenGenerator(
      tokenizer.get(), text_decoder_runner.get(), stats.get());

  // Create a Runner with our mocked components
  auto module = std::make_unique<MockModule>();
  auto io_manager =
      std::make_unique<executorch::extension::llm::IOManager>(*module);
  TextLLMRunner runner(
      createDefaultMetadata(),
      std::move(tokenizer),
      std::move(module),
      std::move(text_decoder_runner),
      std::unique_ptr<::executorch::extension::llm::TextPrefiller>(
          text_prefiller.release()),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats));

  // Load
  runner.load();

  // Call warmup
  Error err = runner.warmup("test prompt", 5);

  // Verify the result
  EXPECT_EQ(err, Error::Ok);
}

// Test that is_loaded() returns true when components are initialized
TEST_F(RunnerTest, IsLoadedReturnsTrueWhenComponentsInitialized) {
  // Create mock instances using helper functions
  auto tokenizer = createMockTokenizer();
  auto text_decoder_runner = createMockTextDecoderRunner();
  auto text_prefiller = createMockTextPrefiller(text_decoder_runner.get());

  std::unique_ptr<executorch::llm::Stats> stats =
      std::make_unique<executorch::llm::Stats>();
  // Create a real TextTokenGenerator
  auto text_token_generator = createTextTokenGenerator(
      tokenizer.get(), text_decoder_runner.get(), stats.get());

  // Create a Runner with our mocked components
  auto module = std::make_unique<MockModule>();
  auto io_manager =
      std::make_unique<executorch::extension::llm::IOManager>(*module);
  TextLLMRunner runner(
      createDefaultMetadata(),
      std::unique_ptr<::tokenizers::Tokenizer>(tokenizer.release()),
      std::move(module),
      std::move(text_decoder_runner),
      std::unique_ptr<::executorch::extension::llm::TextPrefiller>(
          text_prefiller.release()),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats));

  // Load
  runner.load();

  // Verify is_loaded returns true
  EXPECT_TRUE(runner.is_loaded());
}

// Test that prefill() returns the predicted next token
TEST_F(RunnerTest, PrefillReturnsNextToken) {
  auto tokenizer = createMockTokenizer();
  auto text_decoder_runner = createMockTextDecoderRunner();
  auto text_prefiller = createMockTextPrefiller(text_decoder_runner.get());

  ON_CALL(*tokenizer, encode(_, _, _))
      .WillByDefault([&](const std::string&, int8_t, int8_t) {
        return ::tokenizers::Result<std::vector<uint64_t>>(
            std::vector<uint64_t>{1, 2, 3});
      });

  ON_CALL(*text_prefiller, prefill(_, _))
      .WillByDefault([&](std::vector<uint64_t>& tokens, int64_t& pos) {
        pos += tokens.size();
        return Result<uint64_t>(42);
      });

  ON_CALL(*text_prefiller, is_loaded()).WillByDefault(Return(true));

  std::unique_ptr<executorch::llm::Stats> stats =
      std::make_unique<executorch::llm::Stats>();
  auto text_token_generator = createTextTokenGenerator(
      tokenizer.get(), text_decoder_runner.get(), stats.get());

  auto module = std::make_unique<MockModule>();
  auto io_manager =
      std::make_unique<executorch::extension::llm::IOManager>(*module);
  TextLLMRunner runner(
      createDefaultMetadata(),
      std::unique_ptr<::tokenizers::Tokenizer>(tokenizer.release()),
      std::move(module),
      std::move(text_decoder_runner),
      std::unique_ptr<::executorch::extension::llm::TextPrefiller>(
          text_prefiller.release()),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats));

  runner.load();

  auto result = runner.prefill("system prompt", 1, 0);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.get(), 42);
}

// Test the prefill() → generate("") workflow
TEST_F(RunnerTest, PrefillThenGenerateEmpty) {
  auto tokenizer = createMockTokenizer();
  auto text_decoder_runner = createMockTextDecoderRunner();
  auto text_prefiller = createMockTextPrefiller(text_decoder_runner.get());

  ON_CALL(*tokenizer, encode(_, _, _))
      .WillByDefault([&](const std::string&, int8_t, int8_t) {
        return ::tokenizers::Result<std::vector<uint64_t>>(
            std::vector<uint64_t>{1, 2, 3});
      });

  ON_CALL(*text_prefiller, prefill(_, _))
      .WillByDefault([&](std::vector<uint64_t>& tokens, int64_t& pos) {
        pos += tokens.size();
        return Result<uint64_t>(4);
      });

  ON_CALL(*text_prefiller, is_loaded()).WillByDefault(Return(true));

  std::unique_ptr<executorch::llm::Stats> stats =
      std::make_unique<executorch::llm::Stats>();
  auto text_token_generator = createTextTokenGenerator(
      tokenizer.get(), text_decoder_runner.get(), stats.get());

  auto module = std::make_unique<MockModule>();
  auto io_manager =
      std::make_unique<executorch::extension::llm::IOManager>(*module);
  TextLLMRunner runner(
      createDefaultMetadata(),
      std::unique_ptr<::tokenizers::Tokenizer>(tokenizer.release()),
      std::move(module),
      std::move(text_decoder_runner),
      std::unique_ptr<::executorch::extension::llm::TextPrefiller>(
          text_prefiller.release()),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats));

  runner.load();

  // Prefill first
  auto prefill_result = runner.prefill("system prompt", 1, 0);
  EXPECT_TRUE(prefill_result.ok());

  // Generate with empty prompt — should consume prefill_next_token_
  GenerationConfig config;
  config.max_new_tokens = 5;
  config.echo = false;

  CallbackCounter counter;
  Error err = runner.generate("", config, [&counter](const std::string& token) {
    counter.callback(token);
  });

  EXPECT_EQ(err, Error::Ok);
  // First token from prefill + remaining from decode loop
  EXPECT_EQ(counter.getCount(), config.max_new_tokens);
}

// prefill_tokens() must reject a suffix larger than the KV cache, mirroring the
// capacity bound generate(prompt) enforces (prefill_tokens is the public
// prefix-cache primitive and the only place this is checked for it).
TEST_F(RunnerTest, PrefillTokensRejectsOverContext) {
  auto runner = makeRunner(createDefaultMetadata()); // max_context_len = 128
  auto result =
      runner->prefill_tokens(std::vector<uint64_t>(200, 1)); // 200 > 128
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

// seek() + prefill_tokens() across the boundary is rejected: a valid prefill
// followed by a suffix that pushes pos_ past max_context_len must fail.
TEST_F(RunnerTest, PrefillTokensRejectsWhenPosPlusSuffixExceedsContext) {
  auto runner = makeRunner(createDefaultMetadata()); // 128
  auto ok = runner->prefill_tokens(std::vector<uint64_t>(100, 1));
  EXPECT_TRUE(ok.ok());
  EXPECT_EQ(runner->position(), 100);
  auto bad =
      runner->prefill_tokens(std::vector<uint64_t>(50, 1)); // 100 + 50 > 128
  EXPECT_FALSE(bad.ok());
  EXPECT_EQ(bad.error(), Error::InvalidArgument);
  EXPECT_EQ(runner->position(), 100); // rejected before advancing
}

// Empty tokens are rejected.
TEST_F(RunnerTest, PrefillTokensRejectsEmpty) {
  auto runner = makeRunner(createDefaultMetadata());
  auto result = runner->prefill_tokens(std::vector<uint64_t>{});
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

// position() tracks prefilled tokens; seek() rewinds within range and rejects
// out-of-range targets.
TEST_F(RunnerTest, SeekAndPositionTrackResidentTokens) {
  auto runner = makeRunner(createDefaultMetadata());
  EXPECT_EQ(runner->position(), 0);
  EXPECT_TRUE(runner->prefill_tokens(std::vector<uint64_t>(10, 1)).ok());
  EXPECT_EQ(runner->position(), 10);
  EXPECT_EQ(runner->seek(5), Error::Ok);
  EXPECT_EQ(runner->position(), 5);
  EXPECT_EQ(runner->seek(999), Error::InvalidArgument); // past current position
}

// seek() is refused on sliding-window models (max_seq_len < max_context_len),
// so the prefix cache falls back to a full reset+prefill instead of corrupting.
TEST_F(RunnerTest, SeekRejectedForSlidingWindow) {
  auto md = createDefaultMetadata();
  md["get_max_seq_len"] = 64; // < get_max_context_len (128) => sliding window
  auto runner = makeRunner(md);
  EXPECT_EQ(runner->seek(0), Error::NotSupported);
}

// decode_one() emits the pending token id exactly, then forwards it (advancing
// position by one). Looping it reproduces a generation while exposing ids.
TEST_F(RunnerTest, DecodeOneReturnsExactTokenIdAndAdvances) {
  auto runner = makeRunner(createDefaultMetadata());
  ASSERT_TRUE(runner->prefill_tokens(std::vector<uint64_t>{1, 2, 3}).ok());
  EXPECT_EQ(runner->position(), 3);

  auto r1 = runner->decode_one(0.0f);
  ASSERT_TRUE(r1.ok());
  EXPECT_EQ(r1.get().token_id, 42u); // the prefill-pending token (mock prefill)
  EXPECT_FALSE(r1.get().is_eos);
  EXPECT_EQ(runner->position(), 4); // forwarded one token

  auto r2 = runner->decode_one(0.0f);
  ASSERT_TRUE(r2.ok());
  EXPECT_EQ(r2.get().token_id, 3u); // argmax of canned logits {.1,.2,.3,.4}
  EXPECT_EQ(runner->position(), 5);
}

// decode_one() without a pending token (no prior prefill) must error.
TEST_F(RunnerTest, DecodeOneWithoutPendingTokenFails) {
  auto runner = makeRunner(createDefaultMetadata());
  EXPECT_FALSE(runner->decode_one(0.0f).ok());
}

// decode_one() is the last-resort safety net for session drivers: even if a
// caller forgets to resolve max_new_tokens, it must not step past KV capacity.
TEST_F(RunnerTest, DecodeOneRejectsWhenContextFull) {
  auto md = createDefaultMetadata(); // max_context_len = 128
  auto runner = makeRunner(md);
  ASSERT_TRUE(runner->prefill_tokens(std::vector<uint64_t>(127, 1)).ok());
  auto r1 = runner->decode_one(0.0f);
  ASSERT_TRUE(r1.ok());
  EXPECT_EQ(runner->position(), 128);

  auto r2 = runner->decode_one(0.0f);
  EXPECT_FALSE(r2.ok());
  EXPECT_EQ(r2.error(), Error::InvalidArgument);
}

// decode_one() must apply the generator's logit processors before sampling,
// exactly like generate(). Argmax of {0.1,0.2,0.3,0.4} is token 3; masking it
// makes the next sampled (pending) token 2 — proving the session decode path
// honors grammar/tool masks/penalties and can't diverge from generate().
TEST_F(RunnerTest, DecodeOneAppliesLogitProcessors) {
  auto runner = makeRunner(
      createDefaultMetadata(), std::make_shared<MaskTokenProcessor>(3));
  ASSERT_TRUE(runner->prefill_tokens(std::vector<uint64_t>{1, 2, 3}).ok());

  // r1 emits the prefill-pending token (42, not sampled here); its forward pass
  // samples the next pending token from masked logits.
  ASSERT_TRUE(runner->decode_one(0.0f).ok());
  auto r2 = runner->decode_one(0.0f);
  ASSERT_TRUE(r2.ok());
  EXPECT_EQ(r2.get().token_id, 2u); // token 3 masked -> argmax is now 2
}

// v1 serving capacity is conservatively single-slot: a self-contained .pte
// repacks XNNPACK weights per runtime, so we don't claim shared physical
// sessions. (TextLLMEngine::serving_capacity() returns this default; the
// engine-backed end-to-end check is in the pybinding test.)
TEST_F(RunnerTest, ServingCapacityIsSingleSlotByDefault) {
  LLMServingCapacity cap;
  EXPECT_EQ(cap.max_physical_sessions_without_weight_duplication, 1);
  EXPECT_EQ(cap.estimated_bytes_per_session, 0);
}

// Test that generate("") without prior prefill() returns an error
TEST_F(RunnerTest, GenerateEmptyWithoutPrefillFails) {
  auto tokenizer = createMockTokenizer();
  auto text_decoder_runner = createMockTextDecoderRunner();
  auto text_prefiller = createMockTextPrefiller(text_decoder_runner.get());

  ON_CALL(*text_prefiller, is_loaded()).WillByDefault(Return(true));

  std::unique_ptr<executorch::llm::Stats> stats =
      std::make_unique<executorch::llm::Stats>();
  auto text_token_generator = createTextTokenGenerator(
      tokenizer.get(), text_decoder_runner.get(), stats.get());

  auto module = std::make_unique<MockModule>();
  auto io_manager =
      std::make_unique<executorch::extension::llm::IOManager>(*module);
  TextLLMRunner runner(
      createDefaultMetadata(),
      std::unique_ptr<::tokenizers::Tokenizer>(tokenizer.release()),
      std::move(module),
      std::move(text_decoder_runner),
      std::unique_ptr<::executorch::extension::llm::TextPrefiller>(
          text_prefiller.release()),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats));

  runner.load();

  GenerationConfig config;
  Error err = runner.generate("", config);
  EXPECT_EQ(err, Error::InvalidState);
}

// Test that TextTokenGenerator works correctly in non-kv-cache mode.
// Exercises the code path fixed by reserving capacity before from_blob:
// without reserve(), vector reallocation would invalidate the data pointer.
TEST_F(RunnerTest, NonKvCacheGenerateCompletesSuccessfully) {
  auto tokenizer = createMockTokenizer();
  auto text_decoder_runner = createMockTextDecoderRunner();

  // In non-kv-cache mode, the input tensor should grow by 1 token each step.
  // Verify data is readable each time (catches dangling pointers under ASan).
  int step_count = 0;
  ON_CALL(*text_decoder_runner, step)
      .WillByDefault(
          [&](executorch::extension::TensorPtr& tokens_tensor, int64_t) {
            // Initial tokens = 4 (prompt 1,2,3 + prefill token 4).
            // Each step appends one token before the next call.
            int64_t expected_size = 4 + step_count;
            EXPECT_EQ(tokens_tensor->size(1), expected_size);

            // Read data to verify the pointer is still valid.
            auto* data = tokens_tensor->const_data_ptr<int64_t>();
            EXPECT_EQ(data[0], 1); // first prompt token
            EXPECT_EQ(data[1], 2);
            EXPECT_EQ(data[2], 3);
            EXPECT_EQ(data[3], 4); // prefill token

            step_count++;
            return Result<executorch::aten::Tensor>(tensor);
          });

  Stats stats;
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>(
      std::unordered_set<uint64_t>{100});
  TextTokenGenerator generator(
      tokenizer.get(),
      text_decoder_runner.get(),
      false, // use_kv_cache = false
      std::move(eos_ids),
      &stats);

  // 4 tokens: prompt (1,2,3) + prefill token (4)
  std::vector<uint64_t> tokens = {1, 2, 3, 4};
  // Generate enough tokens that the vector would reallocate without reserve.
  int32_t max_new_tokens = 20;

  auto result = generator.generate(
      tokens, 4, max_new_tokens, 0.0f, [](const std::string&) {});

  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.get(), max_new_tokens);
  EXPECT_EQ(step_count, max_new_tokens);
}

// Test that multi-turn generation with seq_len correctly accounts for pos_.
// Regression test for a bug where max_context_len was pre-adjusted by pos_,
// causing resolve_max_new_tokens to under-count occupied positions when
// seq_len is set.
TEST_F(RunnerTest, MultiTurnWithSeqLenRespectsPos) {
  auto tokenizer = createMockTokenizer();
  auto text_decoder_runner = createMockTextDecoderRunner();
  auto text_prefiller = createMockTextPrefiller(text_decoder_runner.get());

  ON_CALL(*tokenizer, encode(_, _, _))
      .WillByDefault([&](const std::string&, int8_t, int8_t) {
        return ::tokenizers::Result<std::vector<uint64_t>>(
            std::vector<uint64_t>{1, 2, 3});
      });

  ON_CALL(*text_prefiller, prefill(_, _))
      .WillByDefault([&](std::vector<uint64_t>& tokens, int64_t& pos) {
        pos += tokens.size();
        return Result<uint64_t>(4);
      });

  ON_CALL(*text_prefiller, is_loaded()).WillByDefault(Return(true));

  std::unique_ptr<executorch::llm::Stats> stats =
      std::make_unique<executorch::llm::Stats>();
  auto text_token_generator = createTextTokenGenerator(
      tokenizer.get(), text_decoder_runner.get(), stats.get());

  auto module = std::make_unique<MockModule>();
  auto io_manager =
      std::make_unique<executorch::extension::llm::IOManager>(*module);
  TextLLMRunner runner(
      createDefaultMetadata(), // kMaxContextLen = 128
      std::unique_ptr<::tokenizers::Tokenizer>(tokenizer.release()),
      std::move(module),
      std::move(text_decoder_runner),
      std::unique_ptr<::executorch::extension::llm::TextPrefiller>(
          text_prefiller.release()),
      std::move(io_manager),
      std::move(text_token_generator),
      std::move(stats));

  runner.load();

  // First turn: advance pos_ to 7 (3 prompt + 4 generated)
  GenerationConfig config1;
  config1.max_new_tokens = 5; // prefill generates 1, loop generates 4
  config1.echo = false;
  Error err1 = runner.generate("first turn", config1);
  EXPECT_EQ(err1, Error::Ok);

  // Second turn with seq_len=20: pos_ is now 7, prompt adds 3 more → pos_=10
  // Correct max_new_tokens = min(20, 128) - 10 = 10
  // Bug would give: min(20, 128-7) - 3 = 17
  GenerationConfig config2;
  config2.seq_len = 20;
  config2.echo = false;

  CallbackCounter counter;
  Error err2 = runner.generate(
      "second turn", config2, [&counter](const std::string& token) {
        counter.callback(token);
      });

  EXPECT_EQ(err2, Error::Ok);
  // With correct pos_ accounting: min(20, 128) - 10 = 10 new tokens
  EXPECT_EQ(counter.getCount(), 10);
}

// Verify that a LogitProcessor injected into TextTokenGenerator actually
// affects token selection. Without the processor, greedy argmax of
// {0.1, 0.2, 0.3, 0.4} picks token 3. Masking token 3 should pick token 2.
TEST_F(RunnerTest, TextTokenGeneratorWithProcessorMasksToken) {
  auto tokenizer = createMockTokenizer();
  auto text_decoder_runner = createMockTextDecoderRunner();
  Stats stats;
  auto generator = createTextTokenGenerator(
      tokenizer.get(), text_decoder_runner.get(), &stats);

  generator->add_logit_processor(
      std::make_shared<MaskTokenProcessor>(/*banned_token=*/3));

  std::vector<uint64_t> generated_tokens;
  ON_CALL(*tokenizer, decode)
      .WillByDefault(
          [&](uint64_t,
              uint64_t cur,
              bool) -> ::tokenizers::Result<std::string> {
            generated_tokens.push_back(cur);
            return ::tokenizers::Result<std::string>(std::string("token"));
          });

  std::vector<uint64_t> tokens = {1, 2, 3};
  auto result =
      generator->generate(tokens, 3, 3, 0.0f, [](const std::string&) {});

  EXPECT_TRUE(result.ok());
  const std::vector<uint64_t> expected(3, 2);
  EXPECT_EQ(generated_tokens, expected);
}

// Multiple processors in chain should all take effect.
TEST_F(RunnerTest, TextTokenGeneratorProcessorChainMasksMultipleTokens) {
  auto tokenizer = createMockTokenizer();
  auto text_decoder_runner = createMockTextDecoderRunner();
  Stats stats;
  auto generator = createTextTokenGenerator(
      tokenizer.get(), text_decoder_runner.get(), &stats);

  generator->add_logit_processor(
      std::make_shared<MaskTokenProcessor>(/*banned_token=*/3));
  generator->add_logit_processor(
      std::make_shared<MaskTokenProcessor>(/*banned_token=*/2));

  std::vector<uint64_t> generated_tokens;
  ON_CALL(*tokenizer, decode)
      .WillByDefault(
          [&](uint64_t,
              uint64_t cur,
              bool) -> ::tokenizers::Result<std::string> {
            generated_tokens.push_back(cur);
            return ::tokenizers::Result<std::string>(std::string("token"));
          });

  std::vector<uint64_t> tokens = {1, 2, 3};
  auto result =
      generator->generate(tokens, 3, 3, 0.0f, [](const std::string&) {});

  EXPECT_TRUE(result.ok());
  const std::vector<uint64_t> expected(3, 1);
  EXPECT_EQ(generated_tokens, expected);
}

// Without any processors, greedy argmax picks token 3 (zero-overhead path).
TEST_F(RunnerTest, TextTokenGeneratorWithoutProcessorPicksArgmax) {
  auto tokenizer = createMockTokenizer();
  auto text_decoder_runner = createMockTextDecoderRunner();
  Stats stats;
  auto generator = createTextTokenGenerator(
      tokenizer.get(), text_decoder_runner.get(), &stats);

  std::vector<uint64_t> generated_tokens;
  ON_CALL(*tokenizer, decode)
      .WillByDefault(
          [&](uint64_t,
              uint64_t cur,
              bool) -> ::tokenizers::Result<std::string> {
            generated_tokens.push_back(cur);
            return ::tokenizers::Result<std::string>(std::string("token"));
          });

  std::vector<uint64_t> tokens = {1, 2, 3};
  auto result =
      generator->generate(tokens, 3, 3, 0.0f, [](const std::string&) {});

  EXPECT_TRUE(result.ok());
  const std::vector<uint64_t> expected(3, 3);
  EXPECT_EQ(generated_tokens, expected);
}

} // namespace
