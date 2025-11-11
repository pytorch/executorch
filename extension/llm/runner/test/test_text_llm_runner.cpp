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
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace ::testing;
using executorch::extension::llm::GenerationConfig;
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
  MOCK_METHOD(::tokenizers::Error, load, (const std::string&), ());
  MOCK_METHOD(bool, is_loaded, (), (const));
  MOCK_METHOD(
      ::tokenizers::Result<std::vector<uint64_t>>,
      encode,
      (const std::string&, int8_t, int8_t),
      (const));
  MOCK_METHOD(
      ::tokenizers::Result<std::string>,
      decode,
      (uint64_t, uint64_t),
      (const));
  MOCK_METHOD(uint64_t, bos_tok, (), (const));
  MOCK_METHOD(uint64_t, eos_tok, (), (const));
  MOCK_METHOD(uint64_t, vocab_size, (), (const));
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

    ON_CALL(*tokenizer, decode).WillByDefault([](uint64_t, uint64_t) {
      return ::tokenizers::Result<std::string>("token");
    });

    ON_CALL(*tokenizer, bos_tok()).WillByDefault(Return(1));
    ON_CALL(*tokenizer, eos_tok()).WillByDefault(Return(2));
    ON_CALL(*tokenizer, vocab_size()).WillByDefault(Return(100));

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

} // namespace
