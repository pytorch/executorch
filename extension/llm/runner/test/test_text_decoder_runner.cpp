/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */

#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cstdlib>

using namespace ::testing;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::llm::IOManager;
using executorch::extension::llm::TextDecoderRunner;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;
using executorch::runtime::testing::TensorFactory;

namespace {
class MockModule : public Module {
 public:
  MockModule() : Module("") {}
};

class TextDecoderRunnerTest : public Test {
 protected:
  void SetUp() override {
    mock_module_ = std::make_unique<MockModule>();
    io_manager_ =
        std::make_unique<executorch::extension::llm::IOManager>(*mock_module_);
    runner_ = std::make_unique<TextDecoderRunner>(
        mock_module_.get(), io_manager_.get());
  }

  std::unique_ptr<MockModule> mock_module_;
  std::unique_ptr<TextDecoderRunner> runner_;
  std::unique_ptr<IOManager> io_manager_;
};

// Test logits_to_token() method with Float tensor
TEST_F(TextDecoderRunnerTest, LogitsToTokenFloat) {
  TensorFactory<executorch::aten::ScalarType::Float> tf_float;
  auto logits = tf_float.make({1, 4}, {0.1f, 0.2f, 0.8f, 0.4f});

  // Call logits_to_token with temperature 0 (deterministic)
  int32_t token = runner_->logits_to_token(logits, 0.0f);

  // With temperature 0, should return the argmax (index 2)
  EXPECT_EQ(token, 2);
}

// Test logits_to_token() method with 3D tensor (batch, seq_length, vocab_size)
TEST_F(TextDecoderRunnerTest, LogitsToToken3D) {
  TensorFactory<executorch::aten::ScalarType::Float> tf_float;
  // Shape: [1, 2, 4] - batch=1, seq_length=2, vocab_size=4
  auto logits = tf_float.make(
      {1, 2, 4},
      {
          0.1f,
          0.2f,
          0.3f,
          0.4f, // First sequence position
          0.5f,
          0.6f,
          0.9f,
          0.8f // Second sequence position (last)
      });

  // Call logits_to_token with temperature 0 (deterministic)
  int32_t token = runner_->logits_to_token(logits, 0.0f);

  // Should use the last sequence position and return argmax (index 2)
  EXPECT_EQ(token, 2);
}

// Test logits_to_token() method with Half tensor
TEST_F(TextDecoderRunnerTest, LogitsToTokenHalf) {
  TensorFactory<executorch::aten::ScalarType::Half> tf_half;
  auto logits = tf_half.make({1, 4}, {0.1f, 0.2f, 0.8f, 0.4f});

  // Call logits_to_token with temperature 0 (deterministic)
  int32_t token = runner_->logits_to_token(logits, 0.0f);

  // With temperature 0, should return the argmax (index 2)
  EXPECT_EQ(token, 2);
}

// Test logits_to_token() method with BFloat16 tensor
TEST_F(TextDecoderRunnerTest, LogitsToTokenBFloat16) {
  TensorFactory<executorch::aten::ScalarType::BFloat16> tf_bfloat16;
  auto logits = tf_bfloat16.make({1, 4}, {0.1f, 0.2f, 0.8f, 0.4f});

  // Call logits_to_token with temperature 0 (deterministic)
  int32_t token = runner_->logits_to_token(logits, 0.0f);

  // With temperature 0, should return the argmax (index 2)
  EXPECT_EQ(token, 2);
}

// Test logits_to_token() method with non-zero temperature
TEST_F(TextDecoderRunnerTest, LogitsToTokenWithTemperature) {
  TensorFactory<executorch::aten::ScalarType::Float> tf_float;
  auto logits = tf_float.make({1, 4}, {0.1f, 0.2f, 0.8f, 0.4f});

  // Call logits_to_token with temperature > 0 (stochastic)
  int32_t token = runner_->logits_to_token(logits, 1.0f);

  // With temperature > 0, result should be within valid range
  EXPECT_GE(token, 0);
  EXPECT_LT(token, 4);
}

// Test step() method with all available PTE models
TEST_F(TextDecoderRunnerTest, StepWithAllModels) {
  // List of all environment variables for PTE models
  std::vector<std::pair<std::string, const char*>> env_vars = {
      {"KVCACHE_CACHE_POS", "KVCACHE_CACHE_POS"},
      {"KVCACHE_INPUT_POS", "KVCACHE_INPUT_POS"},
      {"NO_KVCACHE", "NO_KVCACHE"}};

  // Check if any environment variables are set up front
  bool any_env_set = false;
  for (const auto& [model_name, env_var] : env_vars) {
    if (std::getenv(env_var)) {
      any_env_set = true;
      break;
    }
  }

  // Skip test if no environment variables are set
  if (!any_env_set) {
    GTEST_SKIP() << "No PTE model environment variables were set";
  }

  bool any_model_tested = false;

  // Loop through all available models
  for (const auto& [model_name, env_var] : env_vars) {
    const char* model_path = std::getenv(env_var);
    if (!model_path) {
      continue; // Skip if environment variable not set
    }

    SCOPED_TRACE(
        "Testing model: " + model_name + " from " + std::string(model_path));

    // Load the model
    auto module = std::make_unique<Module>(model_path);

    auto load_result = module->load();
    if (load_result != Error::Ok) {
      ADD_FAILURE() << "Failed to load model " << model_name << " from "
                    << model_path << " with error: " << (int)load_result;
      continue;
    }
    auto io_manager =
        std::make_unique<executorch::extension::llm::IOManager>(*module);
    // Create TextDecoderRunner
    TextDecoderRunner runner(module.get(), io_manager.get());
    auto runner_load_result = runner.load();
    ASSERT_EQ(runner_load_result, Error::Ok)
        << "Failed to load runner for " << model_name;

    // Verify method is loaded
    EXPECT_TRUE(runner.is_method_loaded())
        << "Method not loaded for " << model_name;

    // Create input tensor pointer

    TensorFactory<executorch::aten::ScalarType::Long> tf_long;
    auto input_tokens_ =
        tf_long.make({1, 3}, {50, 7, 11}); // Single token input

    auto input_ptr = std::make_shared<executorch::aten::Tensor>(input_tokens_);
    int64_t start_pos = 0;

    // Call step() and verify result is ok
    auto result = runner.step(input_ptr, start_pos);
    ASSERT_TRUE(result.ok()) << "step() failed for " << model_name
                             << " with error: " << (int)result.error();

    // Verify output tensor is valid
    auto output_tensor = result.get();
    EXPECT_GT(output_tensor.numel(), 0)
        << "Output tensor empty for " << model_name;

    // Test logits_to_token works
    int32_t token = runner.logits_to_token(output_tensor, 0.0f);
    EXPECT_GE(token, 0) << "Invalid token for " << model_name;

    any_model_tested = true;
  }

  // This should not happen since we checked environment variables up front
  ASSERT_TRUE(any_model_tested)
      << "No models were tested despite environment variables being set";
}

} // namespace
