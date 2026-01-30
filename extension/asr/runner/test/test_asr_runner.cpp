/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/asr/runner/runner.h>
#include <gtest/gtest.h>

#include <functional>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using executorch::extension::asr::AsrTranscribeConfig;
using executorch::extension::asr::TokenCallback;

// Test AsrTranscribeConfig default values
TEST(AsrTranscribeConfigTest, DefaultValues) {
  AsrTranscribeConfig config;

  EXPECT_EQ(config.max_new_tokens, 128);
  EXPECT_TRUE(config.eos_token_ids.empty());
  EXPECT_FLOAT_EQ(config.temperature, 0.0f);
  EXPECT_EQ(config.decoder_start_token_id, 0);
}

// Test AsrTranscribeConfig with custom values
TEST(AsrTranscribeConfigTest, CustomValues) {
  AsrTranscribeConfig config;
  config.max_new_tokens = 256;
  config.eos_token_ids = {50257, 50256};
  config.temperature = 0.5f;
  config.decoder_start_token_id = 50258;

  EXPECT_EQ(config.max_new_tokens, 256);
  EXPECT_EQ(config.eos_token_ids.size(), 2);
  EXPECT_TRUE(config.eos_token_ids.count(50257) > 0);
  EXPECT_TRUE(config.eos_token_ids.count(50256) > 0);
  EXPECT_FLOAT_EQ(config.temperature, 0.5f);
  EXPECT_EQ(config.decoder_start_token_id, 50258);
}

// Test TokenCallback type alias is correctly defined
TEST(TokenCallbackTest, TypeIsCallable) {
  // Verify TokenCallback is a std::function<void(const std::string&)>
  std::string captured_token;
  TokenCallback callback = [&captured_token](const std::string& token) {
    captured_token = token;
  };

  callback("hello");
  EXPECT_EQ(captured_token, "hello");

  callback("world");
  EXPECT_EQ(captured_token, "world");
}

// Test TokenCallback works with optional
TEST(TokenCallbackTest, WorksWithOptional) {
  std::optional<TokenCallback> optional_callback;
  EXPECT_FALSE(optional_callback.has_value());

  std::vector<std::string> captured_tokens;
  optional_callback = [&captured_tokens](const std::string& token) {
    captured_tokens.push_back(token);
  };
  EXPECT_TRUE(optional_callback.has_value());

  // Call through optional
  (*optional_callback)("token1");
  (*optional_callback)("token2");

  EXPECT_EQ(captured_tokens.size(), 2);
  EXPECT_EQ(captured_tokens[0], "token1");
  EXPECT_EQ(captured_tokens[1], "token2");
}

// Test TokenCallback with nullopt
TEST(TokenCallbackTest, NulloptIsValid) {
  std::optional<TokenCallback> optional_callback = std::nullopt;
  EXPECT_FALSE(optional_callback.has_value());

  // This is how the runner checks for callback presence
  if (optional_callback.has_value()) {
    FAIL() << "Nullopt should not have value";
  }
}

// Test lambda conversion to optional<TokenCallback>
TEST(TokenCallbackTest, LambdaConvertsToOptional) {
  std::string result;

  // This tests the implicit conversion that happens in the transcribe call
  auto lambda = [&result](const std::string& token) { result = token; };

  // Lambda should be convertible to TokenCallback
  TokenCallback callback = lambda;
  callback("test");
  EXPECT_EQ(result, "test");

  // And TokenCallback should be convertible to optional<TokenCallback>
  std::optional<TokenCallback> optional_callback = callback;
  EXPECT_TRUE(optional_callback.has_value());
  (*optional_callback)("test2");
  EXPECT_EQ(result, "test2");
}

} // namespace
