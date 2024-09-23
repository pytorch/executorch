/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/tokenizer/tiktoken.h>
#include <executorch/runtime/platform/runtime.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>

using namespace ::testing;
using ::executorch::extension::llm::Tiktoken;
using ::executorch::extension::llm::Tokenizer;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace {
// Test case based on Llama 2
static constexpr int32_t kSpecialTokensSize = 256;
static constexpr size_t kBOSTokenIndex = 0;
static constexpr size_t kEOSTokenIndex = 1;
static inline std::unique_ptr<std::vector<std::string>> _get_special_tokens() {
  auto special_tokens =
      std::make_unique<std::vector<std::string>>(std::vector<std::string>{
          "<|begin_of_text|>",
          "<|end_of_text|>",
          "<|reserved_special_token_0|>",
          "<|reserved_special_token_1|>",
          "<|reserved_special_token_2|>",
          "<|reserved_special_token_3|>",
          "<|start_header_id|>",
          "<|end_header_id|>",
          "<|reserved_special_token_4|>",
          "<|eot_id|>"});

  // pad the rest of the special tokens with reserved tokens
  ssize_t reserved_special_token_num = 5;
  while (special_tokens->size() < kSpecialTokensSize) {
    special_tokens->emplace_back(
        "<|reserved_special_token_" +
        std::to_string(reserved_special_token_num++) + "|>");
  }
  return special_tokens;
}
} // namespace

class TiktokenExtensionTest : public Test {
 public:
  void SetUp() override {
    executorch::runtime::runtime_init();
    tokenizer_ = std::make_unique<Tiktoken>(
        _get_special_tokens(), kBOSTokenIndex, kEOSTokenIndex);
    modelPath_ = std::getenv("RESOURCES_PATH") +
        std::string("/test_tiktoken_tokenizer.model");
  }

  std::unique_ptr<Tokenizer> tokenizer_;
  std::string modelPath_;
};

TEST_F(TiktokenExtensionTest, EncodeWithoutLoadFails) {
  Result<std::vector<uint64_t>> res = tokenizer_->encode("hello world", 0, 0);
  EXPECT_EQ(res.error(), Error::NotSupported);
}

TEST_F(TiktokenExtensionTest, DecodeWithoutLoadFails) {
  auto result = tokenizer_->decode(0, 0);
  EXPECT_EQ(result.error(), Error::NotSupported);
}

TEST_F(TiktokenExtensionTest, TokenizerVocabSizeIsExpected) {
  Error res = tokenizer_->load(modelPath_.c_str());
  EXPECT_EQ(res, Error::Ok);
  EXPECT_EQ(tokenizer_->vocab_size(), 128256);
  EXPECT_EQ(tokenizer_->bos_tok(), 128000);
  EXPECT_EQ(tokenizer_->eos_tok(), 128001);
}

TEST_F(TiktokenExtensionTest, TokenizerEncodeCorrectly) {
  Error res = tokenizer_->load(modelPath_.c_str());
  EXPECT_EQ(res, Error::Ok);
  Result<std::vector<uint64_t>> out = tokenizer_->encode("hello world", 1, 0);
  EXPECT_EQ(out.error(), Error::Ok);
  EXPECT_EQ(out.get().size(), 3);
  EXPECT_EQ(out.get()[0], 128000);
  EXPECT_EQ(out.get()[1], 15339);
  EXPECT_EQ(out.get()[2], 1917);
}

TEST_F(TiktokenExtensionTest, TokenizerDecodeCorrectly) {
  Error res = tokenizer_->load(modelPath_.c_str());
  EXPECT_EQ(res, Error::Ok);
  std::vector<std::string> expected = {"<|begin_of_text|>", "hello", " world"};
  std::vector<uint64_t> tokens = {128000, 15339, 1917};
  for (size_t i = 0; i < tokens.size(); i++) {
    Result<std::string> out = tokenizer_->decode(0, tokens[i]);
    EXPECT_EQ(out.error(), Error::Ok);
    EXPECT_EQ(out.get(), expected[i]);
  }
}

TEST_F(TiktokenExtensionTest, TokenizerDecodeOutOfRangeFails) {
  Error res = tokenizer_->load(modelPath_.c_str());
  EXPECT_EQ(res, Error::Ok);
  // The vocab size is 128256, addes 256 just so the token is out of vocab
  // range.
  Result<std::string> out = tokenizer_->decode(0, 128256 + 256);
  EXPECT_EQ(out.error(), Error::NotSupported);
}

TEST_F(TiktokenExtensionTest, ConstructionWithInvalidBOSIndex) {
  // gtest death test doesn't work on iOS:
  // https://github.com/google/googletest/issues/2834
#if !GTEST_OS_IOS
  EXPECT_EXIT(
      std::make_unique<Tiktoken>(
          std::make_unique<std::vector<std::string>>(
              std::vector<std::string>{"<|end_of_text|>"}),
          1,
          0),
      ::testing::KilledBySignal(SIGABRT),
      "");
#endif
}

TEST_F(TiktokenExtensionTest, ConstructionWithInvalidEOSIndex) {
  // gtest death test doesn't work on iOS:
  // https://github.com/google/googletest/issues/2834
#if !GTEST_OS_IOS
  EXPECT_EXIT(
      std::make_unique<Tiktoken>(
          std::make_unique<std::vector<std::string>>(
              std::vector<std::string>{"<|begin_of_text|>"}),
          0,
          1),
      ::testing::KilledBySignal(SIGABRT),
      "");
#endif
}

TEST_F(TiktokenExtensionTest, LoadWithInvalidPath) {
  auto invalidModelPath =
      std::getenv("RESOURCES_PATH") + std::string("/nonexistent.model");

  Error res = tokenizer_->load(invalidModelPath.c_str());
  EXPECT_EQ(res, Error::InvalidArgument);
}

TEST_F(TiktokenExtensionTest, LoadTiktokenFileWithInvalidRank) {
  auto invalidModelPath = std::getenv("RESOURCES_PATH") +
      std::string("/test_tiktoken_invalid_rank.model");

  Error res = tokenizer_->load(invalidModelPath.c_str());

  EXPECT_EQ(res, Error::InvalidArgument);
}

TEST_F(TiktokenExtensionTest, LoadTiktokenFileWithInvalidBase64) {
  auto invalidModelPath = std::getenv("RESOURCES_PATH") +
      std::string("/test_tiktoken_invalid_base64.model");

  Error res = tokenizer_->load(invalidModelPath.c_str());

  EXPECT_EQ(res, Error::InvalidArgument);
}

TEST_F(TiktokenExtensionTest, LoadTiktokenFileWithNoSpace) {
  auto invalidModelPath = std::getenv("RESOURCES_PATH") +
      std::string("/test_tiktoken_no_space.model");

  Error res = tokenizer_->load(invalidModelPath.c_str());

  EXPECT_EQ(res, Error::InvalidArgument);
}

TEST_F(TiktokenExtensionTest, LoadTiktokenFileWithBPEFile) {
  auto invalidModelPath =
      std::getenv("RESOURCES_PATH") + std::string("/test_bpe_tokenizer.bin");

  Error res = tokenizer_->load(invalidModelPath.c_str());

  EXPECT_EQ(res, Error::InvalidArgument);
}
