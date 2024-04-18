/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/llama2/tokenizer/tiktoken.h>
#include <executorch/examples/models/llama2/tokenizer/tokenizer.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace ::testing;

namespace torch {
namespace executor {

class TiktokenExtensionTest : public Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
    tokenizer_ = std::make_unique<Tiktoken>(128256, 128000, 128001);
    modelPath_ =
        std::getenv("RESOURCES_PATH") + std::string("/fb/tokenizer.model");
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
  // test.bin has vocab size 0 but the tokenizer respects the vocab size being
  // passed in and add placeholder tokens.
  EXPECT_EQ(tokenizer_->vocab_size(), 128256);
  EXPECT_EQ(tokenizer_->bos_tok(), 128000);
  EXPECT_EQ(tokenizer_->eos_tok(), 128001);
}

TEST_F(TiktokenExtensionTest, TokenizerEncodeCorrectly) {
  Error res = tokenizer_->load(modelPath_.c_str());
  EXPECT_EQ(res, Error::Ok);
  // test.bin has vocab size 0 but the tokenizer respects the vocab size being
  // passed in and add placeholder tokens.
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
  // test.bin has vocab size 0 but the tokenizer respects the vocab size being
  // passed in and add placeholder tokens.
  std::vector<std::string> expected = {"<|begin_of_text|>", "hello", " world"};
  std::vector<uint64_t> tokens = {128000, 15339, 1917};
  for (size_t i = 0; i < tokens.size(); i++) {
    Result<std::string> out = tokenizer_->decode(0, tokens[i]);
    EXPECT_EQ(out.error(), Error::Ok);
    EXPECT_EQ(out.get(), expected[i]);
  }
}

} // namespace executor
} // namespace torch
