/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/llama2/tokenizer/bpe_tokenizer.h>
#include <executorch/examples/models/llama2/tokenizer/tokenizer.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace ::testing;

namespace torch {
namespace executor {

class TokenizerExtensionTest : public Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
    tokenizer_ = std::make_unique<BPETokenizer>(32000, 1, 2);
    modelPath_ = std::getenv("RESOURCES_PATH") + std::string("/test.bin");
  }

  std::unique_ptr<Tokenizer> tokenizer_;
  std::string modelPath_;
};

TEST_F(TokenizerExtensionTest, EncodeWithoutLoadFails) {
  Result<std::vector<uint64_t>> res = tokenizer_->encode("hello world", 0, 0);
  EXPECT_EQ(res.error(), Error::NotSupported);
}

TEST_F(TokenizerExtensionTest, DecodeWithoutLoadFails) {
  auto result = tokenizer_->decode(0, 0);
  EXPECT_EQ(result.error(), Error::NotSupported);
}

TEST_F(TokenizerExtensionTest, TokenizerVocabSizeIsExpected) {
  Error res = tokenizer_->load(modelPath_.c_str());
  EXPECT_EQ(res, Error::Ok);
  // test.bin has vocab size 0 but the tokenizer respects the vocab size being
  // passed in and add placeholder tokens.
  EXPECT_EQ(tokenizer_->vocab_size(), 32000);
  EXPECT_EQ(tokenizer_->bos_tok(), 1);
  EXPECT_EQ(tokenizer_->eos_tok(), 2);
}

} // namespace executor
} // namespace torch
