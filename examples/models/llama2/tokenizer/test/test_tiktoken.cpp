/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/llama2/tokenizer/llama_tiktoken.h>

#include <vector>

#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;

using ::example::Version;
using ::executorch::extension::llm::Tokenizer;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

class MultimodalTiktokenV5ExtensionTest : public Test {
 public:
  void SetUp() override {
    executorch::runtime::runtime_init();
    tokenizer_ = get_tiktoken_for_llama(Version::Multimodal);
    modelPath_ = std::getenv("RESOURCES_PATH") +
        std::string("/test_tiktoken_tokenizer.model");
  }

  std::unique_ptr<Tokenizer> tokenizer_;
  std::string modelPath_;
};

TEST_F(MultimodalTiktokenV5ExtensionTest, TokenizerVocabSizeIsExpected) {
  Error res = tokenizer_->load(modelPath_.c_str());
  EXPECT_EQ(res, Error::Ok);
  EXPECT_EQ(tokenizer_->vocab_size(), 128256);
  EXPECT_EQ(tokenizer_->bos_tok(), 128000);
  EXPECT_EQ(tokenizer_->eos_tok(), 128001);
}

TEST_F(MultimodalTiktokenV5ExtensionTest, TokenizerEncodeCorrectly) {
  Error res = tokenizer_->load(modelPath_.c_str());
  EXPECT_EQ(res, Error::Ok);
  Result<std::vector<uint64_t>> out = tokenizer_->encode(
      "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>What do you think is going on in this snapshot?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAmidst a scenic garden backdrop, a man dressed in a suit with a distinct button on its lower portion stands prominently.<|eom_id|>",
      0,
      0);
  EXPECT_EQ(out.error(), Error::Ok);
  EXPECT_EQ(out.get().size(), 48);
  std::vector<uint64_t> expected_out = {
      128000, 128006, 882,    128007, 271,    128010, 3923,  656,
      499,    1781,   374,    2133,   389,    304,    420,   16694,
      30,     128009, 128006, 78191,  128007, 271,    6219,  307,
      267,    264,    62081,  13863,  39577,  11,     264,   893,
      26435,  304,    264,    7937,   449,    264,    12742, 3215,
      389,    1202,   4827,   13651,  13656,  74088,  13,    128008};
  for (size_t i = 0; i < expected_out.size(); ++i) {
    EXPECT_EQ(expected_out[i], out.get()[i]);
  }
}

TEST_F(MultimodalTiktokenV5ExtensionTest, TokenizerDecodeCorrectly) {
  Error res = tokenizer_->load(modelPath_.c_str());
  EXPECT_EQ(res, Error::Ok);
  std::vector<std::string> expected = {
      "<|begin_of_text|>",
      "<|start_header_id|>",
      "user",
      "<|end_header_id|>",
      "<|image|>",
      "<|image|>",
      "hello",
      "<|image|>",
      "<|eom_id|>"};
  std::vector<uint64_t> tokens = {
      128000, 128006, 882, 128007, 128010, 128010, 15339, 128010, 128008};
  for (size_t i = 0; i < tokens.size(); i++) {
    Result<std::string> out = tokenizer_->decode(0, tokens[i]);
    EXPECT_EQ(out.error(), Error::Ok);
    EXPECT_EQ(out.get(), expected[i]);
  }
}
