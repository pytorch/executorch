/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <gtest/gtest.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

namespace tokenizers {

namespace {
static inline std::string _get_resource_path(const std::string& name) {
  return std::getenv("RESOURCES_PATH") + std::string("/") + name;
}
} // namespace

TEST(HFTokenizerTest, TestEncodeWithoutLoad) {
  HFTokenizer tokenizer;
  std::string text = "Hello world!";
  auto result = tokenizer.encode(text, /*bos*/ 0, /*eos*/ 1);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(HFTokenizerTest, TestDecodeWithoutLoad) {
  HFTokenizer tokenizer;
  auto result = tokenizer.decode(0, 0);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(HFTokenizerTest, TestLoad) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("test_hf_tokenizer.json");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
}

TEST(HFTokenizerTest, TestLoadInvalidPath) {
  HFTokenizer tokenizer;
  auto error = tokenizer.load("invalid_path");
  EXPECT_EQ(error, Error::LoadFailure);
}

TEST(HFTokenizerTest, TestEncode) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("test_hf_tokenizer.json");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
  std::string text = "Hello world!";
  auto result = tokenizer.encode(text, /*bos*/ 1, /*eos*/ 0);
  EXPECT_TRUE(result.ok());
  // Based on our test tokenizer vocab:
  // "Hello world!" should tokenize to something like [1, 8, 9] or [1, 4, 5, 6,
  // 7] depending on how the BPE merges work
  EXPECT_GT(result.get().size(), 0);
  EXPECT_EQ(result.get()[0], 0); // BOS token (ID 0 in our test tokenizer)
}

TEST(HFTokenizerTest, TestDecode) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("test_hf_tokenizer.json");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
  // Test with tokens from our vocab: <s>, ▁Hello, ▁world!
  std::vector<uint64_t> tokens = {1, 8, 9}; // <s>, ▁Hello, ▁world!
  for (auto i = 0; i < static_cast<int>(tokens.size()) - 1; ++i) {
    auto result = tokenizer.decode(tokens[i], tokens[i + 1]);
    EXPECT_TRUE(result.ok());
    // The decoded strings should not be empty
    EXPECT_FALSE(result.get().empty());
  }
}

} // namespace tokenizers
