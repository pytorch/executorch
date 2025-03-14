/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <gtest/gtest.h>
#include <pytorch/tokenizers/sentencepiece.h>

namespace tokenizers {

namespace {
static inline std::string _get_resource_path(const std::string& name) {
  return std::getenv("RESOURCES_PATH") + std::string("/") + name;
}
} // namespace

TEST(SPTokenizerTest, TestEncodeWithoutLoad) {
  SPTokenizer tokenizer;
  std::string text = "Hello world!";
  auto result = tokenizer.encode(text, /*bos*/ 0, /*eos*/ 1);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(SPTokenizerTest, TestDecodeWithoutLoad) {
  SPTokenizer tokenizer;
  auto result = tokenizer.decode(0, 0);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(SPTokenizerTest, TestLoad) {
  SPTokenizer tokenizer;
  auto path = _get_resource_path("test_sentencepiece.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
}

TEST(SPTokenizerTest, TestLoadInvalidPath) {
  SPTokenizer tokenizer;
  auto error = tokenizer.load("invalid_path");
  EXPECT_EQ(error, Error::LoadFailure);
}

TEST(SPTokenizerTest, TestEncode) {
  SPTokenizer tokenizer;
  auto path = _get_resource_path("test_sentencepiece.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
  std::string text = "Hello world!";
  auto result = tokenizer.encode(text, /*bos*/ 1, /*eos*/ 0);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.get().size(), 4);
  EXPECT_EQ(result.get()[0], 1);
  EXPECT_EQ(result.get()[1], 15043);
  EXPECT_EQ(result.get()[2], 3186);
  EXPECT_EQ(result.get()[3], 29991);
}

TEST(SPTokenizerTest, TestDecode) {
  SPTokenizer tokenizer;
  auto path = _get_resource_path("test_sentencepiece.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
  std::vector<uint64_t> tokens = {1, 15043, 3186, 29991};
  std::vector<std::string> expected = {"", "Hello", " world", "!"};
  for (auto i = 0; i < 3; ++i) {
    auto result = tokenizer.decode(tokens[i], tokens[i + 1]);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(result.get(), expected[i + 1]);
  }
}

} // namespace tokenizers
