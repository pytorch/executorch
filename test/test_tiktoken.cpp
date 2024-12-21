/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "tiktoken.h"
#include "gtest/gtest.h"

namespace tokenizers {

TEST(TiktokenTest, TestEncodeWithoutLoad) {
  Tiktoken tokenizer;
  std::string text = "Hello world!";
  auto result = tokenizer.encode(text, /*bos*/ 0, /*eos*/ 1);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(TiktokenTest, TestDecodeWithoutLoad) {
  Tiktoken tokenizer;
  auto result = tokenizer.decode(0, 0);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(TiktokenTest, TestLoad) {
  Tiktoken tokenizer;
  auto path = RESOURCES_PATH + std::string("/test_tiktoken.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
}

TEST(TiktokenTest, TestLoadInvalidPath) {
  Tiktoken tokenizer;
  auto error = tokenizer.load("invalid_path");
  EXPECT_EQ(error, Error::LoadFailure);
}

TEST(TiktokenTest, TestEncode) {
  Tiktoken tokenizer;
  auto path = RESOURCES_PATH + std::string("/test_tiktoken.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
  std::string text = "Hello world!";
  auto result = tokenizer.encode(text, /*bos*/ 1, /*eos*/ 0);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.get().size(), 4);
  EXPECT_EQ(result.get()[0], 128000);
  EXPECT_EQ(result.get()[1], 9906);
  EXPECT_EQ(result.get()[2], 1917);
  EXPECT_EQ(result.get()[3], 0);
}

TEST(TiktokenTest, TestDecode) {
  Tiktoken tokenizer;
  auto path = RESOURCES_PATH + std::string("/test_tiktoken.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
  std::vector<uint64_t> tokens = {128000, 9906, 1917, 0};
  std::vector<std::string> expected = {"", "Hello", " world", "!"};
  for (auto i = 0; i < 3; ++i) {
    auto result = tokenizer.decode(tokens[i], tokens[i + 1]);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(result.get(), expected[i + 1]);
  }
}

} // namespace tokenizers
