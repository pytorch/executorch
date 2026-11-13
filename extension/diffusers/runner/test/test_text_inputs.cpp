/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/diffusers/runner/text_inputs.h>

#include <gtest/gtest.h>
#include <pytorch/tokenizers/sentencepiece.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

using executorch::extension::diffusers::PromptTokenizer;
namespace fs = std::filesystem;

namespace {

class TextInputsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    directory_ = fs::temp_directory_path() /
        ("diffusers_text_inputs_" +
         std::string(
             ::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::remove_all(directory_);
    fs::create_directories(directory_);
  }

  void TearDown() override {
    fs::remove_all(directory_);
  }

  void write(const std::string& name, const std::string& contents) {
    std::ofstream(directory_ / name) << contents;
  }

  // test/test_tokenizer.json has the words a, b and c, wrapped as <s> ... </s>
  // by its post-processor.
  void write_template_tokenizer(const std::string& config) {
    fs::copy_file(
        std::getenv("DIFFUSERS_TOKENIZER_PATH"),
        directory_ / "tokenizer.json",
        fs::copy_options::overwrite_existing);
    write("tokenizer_config.json", config);
  }

  fs::path directory_;
};

} // namespace

TEST_F(TextInputsTest, PadsToMaxLength) {
  write_template_tokenizer(
      R"({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})");
  auto tokenizer = PromptTokenizer::load(directory_.string(), 8);
  ASSERT_TRUE(tokenizer.ok());
  auto inputs = tokenizer.get()->encode("a b c");
  ASSERT_TRUE(inputs.ok());
  EXPECT_EQ(inputs->input_ids, (std::vector<int64_t>{1, 3, 4, 5, 2, 0, 0, 0}));
  EXPECT_EQ(
      inputs->attention_mask, (std::vector<int64_t>{1, 1, 1, 1, 1, 0, 0, 0}));
}

TEST_F(TextInputsTest, TruncatesContentAndKeepsSpecialTokens) {
  write_template_tokenizer(
      R"({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})");
  auto tokenizer = PromptTokenizer::load(directory_.string(), 5);
  ASSERT_TRUE(tokenizer.ok());
  auto inputs = tokenizer.get()->encode("a b c a b c");
  ASSERT_TRUE(inputs.ok());
  EXPECT_EQ(inputs->input_ids, (std::vector<int64_t>{1, 3, 4, 5, 2}));
  EXPECT_EQ(inputs->attention_mask, (std::vector<int64_t>{1, 1, 1, 1, 1}));
}

TEST_F(TextInputsTest, PadsOnTheConfiguredSide) {
  write_template_tokenizer(
      R"({"bos_token": "<s>", "eos_token": "</s>",
          "pad_token": {"content": "<pad>"}, "padding_side": "left"})");
  auto tokenizer = PromptTokenizer::load(directory_.string(), 7);
  ASSERT_TRUE(tokenizer.ok());
  auto inputs = tokenizer.get()->encode("a b");
  ASSERT_TRUE(inputs.ok());
  EXPECT_EQ(inputs->input_ids, (std::vector<int64_t>{0, 0, 0, 1, 3, 4, 2}));
  EXPECT_EQ(
      inputs->attention_mask, (std::vector<int64_t>{0, 0, 0, 1, 1, 1, 1}));
}

TEST_F(TextInputsTest, RejectsUnsupportedSetups) {
  EXPECT_FALSE(PromptTokenizer::load(directory_.string(), 8).ok());
  write_template_tokenizer(R"({"bos_token": "<s>", "eos_token": "</s>"})");
  EXPECT_FALSE(PromptTokenizer::load(directory_.string(), 8).ok());
  write_template_tokenizer(
      R"({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>",
          "model_max_length": 4})");
  EXPECT_FALSE(PromptTokenizer::load(directory_.string(), 8).ok());
}

TEST_F(TextInputsTest, SentencePieceAppendsEndToken) {
  const std::string model = std::getenv("DIFFUSERS_SENTENCEPIECE_MODEL");
  fs::copy_file(model, directory_ / "spiece.model");
  write("tokenizer_config.json", R"({"pad_token": "<unk>"})");

  tokenizers::SPTokenizer reference;
  ASSERT_EQ(reference.load(model), tokenizers::Error::Ok);
  const auto content = reference.encode("hello world", 0, 0).get();
  const auto end = static_cast<int64_t>(reference.eos_tok());
  const auto pad = static_cast<int64_t>(reference.piece_to_id("<unk>").get());

  const auto length = static_cast<int64_t>(content.size()) + 3;
  auto tokenizer = PromptTokenizer::load(directory_.string(), length);
  ASSERT_TRUE(tokenizer.ok());
  auto inputs = tokenizer.get()->encode("hello world");
  ASSERT_TRUE(inputs.ok());
  std::vector<int64_t> expected(content.begin(), content.end());
  expected.insert(expected.end(), {end, pad, pad});
  EXPECT_EQ(inputs->input_ids, expected);

  auto short_tokenizer = PromptTokenizer::load(
      directory_.string(), static_cast<int64_t>(content.size()));
  ASSERT_TRUE(short_tokenizer.ok());
  auto truncated = short_tokenizer.get()->encode("hello world");
  ASSERT_TRUE(truncated.ok());
  std::vector<int64_t> expected_truncated(content.begin(), content.end() - 1);
  expected_truncated.push_back(end);
  EXPECT_EQ(truncated->input_ids, expected_truncated);
}
