/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/module/module.h>

#include <cstdlib>
#include <memory>

#include <gtest/gtest.h>

namespace {

using ::executorch::extension::Module;
using ::executorch::extension::llm::check_vocab_size;
using ::executorch::extension::llm::LogitsToKeepMode;
using ::executorch::extension::llm::read_activation_dtype;
using ::executorch::extension::llm::read_logits_to_keep_mode;
using ::executorch::extension::llm::read_max_context_length;
using ::executorch::extension::llm::read_max_seq_len;
using ::executorch::extension::llm::read_vocab_size;
using ::executorch::runtime::Error;

std::unique_ptr<Module> load_fixture(const char* environment_variable) {
  const char* path = std::getenv(environment_variable);
  EXPECT_NE(path, nullptr);
  auto module = std::make_unique<Module>(path);
  EXPECT_EQ(module->load(), Error::Ok);
  return module;
}

struct ModeCase {
  const char* environment_variable;
  LogitsToKeepMode expected_mode;
  ::executorch::aten::ScalarType expected_dtype;
};

class ModelMetadataTest : public ::testing::TestWithParam<ModeCase> {};

TEST_P(ModelMetadataTest, ReadsPythonExportedConstants) {
  auto module = load_fixture(GetParam().environment_variable);

  const auto max_context_length = read_max_context_length(*module);
  ASSERT_TRUE(max_context_length.ok());
  EXPECT_EQ(*max_context_length, 4096);

  const auto max_seq_len = read_max_seq_len(*module);
  ASSERT_TRUE(max_seq_len.ok());
  EXPECT_EQ(*max_seq_len, 512);

  const auto vocab_size = read_vocab_size(*module);
  ASSERT_TRUE(vocab_size.ok());
  EXPECT_EQ(*vocab_size, 128256);

  const auto activation_dtype = read_activation_dtype(*module);
  ASSERT_TRUE(activation_dtype.ok());
  EXPECT_EQ(*activation_dtype, GetParam().expected_dtype);

  const auto logits_to_keep_mode = read_logits_to_keep_mode(*module);
  ASSERT_TRUE(logits_to_keep_mode.ok());
  EXPECT_EQ(*logits_to_keep_mode, GetParam().expected_mode);
}

TEST(ModelMetadataTest, RejectsNonPositiveSizes) {
  {
    auto module = load_fixture("ET_MODEL_METADATA_INVALID_CONTEXT_PATH");
    const auto value = read_max_context_length(*module);
    ASSERT_FALSE(value.ok());
    EXPECT_EQ(value.error(), Error::InvalidProgram);
  }
  {
    auto module = load_fixture("ET_MODEL_METADATA_INVALID_PREFILL_PATH");
    const auto value = read_max_seq_len(*module);
    ASSERT_FALSE(value.ok());
    EXPECT_EQ(value.error(), Error::InvalidProgram);
  }
  {
    auto module = load_fixture("ET_MODEL_METADATA_INVALID_VOCAB_PATH");
    const auto value = read_vocab_size(*module);
    ASSERT_FALSE(value.ok());
    EXPECT_EQ(value.error(), Error::InvalidProgram);
  }
}

TEST(ModelMetadataTest, ChecksVocabAgainstForwardOutput) {
  auto matched = check_vocab_size(128256, 128256);
  ASSERT_TRUE(matched.ok());
  EXPECT_EQ(*matched, 128256);

  auto mismatched = check_vocab_size(128256, 128000);
  ASSERT_FALSE(mismatched.ok());
  EXPECT_EQ(mismatched.error(), Error::InvalidProgram);
}

TEST(ModelMetadataTest, RejectsMissingRequiredFields) {
  auto module = load_fixture("ET_MODEL_METADATA_MISSING_PATH");
  EXPECT_FALSE(read_max_context_length(*module).ok());
  EXPECT_FALSE(read_max_seq_len(*module).ok());
  EXPECT_FALSE(read_vocab_size(*module).ok());
  EXPECT_FALSE(read_activation_dtype(*module).ok());
  EXPECT_FALSE(read_logits_to_keep_mode(*module).ok());
}

INSTANTIATE_TEST_SUITE_P(
    RoundTrip,
    ModelMetadataTest,
    ::testing::Values(
        ModeCase{
            "ET_MODEL_METADATA_FULL_PATH",
            LogitsToKeepMode::Full,
            ::executorch::aten::ScalarType::Float},
        ModeCase{
            "ET_MODEL_METADATA_LAST_PATH",
            LogitsToKeepMode::Last,
            ::executorch::aten::ScalarType::Half},
        ModeCase{
            "ET_MODEL_METADATA_SELECTED_PATH",
            LogitsToKeepMode::Selected,
            ::executorch::aten::ScalarType::BFloat16}));

} // namespace
