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
#include <string>

#include <gtest/gtest.h>

namespace {

using ::executorch::extension::Module;
using ::executorch::extension::llm::LogitsToKeepMode;
using ::executorch::extension::llm::read_model_metadata;
using ::executorch::runtime::Error;

struct ModeCase {
  const char* environment_variable;
  LogitsToKeepMode expected_mode;
  ::executorch::aten::ScalarType expected_dtype;
};

class ModelMetadataTest : public ::testing::TestWithParam<ModeCase> {};

TEST_P(ModelMetadataTest, ReadsPythonExportedConstants) {
  const char* path = std::getenv(GetParam().environment_variable);
  ASSERT_NE(path, nullptr);
  Module module(path);
  ASSERT_EQ(module.load(), Error::Ok);

  const auto metadata = read_model_metadata(module);
  ASSERT_TRUE(metadata.ok());
  ASSERT_TRUE(metadata->max_seq_len.has_value());
  EXPECT_EQ(metadata->max_context_length, 4096);
  EXPECT_EQ(*metadata->max_seq_len, 512);
  EXPECT_EQ(metadata->vocab_size, 128256);
  EXPECT_EQ(metadata->activation_dtype, GetParam().expected_dtype);
  EXPECT_EQ(metadata->logits_to_keep_mode, GetParam().expected_mode);
}

TEST(ModelMetadataTest, RejectsNonPositiveSizes) {
  for (const char* environment_variable : {
           "ET_MODEL_METADATA_INVALID_CONTEXT_PATH",
           "ET_MODEL_METADATA_INVALID_PREFILL_PATH",
           "ET_MODEL_METADATA_INVALID_VOCAB_PATH",
       }) {
    const char* path = std::getenv(environment_variable);
    ASSERT_NE(path, nullptr);
    Module module(path);
    ASSERT_EQ(module.load(), Error::Ok);

    const auto metadata = read_model_metadata(module);
    ASSERT_FALSE(metadata.ok());
    EXPECT_EQ(metadata.error(), Error::InvalidProgram);
  }
}

TEST(ModelMetadataTest, ResolvesAndValidatesVocabSize) {
  ::executorch::extension::llm::ModelMetadata metadata;
  metadata.vocab_size = 128256;
  auto matched = ::executorch::extension::llm::resolve_vocab_size(metadata, 128256);
  ASSERT_TRUE(matched.ok());
  EXPECT_EQ(*matched, 128256);

  auto mismatched =
      ::executorch::extension::llm::resolve_vocab_size(metadata, 128000);
  ASSERT_FALSE(mismatched.ok());
  EXPECT_EQ(mismatched.error(), Error::InvalidProgram);
}

TEST(ModelMetadataTest, RejectsMissingRequiredFields) {
  const char* path = std::getenv("ET_MODEL_METADATA_MISSING_PATH");
  ASSERT_NE(path, nullptr);
  Module module(path);
  ASSERT_EQ(module.load(), Error::Ok);

  const auto metadata = read_model_metadata(module);
  ASSERT_FALSE(metadata.ok());
  EXPECT_EQ(metadata.error(), Error::InvalidProgram);
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
