/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 * @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
 */

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/text_llm_runner.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/platform/runtime.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>

using namespace ::testing;
using executorch::extension::Module;
using executorch::extension::llm::create_text_llm_runner;
using executorch::runtime::Error;

namespace {
class MockTokenizer : public ::tokenizers::Tokenizer {
 public:
  MockTokenizer() {
    initialized_ = true;
    vocab_size_ = 32;
  }

  MOCK_METHOD(::tokenizers::Error, load, (const std::string&), (override));
  MOCK_METHOD(
      ::tokenizers::Result<std::vector<uint64_t>>,
      encode,
      (const std::string&, int8_t, int8_t),
      (const, override));
  MOCK_METHOD(
      ::tokenizers::Result<std::string>,
      decode,
      (uint64_t, uint64_t, bool),
      (const, override));
  MOCK_METHOD(
      ::tokenizers::Result<std::string>,
      id_to_piece,
      (uint64_t),
      (const, override));
  MOCK_METHOD(
      ::tokenizers::Result<uint64_t>,
      piece_to_id,
      (const std::string&),
      (const, override));
};

struct Fixture {
  const char* env;
  int64_t input_bound;
  uint64_t next_token;
};

class PrefillFixtureTest : public TestWithParam<Fixture> {};

TEST_P(PrefillFixtureTest, FactoryPrefillsEightTokensWithinMethodBound) {
  executorch::runtime::runtime_init();
  const auto fixture = GetParam();
  const char* model_path = std::getenv(fixture.env);
  ASSERT_NE(model_path, nullptr) << "Missing fixture: " << fixture.env;
  Module module(model_path, Module::LoadMode::File);
  auto meta = module.method_meta("forward");
  ASSERT_TRUE(meta.ok());
  ASSERT_EQ(meta->num_inputs(), 2);
  auto tokens = meta->input_tensor_meta(0);
  ASSERT_TRUE(tokens.ok());
  ASSERT_EQ(tokens->sizes().size(), 2);
  EXPECT_EQ(tokens->sizes()[0], 1);
  EXPECT_EQ(tokens->sizes()[1], fixture.input_bound);
  auto seq_len = module.get("get_max_seq_len");
  auto context_len = module.get("get_max_context_len");
  ASSERT_TRUE(seq_len.ok());
  ASSERT_TRUE(context_len.ok());
  EXPECT_EQ(seq_len->toInt(), 8);
  EXPECT_EQ(context_len->toInt(), 16);

  auto tokenizer = std::make_unique<StrictMock<MockTokenizer>>();
  EXPECT_CALL(*tokenizer, encode("eight tokens", 0, 0))
      .WillOnce([](const std::string&, int8_t, int8_t) {
        return ::tokenizers::Result<std::vector<uint64_t>>(
            std::vector<uint64_t>(8, 1));
      });
  auto runner = create_text_llm_runner(
      model_path,
      std::move(tokenizer),
      std::vector<std::string>{},
      0.0f,
      nullptr,
      "forward",
      Module::LoadMode::File);
  ASSERT_NE(runner, nullptr);
  ASSERT_EQ(runner->load(), Error::Ok);
  auto result = runner->prefill("eight tokens", 0, 0);
  ASSERT_TRUE(result.ok()) << static_cast<int>(result.error());
  // The model's peak is calls + 2 * sum(tokens) + start_pos. A 7+1
  // prefill ends at 2+2+7; eight single-token calls end at 8+2+7.
  EXPECT_EQ(result.get(), fixture.next_token);
}

INSTANTIATE_TEST_SUITE_P(
    BoundedInputs,
    PrefillFixtureTest,
    Values(
        Fixture{"ET_PREFILL_DYNAMIC_PATH", 7, 11},
        Fixture{"ET_PREFILL_STATIC_PATH", 1, 17}));
} // namespace
