/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Third Party
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <re2/re2.h>

// Local
#include <pytorch/tokenizers/pre_tokenizer.h>

using json = nlohmann::json;
using namespace tokenizers;

// Helpers /////////////////////////////////////////////////////////////////////

static void assert_split_match(
    const PreTokenizer& ptok,
    const std::string& prompt,
    const std::vector<std::string>& expected) {
  re2::StringPiece prompt_view(prompt);
  const auto& got = ptok.pre_tokenize(prompt_view);
  EXPECT_EQ(expected.size(), got.size());
  for (auto i = 0; i < got.size(); ++i) {
    EXPECT_EQ(expected[i], got[i]);
  }
}

// RegexPreTokenizer ///////////////////////////////////////////////////////////
class RegexPreTokenizerTest : public ::testing::Test {};

// Test the basic construction
TEST_F(RegexPreTokenizerTest, Construct) {
  RegexPreTokenizer ptok("[0-9]+");
}

// Test basic splitting using the expression for Tiktoken
TEST_F(RegexPreTokenizerTest, TiktokenExpr) {
  RegexPreTokenizer ptok(
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)");
  assert_split_match(
      ptok, "How are you doing?", {"How", " are", " you", " doing", "?"});
}

// DigitsPreTokenizer //////////////////////////////////////////////////////////
class DigitsPreTokenizerTest : public ::testing::Test {};

// Test digit splitting with individual digits
TEST_F(DigitsPreTokenizerTest, IndividualDigits) {
  DigitsPreTokenizer ptok(true);
  assert_split_match(
      ptok,
      "The number 1 then 234 then 5.",
      {"The number ", "1", " then ", "2", "3", "4", " then ", "5", "."});
}

// Test digit splitting with contiguous digits
TEST_F(DigitsPreTokenizerTest, ContiguousDigits) {
  DigitsPreTokenizer ptok(false);
  assert_split_match(
      ptok,
      "The number 1 then 234 then 5.",
      {"The number ", "1", " then ", "234", " then ", "5", "."});
}

// ByteLevelPreTokenizer ///////////////////////////////////////////////////////
class ByteLevelPreTokenizerTest : public ::testing::Test {};

TEST_F(ByteLevelPreTokenizerTest, PreTokenizeDefault) {
  ByteLevelPreTokenizer ptok;
  assert_split_match(ptok, "Hello World", {"ĠHello", "ĠWorld"});
  assert_split_match(
      ptok,
      "The number 1 then 234 then 5.",
      {"ĠThe", "Ġnumber", "Ġ1", "Ġthen", "Ġ234", "Ġthen", "Ġ5", "."});
}

TEST_F(ByteLevelPreTokenizerTest, PreTokenizeNoPrefix) {
  ByteLevelPreTokenizer ptok(false);
  assert_split_match(ptok, "Hello World", {"Hello", "ĠWorld"});
}

TEST_F(ByteLevelPreTokenizerTest, PreTokenizeCustomRegex) {
  ByteLevelPreTokenizer ptok(false, R"(o)");
  assert_split_match(ptok, "Hello World", {"Hell", "o", "ĠW", "o", "rld"});
}

// SequencePreTokenizer ////////////////////////////////////////////////////////
class SequencePreTokenizerTest : public ::testing::Test {};

TEST_F(SequencePreTokenizerTest, PreTokenizeDigitAndByteLevel) {
  PreTokenizer::Ptr dptok(new DigitsPreTokenizer(true));
  PreTokenizer::Ptr bptok(new ByteLevelPreTokenizer(false));
  SequencePreTokenizer ptok({dptok, bptok});
  assert_split_match(
      ptok,
      "The number 1 then 234 then 5.",
      {"The",
       "Ġnumber",
       "Ġ",
       "1",
       "Ġthen",
       "Ġ",
       "2",
       "3",
       "4",
       "Ġthen",
       "Ġ",
       "5",
       "."});
}

// PreTokenizerConfig //////////////////////////////////////////////////////////
//
// NOTE: When adding a new pre-tokenizer or changing arguments, add it to these
//  tests!
class PreTokenizerConfigTest : public ::testing::Test {};

TEST_F(PreTokenizerConfigTest, AllTypesSuccess) {
  // Regex
  PreTokenizerConfig("Split").set_pattern(R"(o)").create();

  // Digits
  PreTokenizerConfig("Digits").create();
  PreTokenizerConfig("Digits").set_individual_digits(true).create();
  PreTokenizerConfig("Digits").set_individual_digits(false).create();

  // ByteLevel
  PreTokenizerConfig("ByteLevel").create();
  PreTokenizerConfig("ByteLevel").set_pattern(R"(o)").create();
  PreTokenizerConfig("ByteLevel").set_add_prefix_space(true).create();
  PreTokenizerConfig("ByteLevel")
      .set_add_prefix_space(false)
      .set_pattern(R"(o)")
      .create();

  // Sequence
  PreTokenizerConfig("Sequence")
      .set_pretokenizers(
          {PreTokenizerConfig("Digits"), PreTokenizerConfig("ByteLevel")})
      .create();
}

TEST_F(PreTokenizerConfigTest, AllTypesFailureCases) {
  // Regex
  EXPECT_THROW(PreTokenizerConfig("Split").create(), std::runtime_error);

  // Sequence
  EXPECT_THROW(PreTokenizerConfig("Sequence").create(), std::runtime_error);
  EXPECT_THROW(
      PreTokenizerConfig("Sequence").set_pretokenizers({}).create(),
      std::runtime_error);
  EXPECT_THROW(
      PreTokenizerConfig("Sequence")
          .set_pretokenizers({PreTokenizerConfig("Split")})
          .create(),
      std::runtime_error);

  // Unsupported
  EXPECT_THROW(PreTokenizerConfig("Unsupported").create(), std::runtime_error);
}

TEST_F(PreTokenizerConfigTest, ParseJson) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(json{
                            {"type", "Sequence"},
                            {"pretokenizers",
                             json{
                                 json{
                                     {"type", "Digits"},
                                     {"individual_digits", true},
                                 },
                                 json{
                                     {"type", "ByteLevel"},
                                     {"add_prefix_space", false},
                                 },
                             }},
                        })
                        .create();
  assert_split_match(
      *ptok,
      "The number 1 then 234 then 5.",
      {"The",
       "Ġnumber",
       "Ġ",
       "1",
       "Ġthen",
       "Ġ",
       "2",
       "3",
       "4",
       "Ġthen",
       "Ġ",
       "5",
       "."});
}

TEST_F(PreTokenizerConfigTest, ParseJsonOptionalKey) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(json{
                            {"type", "Digits"},
                        })
                        .create();
  assert_split_match(
      *ptok,
      "The number 1 then 234 then 5.",
      {"The number ", "1", " then ", "234", " then ", "5", "."});
}

TEST_F(PreTokenizerConfigTest, Split) {
  PreTokenizerConfig config;
  const auto ptok =
      config
          .parse_json(json{
              {"type", "Split"},
              {"pattern",
               R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)"},
          })
          .create();
  assert_split_match(*ptok, "Hello World", {"Hello", " World"});
}
