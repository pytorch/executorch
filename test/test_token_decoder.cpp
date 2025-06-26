/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <gtest/gtest.h>
#include <pytorch/tokenizers/token_decoder.h>

namespace tokenizers {

// Test ReplaceTokenDecoder
TEST(ReplaceTokenDecoderTest, TestBasicReplace) {
  ReplaceTokenDecoder decoder("▁", " ");

  EXPECT_EQ(decoder.decode("▁Hello"), " Hello");
  EXPECT_EQ(decoder.decode("▁world!"), " world!");
  EXPECT_EQ(decoder.decode("Hello▁world"), "Hello world");
  EXPECT_EQ(decoder.decode("no_replacement"), "no_replacement");
}

TEST(ReplaceTokenDecoderTest, TestMultipleReplacements) {
  ReplaceTokenDecoder decoder("▁", " ");

  EXPECT_EQ(decoder.decode("▁Hello▁world▁!"), " Hello world !");
}

TEST(ReplaceTokenDecoderTest, TestEmptyPattern) {
  ReplaceTokenDecoder decoder("", "X");

  // Empty pattern should not cause infinite loop
  EXPECT_EQ(decoder.decode("test"), "test");
}

// Test ByteFallbackTokenDecoder
TEST(ByteFallbackTokenDecoderTest, TestValidHexTokens) {
  ByteFallbackTokenDecoder decoder;

  EXPECT_EQ(decoder.decode("<0x41>"), "A"); // 0x41 = 65 = 'A'
  EXPECT_EQ(decoder.decode("<0x42>"), "B"); // 0x42 = 66 = 'B'
  EXPECT_EQ(decoder.decode("<0x20>"), " "); // 0x20 = 32 = space
  EXPECT_EQ(decoder.decode("<0x00>"), std::string(1, '\0')); // null byte
}

TEST(ByteFallbackTokenDecoderTest, TestInvalidHexTokens) {
  ByteFallbackTokenDecoder decoder;

  // Invalid format - should return original token
  EXPECT_EQ(decoder.decode("<0xGG>"), "<0xGG>");
  EXPECT_EQ(decoder.decode("<0x>"), "<0x>");
  EXPECT_EQ(decoder.decode("0x41>"), "0x41>");
  EXPECT_EQ(decoder.decode("<0x41"), "<0x41");
  EXPECT_EQ(decoder.decode("regular_token"), "regular_token");
}

TEST(ByteFallbackTokenDecoderTest, TestOutOfRangeValues) {
  ByteFallbackTokenDecoder decoder;

  // Values > 255 should return original token
  EXPECT_EQ(decoder.decode("<0x100>"), "<0x100>");
  EXPECT_EQ(decoder.decode("<0xFFFF>"), "<0xFFFF>");
}

// Test FuseTokenDecoder
TEST(FuseTokenDecoderTest, TestPassthrough) {
  FuseTokenDecoder decoder;

  EXPECT_EQ(decoder.decode("test"), "test");
  EXPECT_EQ(decoder.decode("▁Hello"), "▁Hello");
  EXPECT_EQ(decoder.decode("<0x41>"), "<0x41>");
  EXPECT_EQ(decoder.decode(""), "");
}

// Test SequenceTokenDecoder
TEST(SequenceTokenDecoderTest, TestEmptySequence) {
  std::vector<TokenDecoder::Ptr> decoders;
  SequenceTokenDecoder sequence_decoder(std::move(decoders));

  EXPECT_EQ(sequence_decoder.decode("test"), "test");
}

TEST(SequenceTokenDecoderTest, TestSingleDecoder) {
  std::vector<TokenDecoder::Ptr> decoders;
  decoders.push_back(TokenDecoder::Ptr(new ReplaceTokenDecoder("▁", " ")));

  SequenceTokenDecoder sequence_decoder(std::move(decoders));

  EXPECT_EQ(sequence_decoder.decode("▁Hello"), " Hello");
}

TEST(SequenceTokenDecoderTest, TestMultipleDecoders) {
  std::vector<TokenDecoder::Ptr> decoders;

  // Add Replace decoder to replace ▁ with space
  decoders.push_back(TokenDecoder::Ptr(new ReplaceTokenDecoder("▁", " ")));

  // Add ByteFallback decoder
  decoders.push_back(TokenDecoder::Ptr(new ByteFallbackTokenDecoder()));

  // Add Fuse decoder
  decoders.push_back(TokenDecoder::Ptr(new FuseTokenDecoder()));

  SequenceTokenDecoder sequence_decoder(std::move(decoders));

  // Test cases
  EXPECT_EQ(sequence_decoder.decode("▁Hello"), " Hello");
  EXPECT_EQ(sequence_decoder.decode("▁world!"), " world!");
  EXPECT_EQ(sequence_decoder.decode("<0x41>"), "A");
  EXPECT_EQ(sequence_decoder.decode("normal_token"), "normal_token");
}

TEST(SequenceTokenDecoderTest, TestComplexSequence) {
  std::vector<TokenDecoder::Ptr> decoders;

  // First replace underscores with spaces
  decoders.push_back(TokenDecoder::Ptr(new ReplaceTokenDecoder("_", " ")));

  // Then replace ▁ with spaces
  decoders.push_back(TokenDecoder::Ptr(new ReplaceTokenDecoder("▁", " ")));

  // Then handle byte fallback
  decoders.push_back(TokenDecoder::Ptr(new ByteFallbackTokenDecoder()));

  SequenceTokenDecoder sequence_decoder(std::move(decoders));

  EXPECT_EQ(sequence_decoder.decode("Hello_world"), "Hello world");
  EXPECT_EQ(sequence_decoder.decode("▁test_token"), " test token");
}

// Test TokenDecoderConfig parsing and creation
TEST(TokenDecoderConfigTest, TestReplaceConfig) {
  nlohmann::json config = {
      {"type", "Replace"}, {"pattern", {{"String", "▁"}}}, {"content", " "}};

  TokenDecoderConfig decoder_config;
  decoder_config.parse_json(config);

  EXPECT_EQ(decoder_config.type, "Replace");
  EXPECT_EQ(decoder_config.replace_pattern, "▁");
  EXPECT_EQ(decoder_config.replace_content, " ");

  auto decoder = decoder_config.create();
  EXPECT_EQ(decoder->decode("▁Hello"), " Hello");
}

TEST(TokenDecoderConfigTest, TestSequenceConfig) {
  nlohmann::json config = {
      {"type", "Sequence"},
      {"decoders",
       {{{"type", "Replace"}, {"pattern", {{"String", "▁"}}}, {"content", " "}},
        {{"type", "ByteFallback"}},
        {{"type", "Fuse"}}}}};

  TokenDecoderConfig decoder_config;
  decoder_config.parse_json(config);

  EXPECT_EQ(decoder_config.type, "Sequence");
  EXPECT_EQ(decoder_config.sequence_decoders.size(), 3);

  auto decoder = decoder_config.create();
  EXPECT_EQ(decoder->decode("▁Hello"), " Hello");
  EXPECT_EQ(decoder->decode("<0x41>"), "A");
}

} // namespace tokenizers
