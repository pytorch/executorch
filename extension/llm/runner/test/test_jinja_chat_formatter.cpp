/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/jinja_chat_formatter.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using executorch::extension::llm::ChatConversation;
using executorch::extension::llm::ChatMessage;
using executorch::extension::llm::ChatTemplateType;
using executorch::extension::llm::JinjaChatFormatter;
using executorch::extension::llm::parseChatTemplateType;
using testing::HasSubstr;

TEST(JinjaChatFormatter, Llama3SingleMessage) {
  auto formatter = JinjaChatFormatter::fromTemplate(ChatTemplateType::Llama3);
  const std::string prompt = "Test prompt";
  const std::string system_prompt = "You are a helpful assistant.";
  // Note: The Jinja template uses {%- ... -%} which strips whitespace,
  // so the output has \n\n after each <|end_header_id|> and content,
  // but no trailing \n\n at the end due to the {%- endif -%} stripping.
  const std::string expected =
      "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
      system_prompt +
      "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + prompt +
      "<|eot_id|><|start_header_id|>assistant<|end_header_id|>";

  EXPECT_EQ(formatter->format(prompt, system_prompt), expected);
}

TEST(JinjaChatFormatter, Gemma3SingleMessage) {
  auto formatter = JinjaChatFormatter::fromTemplate(ChatTemplateType::Gemma3);
  const std::string result = formatter->format("Hello!");

  EXPECT_THAT(result, HasSubstr("<bos>"));
  EXPECT_THAT(result, HasSubstr("<start_of_turn>user"));
  EXPECT_THAT(result, HasSubstr("Hello!"));
  EXPECT_THAT(result, HasSubstr("<start_of_turn>model"));
}

TEST(JinjaChatFormatter, Llama3WithoutSystemPrompt) {
  auto formatter = JinjaChatFormatter::fromTemplate(ChatTemplateType::Llama3);
  const std::string result = formatter->format("Hello!");

  EXPECT_THAT(result, HasSubstr("<|begin_of_text|>"));
  EXPECT_THAT(result, HasSubstr("<|start_header_id|>user<|end_header_id|>"));
  EXPECT_THAT(result, HasSubstr("Hello!"));
  EXPECT_THAT(result, HasSubstr("<|start_header_id|>assistant<|end_header_id|>"));
  // Should not contain system header when no system prompt
  EXPECT_THAT(result, ::testing::Not(HasSubstr("<|start_header_id|>system")));
}

TEST(JinjaChatFormatter, Llama3IncludesBos) {
  auto formatter = JinjaChatFormatter::fromTemplate(ChatTemplateType::Llama3);
  EXPECT_TRUE(formatter->includesBos());
}

TEST(JinjaChatFormatter, Gemma3IncludesBos) {
  auto formatter = JinjaChatFormatter::fromTemplate(ChatTemplateType::Gemma3);
  EXPECT_TRUE(formatter->includesBos());
}

TEST(JinjaChatFormatter, Llama3ModelTokens) {
  auto formatter = JinjaChatFormatter::fromTemplate(ChatTemplateType::Llama3);
  const auto& tokens = formatter->getModelTokens();

  EXPECT_EQ(tokens.bos_token, "<|begin_of_text|>");
  EXPECT_EQ(tokens.eos_token, "<|eot_id|>");
  EXPECT_EQ(tokens.stop_tokens.size(), 2);
}

TEST(JinjaChatFormatter, Gemma3ModelTokens) {
  auto formatter = JinjaChatFormatter::fromTemplate(ChatTemplateType::Gemma3);
  const auto& tokens = formatter->getModelTokens();

  EXPECT_EQ(tokens.bos_token, "<bos>");
  EXPECT_EQ(tokens.eos_token, "<end_of_turn>");
  EXPECT_EQ(tokens.stop_tokens.size(), 2);
}

TEST(JinjaChatFormatter, FormatConversationMultiTurn) {
  auto formatter = JinjaChatFormatter::fromTemplate(ChatTemplateType::Llama3);

  ChatConversation conversation;
  conversation.bos_token = "<|begin_of_text|>";
  conversation.eos_token = "<|eot_id|>";
  conversation.add_generation_prompt = true;
  conversation.messages = {
      {"user", "Hello"},
      {"assistant", "Hi there!"},
      {"user", "How are you?"},
  };

  const std::string result = formatter->formatConversation(conversation);

  EXPECT_THAT(result, HasSubstr("Hello"));
  EXPECT_THAT(result, HasSubstr("Hi there!"));
  EXPECT_THAT(result, HasSubstr("How are you?"));
  EXPECT_THAT(result, HasSubstr("<|start_header_id|>assistant<|end_header_id|>"));
}

TEST(JinjaChatFormatter, FromStringLlama3Template) {
  const std::string llama_template =
      "{{ bos_token }}<|start_header_id|>user<|end_header_id|>\n\n"
      "{{ messages[0].content }}<|eot_id|>";

  auto formatter = JinjaChatFormatter::fromString(llama_template);

  // Should detect Llama3 type from the template content
  const std::string result = formatter->format("Test");
  EXPECT_THAT(result, HasSubstr("Test"));
}

TEST(JinjaChatFormatter, FromStringGemma3Template) {
  const std::string gemma_template =
      "{{ bos_token }}<start_of_turn>user\n"
      "{{ messages[0].content }}<end_of_turn>";

  auto formatter = JinjaChatFormatter::fromString(gemma_template);

  const std::string result = formatter->format("Test");
  EXPECT_THAT(result, HasSubstr("Test"));
}

TEST(JinjaChatFormatter, UnsupportedTemplateTypeThrows) {
  EXPECT_THROW(
      JinjaChatFormatter::fromTemplate(ChatTemplateType::None),
      std::runtime_error);
}

// Tests for parseChatTemplateType
TEST(ParseChatTemplateType, ParseNone) {
  EXPECT_EQ(parseChatTemplateType("none"), ChatTemplateType::None);
  EXPECT_EQ(parseChatTemplateType("None"), ChatTemplateType::None);
  EXPECT_EQ(parseChatTemplateType("NONE"), ChatTemplateType::None);
}

TEST(ParseChatTemplateType, ParseLlama3) {
  EXPECT_EQ(parseChatTemplateType("llama3"), ChatTemplateType::Llama3);
  EXPECT_EQ(parseChatTemplateType("LLAMA3"), ChatTemplateType::Llama3);
  EXPECT_EQ(parseChatTemplateType("Llama3"), ChatTemplateType::Llama3);
}

TEST(ParseChatTemplateType, ParseLlama32Variants) {
  EXPECT_EQ(parseChatTemplateType("llama3.2"), ChatTemplateType::Llama32);
  EXPECT_EQ(parseChatTemplateType("llama32"), ChatTemplateType::Llama32);
  EXPECT_EQ(parseChatTemplateType("llama3_2"), ChatTemplateType::Llama32);
  EXPECT_EQ(parseChatTemplateType("LLAMA3.2"), ChatTemplateType::Llama32);
}

TEST(ParseChatTemplateType, ParseGemma3) {
  EXPECT_EQ(parseChatTemplateType("gemma3"), ChatTemplateType::Gemma3);
  EXPECT_EQ(parseChatTemplateType("GEMMA3"), ChatTemplateType::Gemma3);
  EXPECT_EQ(parseChatTemplateType("Gemma3"), ChatTemplateType::Gemma3);
}

TEST(ParseChatTemplateType, ParseCustom) {
  EXPECT_EQ(parseChatTemplateType("custom"), ChatTemplateType::Custom);
  EXPECT_EQ(parseChatTemplateType("jinja"), ChatTemplateType::Custom);
  EXPECT_EQ(parseChatTemplateType("CUSTOM"), ChatTemplateType::Custom);
}

TEST(ParseChatTemplateType, ParseUnknownReturnsNone) {
  EXPECT_EQ(parseChatTemplateType("unknown"), ChatTemplateType::None);
  EXPECT_EQ(parseChatTemplateType(""), ChatTemplateType::None);
  EXPECT_EQ(parseChatTemplateType("invalid"), ChatTemplateType::None);
}

TEST(JinjaChatFormatter, Llama32SingleMessage) {
  auto formatter = JinjaChatFormatter::fromTemplate(ChatTemplateType::Llama32);
  const std::string result = formatter->format("Hello!");

  // Llama32 uses the same template as Llama3
  EXPECT_THAT(result, HasSubstr("<|begin_of_text|>"));
  EXPECT_THAT(result, HasSubstr("<|start_header_id|>user<|end_header_id|>"));
  EXPECT_THAT(result, HasSubstr("Hello!"));
  EXPECT_THAT(result, HasSubstr("<|start_header_id|>assistant<|end_header_id|>"));
}

TEST(JinjaChatFormatter, Llama32IncludesBos) {
  auto formatter = JinjaChatFormatter::fromTemplate(ChatTemplateType::Llama32);
  EXPECT_TRUE(formatter->includesBos());
}

TEST(JinjaChatFormatter, Llama32ModelTokens) {
  auto formatter = JinjaChatFormatter::fromTemplate(ChatTemplateType::Llama32);
  const auto& tokens = formatter->getModelTokens();

  EXPECT_EQ(tokens.bos_token, "<|begin_of_text|>");
  EXPECT_EQ(tokens.eos_token, "<|eot_id|>");
  EXPECT_EQ(tokens.stop_tokens.size(), 2);
}
