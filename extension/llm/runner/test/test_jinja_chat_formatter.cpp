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

using executorch::extension::llm::ChatTemplateType;
using executorch::extension::llm::JinjaChatFormatter;
using testing::HasSubstr;

TEST(JinjaChatFormatter, Llama3SingleMessage) {
  auto formatter = JinjaChatFormatter::fromTemplate(ChatTemplateType::Llama3);
  const std::string prompt = "Test prompt";
  const std::string system_prompt = "You are a helpful assistant.";
  const std::string expected =
      "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
      system_prompt +
      "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + prompt +
      "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

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
