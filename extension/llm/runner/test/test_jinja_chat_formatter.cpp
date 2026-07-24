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
      system_prompt + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" +
      prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>";

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
  EXPECT_THAT(
      result, HasSubstr("<|start_header_id|>assistant<|end_header_id|>"));
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
  EXPECT_THAT(
      result, HasSubstr("<|start_header_id|>assistant<|end_header_id|>"));
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
  EXPECT_THAT(
      result, HasSubstr("<|start_header_id|>assistant<|end_header_id|>"));
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

TEST(JinjaChatFormatter, UniversalJinjaGenericTemplate) {
  const std::string generic_template =
      "{{ bos_token }}"
      "{%- for message in messages -%}"
      "<|{{ message.role }}|>\n{{ message.content }}<|end|>\n"
      "{%- endfor -%}"
      "{%- if add_generation_prompt -%}<|assistant|>\n{%- endif -%}";

  auto formatter = JinjaChatFormatter::fromString(generic_template);
  ChatConversation conv;
  conv.bos_token = "<s>";
  conv.add_generation_prompt = true;
  conv.messages.push_back(ChatMessage{"user", "Hi there"});

  const std::string result = formatter->formatConversation(conv);

  EXPECT_THAT(result, HasSubstr("<s>"));
  EXPECT_THAT(result, HasSubstr("<|user|>"));
  EXPECT_THAT(result, HasSubstr("Hi there"));
  EXPECT_THAT(result, HasSubstr("<|assistant|>"));
}

TEST(JinjaChatFormatter, UniversalJinjaToolsAwareTemplate) {
  const std::string tools_template =
      "{%- if tools is not none -%}"
      "tools_present"
      "{%- else -%}"
      "no_tools"
      "{%- endif -%}";

  auto formatter = JinjaChatFormatter::fromString(tools_template);
  ChatConversation conv;
  conv.add_generation_prompt = false;

  EXPECT_EQ(formatter->formatConversation(conv), "no_tools");
}

TEST(JinjaChatFormatter, UniversalJinjaNormalizedNotToolsIsNone) {
  const std::string template_str =
      "{%- if not tools is none -%}defined{%- else -%}none{%- endif -%}";

  auto formatter = JinjaChatFormatter::fromString(template_str);
  ChatConversation conv;
  conv.add_generation_prompt = false;

  EXPECT_EQ(formatter->formatConversation(conv), "none");
}

TEST(JinjaChatFormatter, UniversalJinjaToolOutputObjectLiteral) {
  const std::string template_str =
      R"({%- for message in messages -%}{{ { "output": message.content } | tojson }}{%- endfor -%})";

  auto formatter = JinjaChatFormatter::fromString(template_str);
  ChatConversation conv;
  conv.add_generation_prompt = false;
  conv.messages.push_back(ChatMessage{"tool", "done"});

  EXPECT_EQ(formatter->formatConversation(conv), R"({"output":"done"})");
}

TEST(JinjaChatFormatter, VllmLlama32PythonicToolTemplate) {
  // Mirrors vLLM's examples/tool_chat_template_llama3.2_pythonic.jinja.
  const std::string template_str = R"({{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = false %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question." %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call functions, please respond with a python list of the calls. " }}
    {{- 'Respond in the format [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)] ' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
    {%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a python list for function calls " }}
    {{- "with their proper arguments to best answer the given prompt.\n\n" }}
    {{- 'Respond in the format [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)] ' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n[' -}}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- tool_call.name + '(' -}}
            {%- for param in tool_call.arguments %}
                {{- param + '=' -}}
                {{- "%s" | format(tool_call.arguments[param]) -}}
                {% if not loop.last %}, {% endif %}
            {%- endfor %}
            {{- ')' -}}
            {% if not loop.last %}, {% endif %}
        {%- endfor %}
        {{- ']<|eot_id|>' -}}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping %}
            {{- message.content | tojson }}
        {%- else %}
            {{- { "output": message.content } | tojson }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
)";

  auto formatter = JinjaChatFormatter::fromString(template_str);
  const std::string result = formatter->format("Hello!");

  EXPECT_THAT(result, HasSubstr("<|begin_of_text|>"));
  EXPECT_THAT(result, HasSubstr("<|start_header_id|>system<|end_header_id|>"));
  EXPECT_THAT(result, HasSubstr("Today Date: 26 Jul 2024"));
  EXPECT_THAT(result, HasSubstr("<|start_header_id|>user<|end_header_id|>"));
  EXPECT_THAT(result, HasSubstr("Hello!"));
  EXPECT_THAT(
      result, HasSubstr("<|start_header_id|>assistant<|end_header_id|>"));
}
