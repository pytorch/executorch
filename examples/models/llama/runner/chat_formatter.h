/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <executorch/extension/llm/runner/jinja_chat_formatter.h>

namespace example {

/**
 * Supported chat template formats for different model families.
 */
enum class ChatFormat {
  None,   // No formatting (pass-through)
  Llama3, // Llama 3.x Instruct models
  Gemma3, // Gemma 3 Instruct models
  Jinja,  // Custom Jinja template from file
};

/**
 * Abstract base class for chat formatters.
 * Implementations format user prompts into model-specific chat templates.
 */
class ChatFormatter {
 public:
  virtual ~ChatFormatter() = default;

  /**
   * Format a user prompt into the model's expected chat template.
   *
   * @param prompt The user's input message
   * @param system_prompt Optional system prompt to set model behavior
   * @return Formatted string ready for tokenization
   */
  virtual std::string format(
      const std::string& prompt,
      const std::string& system_prompt = "") const = 0;

  /**
   * Whether this formatter includes BOS token in the template.
   * If true, the runner should not prepend additional BOS tokens.
   */
  virtual bool includes_bos() const = 0;
};

/**
 * No formatting (pass-through).
 * Use when the prompt is already formatted or for base models.
 */
class NoChatFormatter : public ChatFormatter {
 public:
  std::string format(
      const std::string& prompt,
      const std::string& system_prompt = "") const override {
    (void)system_prompt; // Unused in pass-through mode
    return prompt;
  }

  bool includes_bos() const override {
    return false; // User controls BOS token
  }
};

class JinjaChatFormatterAdapter : public ChatFormatter {
 public:
  explicit JinjaChatFormatterAdapter(
      std::unique_ptr<executorch::extension::llm::JinjaChatFormatter> formatter)
      : formatter_(std::move(formatter)) {}

  std::string format(
      const std::string& prompt,
      const std::string& system_prompt = "") const override {
    return formatter_->format(prompt, system_prompt);
  }

  bool includes_bos() const override {
    return formatter_->includesBos();
  }

 private:
  std::unique_ptr<executorch::extension::llm::JinjaChatFormatter> formatter_;
};

/**
 * Parse a chat format string into the corresponding enum value.
 *
 * The lookup is case-insensitive and tolerant of surrounding whitespace,
 * so values like "Llama3", " llama3 ", or "LLAMA3" all map to
 * ChatFormat::Llama3.
 *
 * @param format_str String identifier (e.g., "llama3", "none")
 * @return ChatFormat enum value, defaults to None for unknown formats
 */
inline ChatFormat parse_chat_format(const std::string& format_str) {
  static const std::unordered_map<std::string, ChatFormat> format_map = {
      {"none", ChatFormat::None},
      {"llama3", ChatFormat::Llama3},
      {"llama3.2", ChatFormat::Llama3},
      {"llama32", ChatFormat::Llama3},
      {"llama3_2", ChatFormat::Llama3},
      {"gemma3", ChatFormat::Gemma3},
      {"jinja", ChatFormat::Jinja},
  };

  // Trim whitespace and lowercase to make CLI input forgiving.
  std::string normalized = format_str;
  const auto first_non_ws =
      normalized.find_first_not_of(" \t\n\r\f\v");
  const auto last_non_ws = normalized.find_last_not_of(" \t\n\r\f\v");
  if (first_non_ws == std::string::npos) {
    return ChatFormat::None;
  }
  normalized = normalized.substr(first_non_ws, last_non_ws - first_non_ws + 1);
  std::transform(
      normalized.begin(),
      normalized.end(),
      normalized.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  auto it = format_map.find(normalized);
  if (it != format_map.end()) {
    return it->second;
  }
  return ChatFormat::None;
}

/**
 * Get a human-readable list of supported chat formats.
 */
inline std::string get_supported_formats() {
  return "llama3, gemma3, jinja, none";
}

/**
 * Factory function to create the appropriate ChatFormatter instance.
 *
 * Universal Jinja support: when `template_file` is non-empty, any
 * HuggingFace-style or vLLM-style chat template (e.g. files from
 * https://github.com/vllm-project/vllm/tree/main/examples) can be passed in
 * regardless of the `format` value. The Jinja formatter will load and
 * render the template directly.
 *
 * @param format The chat format to use
 * @param template_file Optional path to a Jinja2 chat template file. If
 *   provided, takes precedence over `format`.
 * @return Unique pointer to a ChatFormatter instance
 * @throws std::invalid_argument if `format == ChatFormat::Jinja` and no
 *   `template_file` is provided.
 */
inline std::unique_ptr<ChatFormatter> create_chat_formatter(
    ChatFormat format,
    const std::string& template_file = "") {
  using executorch::extension::llm::ChatTemplateType;
  using executorch::extension::llm::JinjaChatFormatter;

  if (!template_file.empty()) {
    return std::make_unique<JinjaChatFormatterAdapter>(
        JinjaChatFormatter::fromFile(template_file));
  }

  switch (format) {
    case ChatFormat::Llama3:
      return std::make_unique<JinjaChatFormatterAdapter>(
          JinjaChatFormatter::fromTemplate(ChatTemplateType::Llama3));
    case ChatFormat::Gemma3:
      return std::make_unique<JinjaChatFormatterAdapter>(
          JinjaChatFormatter::fromTemplate(ChatTemplateType::Gemma3));
    case ChatFormat::Jinja:
      throw std::invalid_argument(
          "chat_format=jinja requires --chat_template_file=<path-to-template.jinja>");
    case ChatFormat::None:
    default:
      return std::make_unique<NoChatFormatter>();
  }
}

} // namespace example
