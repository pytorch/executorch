/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace example {

/**
 * Supported chat template formats for different model families.
 */
enum class ChatFormat {
  None,   // No formatting (pass-through)
  Llama3, // Llama 3.x Instruct models
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
 * Llama 3.x Instruct chat format.
 * Template: <|begin_of_text|><|start_header_id|>...<|eot_id|>...
 */
class Llama3ChatFormatter : public ChatFormatter {
 public:
  std::string format(
      const std::string& prompt,
      const std::string& system_prompt = "") const override {
    std::string result = "<|begin_of_text|>";
    if (!system_prompt.empty()) {
      result += "<|start_header_id|>system<|end_header_id|>\n\n";
      result += system_prompt;
      result += "<|eot_id|>";
    }
    result += "<|start_header_id|>user<|end_header_id|>\n\n";
    result += prompt;
    result += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    return result;
  }

  bool includes_bos() const override {
    return true; // <|begin_of_text|> is the BOS token
  }
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

/**
 * Parse a chat format string into the corresponding enum value.
 *
 * @param format_str String identifier (e.g., "llama3", "none")
 * @return ChatFormat enum value, defaults to None for unknown formats
 */
inline ChatFormat parse_chat_format(const std::string& format_str) {
  static const std::unordered_map<std::string, ChatFormat> format_map = {
      {"none", ChatFormat::None},
      {"llama3", ChatFormat::Llama3},
  };
  auto it = format_map.find(format_str);
  if (it != format_map.end()) {
    return it->second;
  }
  return ChatFormat::None;
}

/**
 * Get a human-readable list of supported chat formats.
 */
inline std::string get_supported_formats() {
  return "llama3, none";
}

/**
 * Factory function to create the appropriate ChatFormatter instance.
 *
 * @param format The chat format to use
 * @return Unique pointer to a ChatFormatter instance
 */
inline std::unique_ptr<ChatFormatter> create_chat_formatter(ChatFormat format) {
  switch (format) {
    case ChatFormat::Llama3:
      return std::make_unique<Llama3ChatFormatter>();
    case ChatFormat::None:
    default:
      return std::make_unique<NoChatFormatter>();
  }
}

} // namespace example
