/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/chat_template/chat_templates.h>
#include <executorch/extension/llm/runner/chat_types.h>

#include <memory>
#include <string>

namespace jinja2 {
class Template;
}

namespace executorch::extension::llm {

class JinjaChatFormatter {
 public:
  static std::unique_ptr<JinjaChatFormatter> fromTemplate(
      ChatTemplateType type);
  // fromString/fromFile infer only the model family from template syntax.
  // Llama3 and Llama3.2 share template markers and token defaults today.
  static std::unique_ptr<JinjaChatFormatter> fromString(
      const std::string& template_str);
  static std::unique_ptr<JinjaChatFormatter> fromFile(const std::string& path);

  ~JinjaChatFormatter();

  std::string format(
      const std::string& prompt,
      const std::string& system_prompt = "") const;
  // Custom templates whose tokens cannot be inferred should use
  // formatConversation() so the caller can provide bos/eos tokens explicitly.
  std::string formatConversation(const ChatConversation& conversation) const;

  bool includesBos() const {
    return includes_bos_;
  }

  const ModelTokens& getModelTokens() const {
    return model_tokens_;
  }

 private:
  JinjaChatFormatter(const std::string& template_str, ChatTemplateType type);

  std::string template_str_;
  ChatTemplateType type_;
  ModelTokens model_tokens_;
  bool includes_bos_ = false;
  std::unique_ptr<jinja2::Template> compiled_template_;
};

ChatTemplateType parseChatTemplateType(const std::string& type_str);

} // namespace executorch::extension::llm
