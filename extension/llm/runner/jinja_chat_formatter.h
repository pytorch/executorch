#pragma once

#include <executorch/extension/llm/chat_template/chat_templates.h>
#include <executorch/extension/llm/runner/chat_types.h>

#include <filesystem>
#include <memory>
#include <string>

namespace jinja2 {
class Template;
}

namespace executorch::extension::llm {

class JinjaChatFormatter {
 public:
  static std::unique_ptr<JinjaChatFormatter> fromTemplate(ChatTemplateType type);
  static std::unique_ptr<JinjaChatFormatter> fromString(
      const std::string& template_str);
  static std::unique_ptr<JinjaChatFormatter> fromFile(
      const std::filesystem::path& path);

  ~JinjaChatFormatter();

  std::string format(
      const std::string& prompt,
      const std::string& system_prompt = "") const;
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
