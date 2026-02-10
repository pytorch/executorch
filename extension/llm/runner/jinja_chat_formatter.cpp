#include <executorch/extension/llm/runner/jinja_chat_formatter.h>

#include <jinja2cpp/reflected_value.h>
#include <jinja2cpp/template.h>
#include <jinja2cpp/value.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace executorch::extension::llm {
namespace {

std::string readFileToString(const std::filesystem::path& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open template file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

bool templateIncludesBos(
    const std::string& template_str,
    const ModelTokens& model_tokens) {
  if (!model_tokens.bos_token.empty() &&
      template_str.find(model_tokens.bos_token) != std::string::npos) {
    return true;
  }
  return template_str.find("bos_token") != std::string::npos;
}

std::string normalizeTemplate(std::string input) {
  constexpr std::array<std::pair<std::string_view, std::string_view>, 10>
      replacements = {{
          {"tools = none", "tools = []"},
          {"tools = None", "tools = []"},
          {"tools is not none", "tools"},
          {"tools is not None", "tools"},
          {"not tools is none", "not tools"},
          {"not tools is None", "not tools"},
          {"tools is none", "not tools"},
          {"tools is None", "not tools"},
          {"messages[1:]", "messages_tail"},
          {"{ \"output\": message.content } | tojson", "message.content | tojson"},
      }};
  // Handle special case that can't be constexpr due to escape sequence
  const std::pair<std::string, std::string> gemmaReplacement = {
      "{{'<start_of_turn>model\\n'}}", "{{ '<start_of_turn>model\\n' }}"};
  for (const auto& replacement : replacements) {
    size_t pos = 0;
    while ((pos = input.find(replacement.first, pos)) != std::string::npos) {
      input.replace(pos, replacement.first.size(), replacement.second);
      pos += replacement.second.size();
    }
  }
  // Apply the gemma replacement separately
  size_t pos = 0;
  while ((pos = input.find(gemmaReplacement.first, pos)) != std::string::npos) {
    input.replace(pos, gemmaReplacement.first.size(), gemmaReplacement.second);
    pos += gemmaReplacement.second.size();
  }
  return input;
}

ChatTemplateType detectTemplateType(const std::string& template_str) {
  if (template_str.find("<start_of_turn>") != std::string::npos) {
    return ChatTemplateType::Gemma3;
  }
  if (template_str.find("<|start_header_id|>") != std::string::npos) {
    return ChatTemplateType::Llama3;
  }
  return ChatTemplateType::Custom;
}

std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

} // namespace

} // namespace executorch::extension::llm

namespace jinja2 {

// NOLINTBEGIN(facebook-hte-MisplacedTemplateSpecialization,facebook-hte-ShadowingClass)
// This template specialization must be in the jinja2 namespace for the library
// to find it via ADL during template instantiation.
template <>
struct TypeReflection<executorch::extension::llm::ChatMessage>
    : TypeReflected<executorch::extension::llm::ChatMessage> {
  static auto& GetAccessors() {
    static std::unordered_map<std::string, FieldAccessor> accessors = {
        {"role",
         [](const executorch::extension::llm::ChatMessage& msg) {
           return jinja2::Reflect(msg.role);
         }},
        {"content",
         [](const executorch::extension::llm::ChatMessage& msg) {
           return jinja2::Reflect(msg.content);
         }},
    };
    return accessors;
  }
};
// NOLINTEND(facebook-hte-MisplacedTemplateSpecialization,facebook-hte-ShadowingClass)

} // namespace jinja2

namespace executorch::extension::llm {

JinjaChatFormatter::JinjaChatFormatter(
    const std::string& template_str,
    ChatTemplateType type)
    : template_str_(template_str), type_(type) {
  auto tokens_it = kModelTokens.find(type_);
  if (tokens_it != kModelTokens.end()) {
    model_tokens_ = tokens_it->second;
  }
  includes_bos_ = templateIncludesBos(template_str_, model_tokens_);
  const std::string normalized_template = normalizeTemplate(template_str_);
  compiled_template_ = std::make_unique<jinja2::Template>();
  auto load_result = compiled_template_->Load(normalized_template);
  if (!load_result) {
    throw std::runtime_error(
        "Failed to parse chat template: " +
        load_result.error().ToString());
  }
}

JinjaChatFormatter::~JinjaChatFormatter() = default;

std::unique_ptr<JinjaChatFormatter> JinjaChatFormatter::fromTemplate(
    ChatTemplateType type) {
  auto it = kEmbeddedTemplates.find(type);
  if (it == kEmbeddedTemplates.end()) {
    throw std::runtime_error("Unsupported embedded chat template type.");
  }
  return std::unique_ptr<JinjaChatFormatter>(
      new JinjaChatFormatter(std::string(it->second), type));
}

std::unique_ptr<JinjaChatFormatter> JinjaChatFormatter::fromString(
    const std::string& template_str) {
  const ChatTemplateType inferred_type = detectTemplateType(template_str);
  return std::unique_ptr<JinjaChatFormatter>(
      new JinjaChatFormatter(template_str, inferred_type));
}

std::unique_ptr<JinjaChatFormatter> JinjaChatFormatter::fromFile(
    const std::filesystem::path& path) {
  return fromString(readFileToString(path));
}

std::string JinjaChatFormatter::format(
    const std::string& prompt,
    const std::string& system_prompt) const {
  ChatConversation conversation;
  if (!system_prompt.empty()) {
    conversation.messages.push_back({"system", system_prompt});
  }
  conversation.messages.push_back({"user", prompt});
  conversation.bos_token = model_tokens_.bos_token;
  conversation.eos_token = model_tokens_.eos_token;
  conversation.add_generation_prompt = true;
  return formatConversation(conversation);
}

std::string JinjaChatFormatter::formatConversation(
    const ChatConversation& conversation) const {
  jinja2::ValuesMap params;
  params["messages"] = jinja2::ValuesList();
  params["messages_tail"] = jinja2::ValuesList();
  bool is_first = true;
  for (const auto& msg : conversation.messages) {
    params["messages"].asList().push_back(jinja2::Reflect(msg));
    if (!is_first) {
      params["messages_tail"].asList().push_back(jinja2::Reflect(msg));
    }
    is_first = false;
  }
  params["bos_token"] = conversation.bos_token;
  params["eos_token"] = conversation.eos_token;
  params["add_generation_prompt"] = conversation.add_generation_prompt;

  auto rendered = compiled_template_->RenderAsString(params);
  if (!rendered) {
    throw std::runtime_error(
        "Failed to render chat template: " + rendered.error().ToString());
  }
  return rendered.value();
}

ChatTemplateType parseChatTemplateType(const std::string& type_str) {
  const std::string lower = toLower(type_str);
  if (lower == "none") {
    return ChatTemplateType::None;
  }
  if (lower == "llama3") {
    return ChatTemplateType::Llama3;
  }
  if (lower == "llama3.2" || lower == "llama32" || lower == "llama3_2") {
    return ChatTemplateType::Llama32;
  }
  if (lower == "gemma3") {
    return ChatTemplateType::Gemma3;
  }
  if (lower == "custom" || lower == "jinja") {
    return ChatTemplateType::Custom;
  }
  return ChatTemplateType::None;
}

} // namespace executorch::extension::llm
