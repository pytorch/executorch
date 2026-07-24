/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/jinja_chat_formatter.h>

#include <jinja2cpp/generic_list.h>
#include <jinja2cpp/generic_list_iterator.h>
#include <jinja2cpp/reflected_value.h>
#include <jinja2cpp/template.h>
#include <jinja2cpp/user_callable.h>
#include <jinja2cpp/value.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace executorch::extension::llm {
namespace {

std::string readFileToString(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open template file: " + path);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  std::string contents = buffer.str();
  if (contents.empty()) {
    throw std::runtime_error("Template file is empty: " + path);
  }
  return contents;
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
  // These replacements normalize vLLM/HuggingFace Jinja templates so they
  // compile/render correctly with Jinja2Cpp, which has stricter parser
  // semantics than Python Jinja2.
  //
  // IMPORTANT: "not tools is none" in Python Jinja means "tools is not none"
  // (truthy when tools is defined and non-null), so we map it to a simple
  // truthy check on `tools`. Mapping to "not tools" was a bug that would
  // skip tool blocks for non-empty tools lists.
  // Keep longer `tools is ... none` patterns before the shorter
  // `tools is none` patterns to avoid partial replacements.
  //
  // `messages[1:]` is rewritten to a render-time call so each slice operates
  // on the current value of `messages`. Templates reassign `messages`
  // (e.g. `{%- set messages = messages[1:] %}`) and slice it again later, so a
  // precomputed tail of the original list would be wrong for the second slice.
  constexpr std::array<std::pair<std::string_view, std::string_view>, 10>
      replacements = {{
          {"tools = none", "tools = []"},
          {"tools = None", "tools = []"},
          {"tools is not none", "tools"},
          {"tools is not None", "tools"},
          {"not tools is none", "tools"},
          {"not tools is None", "tools"},
          {"tools is none", "not tools"},
          {"tools is None", "not tools"},
          {"messages[1:]", "_executorch_messages_tail(messages)"},
          {"{ \"output\": message.content } | tojson",
           "message.tool_output | tojson"},
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
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return value;
}

// Materializes a GenericList into an owning ValuesList WITHOUT going through
// jinja2::detail::GenericListIterator. Infer's USE_AFTER_LIFETIME flags that
// iterator's begin()/operator++ (its m_current caches the address of a
// temporary Value), so we read the list via its accessor instead: random-access
// lists by index, others by single-pass enumeration. GetItemByIndex() and
// GetCurrent() both return owning Values by value.
jinja2::ValuesList collectGenericList(const jinja2::GenericList& list) {
  jinja2::ValuesList items;
  const auto* accessor = list.GetAccessor();
  if (accessor == nullptr) {
    return items;
  }
  const auto* indexer = accessor->GetIndexer();
  const auto size = list.GetSize();
  if (indexer != nullptr && size) {
    items.reserve(*size);
    for (size_t i = 0; i < *size; ++i) {
      items.push_back(indexer->GetItemByIndex(static_cast<int64_t>(i)));
    }
    return items;
  }
  auto enumerator = accessor->CreateEnumerator();
  while (enumerator && enumerator->MoveNext()) {
    items.push_back(enumerator->GetCurrent());
  }
  return items;
}

// Copies a value into render-lifetime storage so it stays valid after the
// caller's list is freed. Jinja2Cpp represents strings as non-owning
// string_views, and reassigning a template variable (e.g.
// `set messages = messages[1:]`) frees the list that backs them. We copy every
// string's bytes into `string_store` (whose std::list nodes keep stable
// addresses for the whole render) and hand back string_views into that store;
// maps/lists are rebuilt as owning containers whose leaves are store-backed.
jinja2::Value toStableValue(
    const jinja2::Value& value,
    std::list<std::string>& string_store) {
  if (const auto* str = value.getPtr<std::string>()) {
    string_store.push_back(*str);
    return jinja2::Value(nonstd::string_view(string_store.back()));
  }
  if (const auto* view = value.getPtr<nonstd::string_view>()) {
    string_store.emplace_back(view->begin(), view->end());
    return jinja2::Value(nonstd::string_view(string_store.back()));
  }
  // getPtr<GenericMap> is checked before isMap()/asMap() so asMap() is only
  // used on an owning (non-Generic) map.
  if (const auto* generic_map = value.getPtr<jinja2::GenericMap>()) {
    jinja2::ValuesMap owned;
    for (const auto& key : generic_map->GetKeys()) {
      owned[key] =
          toStableValue(generic_map->GetValueByName(key), string_store);
    }
    return jinja2::Value(std::move(owned));
  }
  if (value.isMap()) {
    jinja2::ValuesMap owned;
    for (const auto& entry : value.asMap()) {
      owned[entry.first] = toStableValue(entry.second, string_store);
    }
    return jinja2::Value(std::move(owned));
  }
  if (const auto* generic_list = value.getPtr<jinja2::GenericList>()) {
    jinja2::ValuesList owned;
    for (const auto& item : collectGenericList(*generic_list)) {
      owned.push_back(toStableValue(item, string_store));
    }
    return jinja2::Value(std::move(owned));
  }
  if (value.isList()) {
    jinja2::ValuesList owned;
    for (const auto& item : value.asList()) {
      owned.push_back(toStableValue(item, string_store));
    }
    return jinja2::Value(std::move(owned));
  }
  // Scalars (bool / int64 / double / empty) are already self-owning.
  return value;
}

// Render-time equivalent of the Python slice `messages[1:]`: returns all
// elements after the first. Used because Jinja2Cpp cannot parse slice syntax
// and the slice must reflect the current value of `messages` at call time.
// Jinja2Cpp may pass the list as either a GenericList or a ValuesList, so both
// representations are handled without using the throwing asList() accessor on
// the wrong variant. Each element is materialized into render-lifetime storage
// (see toStableValue) so the slice stays valid after `messages` is reassigned.
jinja2::ValuesList messagesTail(
    const jinja2::Value& messages,
    std::list<std::string>& string_store) {
  jinja2::ValuesList tail;
  if (const auto* generic = messages.getPtr<jinja2::GenericList>()) {
    const jinja2::ValuesList items = collectGenericList(*generic);
    for (size_t i = 1; i < items.size(); ++i) {
      tail.push_back(toStableValue(items[i], string_store));
    }
  } else if (messages.isList()) {
    const auto& values = messages.asList();
    for (size_t i = 1; i < values.size(); ++i) {
      tail.push_back(toStableValue(values[i], string_store));
    }
  }
  return tail;
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
        {"tool_output",
         [](const executorch::extension::llm::ChatMessage& msg) {
           jinja2::ValuesMap output;
           output["output"] = msg.content;
           return output;
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
  const auto& model_tokens = ::executorch::extension::llm::getModelTokens();
  auto tokens_it = model_tokens.find(type_);
  if (tokens_it != model_tokens.end()) {
    model_tokens_ = tokens_it->second;
  }
  includes_bos_ = templateIncludesBos(template_str_, model_tokens_);
  const std::string normalized_template = normalizeTemplate(template_str_);
  compiled_template_ = std::make_unique<jinja2::Template>();
  auto load_result = compiled_template_->Load(normalized_template);
  if (!load_result) {
    throw std::runtime_error(
        "Failed to parse chat template: " + load_result.error().ToString());
  }
}

JinjaChatFormatter::~JinjaChatFormatter() = default;

std::unique_ptr<JinjaChatFormatter> JinjaChatFormatter::fromTemplate(
    ChatTemplateType type) {
  const auto& embedded_templates = getEmbeddedTemplates();
  auto it = embedded_templates.find(type);
  if (it == embedded_templates.end()) {
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
    const std::string& path) {
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
  // Owning, render-lifetime storage for strings produced while slicing
  // `messages`. Jinja2Cpp keeps non-owning string_views and frees the list that
  // backs them when a template reassigns `messages`; keeping the bytes here
  // (stable std::list nodes that outlive RenderAsString) keeps those views
  // valid. shared_ptr lets the callable hold a reference even if Jinja2Cpp
  // copies it internally.
  auto message_string_store = std::make_shared<std::list<std::string>>();
  params["messages"] = jinja2::ValuesList();
  for (const auto& msg : conversation.messages) {
    params["messages"].asList().push_back(jinja2::Reflect(msg));
  }
  // Backs the `messages[1:]` normalization so each slice reflects the current
  // `messages` value rather than a precomputed snapshot.
  //
  // We register this via the raw jinja2::UserCallable struct rather than
  // jinja2::MakeCallable because MakeCallable cannot wrap a function whose
  // parameter is `const jinja2::Value&`: its argument-promotion path produces
  // an ArgPromoter<T> that is not convertible to jinja2::Value, so the wrapper
  // fails to instantiate. The UserCallable form hands us the raw Value via
  // UserCallableParams::operator[], which is exactly what messagesTail needs.
  jinja2::UserCallable messages_tail_callable;
  messages_tail_callable.callable =
      [message_string_store](
          const jinja2::UserCallableParams& callable_params) -> jinja2::Value {
    return jinja2::Value(
        messagesTail(callable_params["messages"], *message_string_store));
  };
  messages_tail_callable.argsInfo = {jinja2::ArgInfo{"messages", true}};
  params["_executorch_messages_tail"] = std::move(messages_tail_callable);
  params["bos_token"] = conversation.bos_token;
  params["eos_token"] = conversation.eos_token;
  params["add_generation_prompt"] = conversation.add_generation_prompt;
  // Provide vLLM/HuggingFace-style defaults that templates often reference.
  // Templates that don't use these will simply ignore them.
  // `tools` is intentionally an empty list (not null): Jinja2Cpp, like Python,
  // treats an empty list as falsy, so `{% if tools %}` and the normalized
  // `tools is not none` -> `tools` checks render the "no tools" path when none
  // are supplied. See normalizeTemplate() and the UniversalJinjaToolsAware
  // test.
  params["tools"] = jinja2::ValuesList();
  params["tool_choice"] = jinja2::Value();
  params["date_string"] = conversation.date_string;
  params["chat_template_kwargs"] = jinja2::ValuesMap();

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
