#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace executorch::extension::llm {

enum class ChatTemplateType {
  None,
  Llama3,
  Llama32,
  Gemma3,
  Custom,
};

constexpr std::string_view kLlama3Template = R"({{ bos_token }}{%- for message in messages -%}<|start_header_id|>{{ message.role }}<|end_header_id|>

{{ message.content }}<|eot_id|>{%- endfor -%}{%- if add_generation_prompt -%}<|start_header_id|>assistant<|end_header_id|>

{%- endif -%})";

constexpr std::string_view kGemma3Template = R"({{ bos_token }}{%- for message in messages -%}{%- if message.role == 'assistant' -%}<start_of_turn>model
{%- else -%}<start_of_turn>{{ message.role }}
{%- endif -%}{{ message.content }}<end_of_turn>{%- endfor -%}{%- if add_generation_prompt -%}<start_of_turn>model
{%- endif -%})";

inline const std::unordered_map<ChatTemplateType, std::string_view>
    kEmbeddedTemplates = {
        {ChatTemplateType::Llama3, kLlama3Template},
        {ChatTemplateType::Llama32, kLlama3Template},
        {ChatTemplateType::Gemma3, kGemma3Template},
    };

struct ModelTokens {
  std::string bos_token;
  std::string eos_token;
  std::vector<std::string> stop_tokens;
};

inline const std::unordered_map<ChatTemplateType, ModelTokens> kModelTokens = {
    {ChatTemplateType::Llama3,
     {"<|begin_of_text|>", "<|eot_id|>", {"<|eot_id|>", "<|end_of_text|>"}}},
    {ChatTemplateType::Llama32,
     {"<|begin_of_text|>", "<|eot_id|>", {"<|eot_id|>", "<|end_of_text|>"}}},
    {ChatTemplateType::Gemma3,
     {"<bos>", "<end_of_turn>", {"<end_of_turn>", "<eos>"}}},
};

} // namespace executorch::extension::llm
