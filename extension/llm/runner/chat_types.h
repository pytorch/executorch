#pragma once

#include <string>
#include <vector>

namespace executorch::extension::llm {

struct ChatMessage {
  std::string role;
  std::string content;
};

struct ChatConversation {
  std::vector<ChatMessage> messages;
  std::string bos_token;
  std::string eos_token;
  bool add_generation_prompt = true;
};

} // namespace executorch::extension::llm
