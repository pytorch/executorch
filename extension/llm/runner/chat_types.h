/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
  // Injected as the `date_string` template variable. Defaults to the fixed
  // date used by HuggingFace/vLLM template fixtures so output stays
  // deterministic; callers may override it.
  std::string date_string = "26 Jul 2024";
};

} // namespace executorch::extension::llm
