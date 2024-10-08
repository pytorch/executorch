/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/llama2/tokenizer/llama_tiktoken.h>

namespace example {

using ::executorch::extension::llm::Tiktoken;

namespace {
static constexpr int32_t kSpecialTokensSize = 256;
static constexpr size_t kBOSTokenIndex = 0;
static constexpr size_t kEOSTokenIndex = 1;

static inline std::unique_ptr<std::vector<std::string>>
_get_default_special_tokens() {
  auto special_tokens =
      std::make_unique<std::vector<std::string>>(std::vector<std::string>{
          "<|begin_of_text|>",
          "<|end_of_text|>",
          "<|reserved_special_token_0|>",
          "<|reserved_special_token_1|>",
          "<|finetune_right_pad_id|>",
          "<|step_id|>",
          "<|start_header_id|>",
          "<|end_header_id|>",
          "<|eom_id|>",
          "<|eot_id|>",
          "<|python_tag|>"});
  // pad the rest of the special tokens with reserved tokens
  ssize_t reserved_special_token_num = 2;
  while (special_tokens->size() < kSpecialTokensSize) {
    special_tokens->emplace_back(
        "<|reserved_special_token_" +
        std::to_string(reserved_special_token_num++) + "|>");
  }
  return special_tokens;
}

static inline std::unique_ptr<std::vector<std::string>>
_get_multimodal_special_tokens() {
  auto special_tokens =
      std::make_unique<std::vector<std::string>>(std::vector<std::string>{
          "<|begin_of_text|>",
          "<|end_of_text|>",
          "<|reserved_special_token_0|>",
          "<|reserved_special_token_1|>",
          "<|reserved_special_token_2|>",
          "<|reserved_special_token_3|>",
          "<|start_header_id|>",
          "<|end_header_id|>",
          "<|eom_id|>",
          "<|eot_id|>",
          "<|image|>"});

  // pad the rest of the special tokens with reserved tokens except the last
  // one
  ssize_t reserved_special_token_num = 4;
  while (special_tokens->size() < kSpecialTokensSize - 1) {
    special_tokens->emplace_back(
        "<|reserved_special_token_" +
        std::to_string(reserved_special_token_num++) + "|>");
  }

  special_tokens->emplace_back("<|python_tag|>");

  return special_tokens;
}

std::unique_ptr<std::vector<std::string>> _get_special_tokens(Version version) {
  switch (version) {
    case Version::Multimodal:
      return _get_multimodal_special_tokens();
    default:
      return _get_default_special_tokens();
  }
}

} // namespace

std::unique_ptr<Tiktoken> get_tiktoken_for_llama(Version version) {
  return std::make_unique<Tiktoken>(
      _get_special_tokens(version), kBOSTokenIndex, kEOSTokenIndex);
}

} // namespace example
