/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/llama2/tokenizer/llama_tiktoken.h>

namespace torch {
namespace executor {
namespace {
static constexpr int32_t kSpecialTokensSize = 256;

static inline const Encoder _get_default_special_tokens(
    ssize_t num_base_tokens) {
  Encoder special_tokens;
  ssize_t special_token_count = 0;
  special_tokens.emplace(
      "<|begin_of_text|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|end_of_text|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|reserved_special_token_0|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|reserved_special_token_1|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|reserved_special_token_2|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|reserved_special_token_3|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|start_header_id|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|end_header_id|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|reserved_special_token_4|>", num_base_tokens + special_token_count++);
  special_tokens.emplace("<|eot_id|>", num_base_tokens + special_token_count++);

  // pad the rest of the special tokens with reserved tokens
  ssize_t reserved_special_token_num = 5;
  while (special_token_count < kSpecialTokensSize) {
    special_tokens.emplace(
        "<|reserved_special_token_" +
            std::to_string(reserved_special_token_num++) + "|>",
        num_base_tokens + special_token_count++);
  }
  return special_tokens;
}

static inline const Encoder _get_multimodal_special_tokens(
    ssize_t num_base_tokens) {
  ssize_t special_token_count = 0;
  Encoder special_tokens;
  special_tokens.emplace(
      "<|begin_of_text|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|end_of_text|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|reserved_special_token_0|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|reserved_special_token_1|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|reserved_special_token_2|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|reserved_special_token_3|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|start_header_id|>", num_base_tokens + special_token_count++);
  special_tokens.emplace(
      "<|end_header_id|>", num_base_tokens + special_token_count++);
  special_tokens.emplace("<|eom_id|>", num_base_tokens + special_token_count++);
  special_tokens.emplace("<|eot_id|>", num_base_tokens + special_token_count++);
  special_tokens.emplace("<|image|>", num_base_tokens + special_token_count++);

  // pad the rest of the special tokens with reserved tokens except the last
  // one
  ssize_t reserved_special_token_num = 4;
  while (special_token_count < kSpecialTokensSize - 1) {
    special_tokens.emplace(
        "<|reserved_special_token_" +
            std::to_string(reserved_special_token_num++) + "|>",
        num_base_tokens + special_token_count++);
  }

  special_tokens.emplace(
      "<|python_tag|>", num_base_tokens + special_token_count++);

  return special_tokens;
}
} // namespace

const Encoder LlamaTiktoken::get_special_tokens(ssize_t num_base_tokens) const {
  switch (_version) {
    case MULTIMODAL:
      return _get_multimodal_special_tokens(num_base_tokens);
    default:
      return _get_default_special_tokens(num_base_tokens);
  }
}
} // namespace executor
} // namespace torch
