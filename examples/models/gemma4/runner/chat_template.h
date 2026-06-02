/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Shared Gemma 4 chat-template special-token IDs and the image+text input
// builder. Single source of truth for the multimodal chat-template token
// layout, mirrored by the Python module
// examples/models/gemma4/chat_template.py so the two implementations can never
// drift.
//
// Image+text turn layout (matches the Gemma 4 HF chat template):
//
//   <bos><start_of_turn>user\n<boi><image>*N<eoi>{prompt}<end_of_turn>\n
//   <start_of_turn>model\n

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace executorch {
namespace examples {
namespace gemma4 {

// Gemma 4 special token IDs (match the tokenizer + the Python chat_template).
static constexpr int64_t kBosId = 2;
static constexpr int64_t kTurnStartId = 105; // <start_of_turn>
static constexpr int64_t kTurnEndId = 106; // <end_of_turn>
static constexpr int64_t kBoiTokenId = 255999; // <start_of_image>
static constexpr int64_t kImageTokenId =
    258880; // <image> soft-token placeholder
static constexpr int64_t kEoiTokenId = 258882; // <end_of_image>

// Build the chat-template token sequence for an image+text turn.
//
// Templated on the tokenizer type so it works for any tokenizer exposing
// ``encode(text, add_bos, add_eos) -> Result<std::vector<uint64_t>>`` (the
// pytorch/tokenizers interface used by both the gemma4_31b runner's
// HFTokenizer and the shared Gemma4Runner's tokenizer).
template <typename Tokenizer>
inline std::vector<int64_t> build_vision_input_ids(
    Tokenizer* tokenizer,
    const std::string& prompt,
    int64_t num_vision_tokens,
    int64_t bos_id = kBosId) {
  auto encode = [&](const std::string& s) -> std::vector<uint64_t> {
    auto r = tokenizer->encode(s, /*add_bos=*/0, /*add_eos=*/0);
    if (!r.ok()) {
      return {};
    }
    return std::move(*r);
  };

  auto user_tokens = encode("user\n");
  auto prompt_tokens = encode(prompt);
  auto newline_tokens = encode("\n");
  auto model_tokens = encode("model\n");

  std::vector<int64_t> ids;
  ids.push_back(bos_id);
  ids.push_back(kTurnStartId);
  for (auto t : user_tokens) {
    ids.push_back(static_cast<int64_t>(t));
  }
  ids.push_back(kBoiTokenId);
  for (int64_t i = 0; i < num_vision_tokens; ++i) {
    ids.push_back(kImageTokenId);
  }
  ids.push_back(kEoiTokenId);
  for (auto t : prompt_tokens) {
    ids.push_back(static_cast<int64_t>(t));
  }
  ids.push_back(kTurnEndId);
  for (auto t : newline_tokens) {
    ids.push_back(static_cast<int64_t>(t));
  }
  ids.push_back(kTurnStartId);
  for (auto t : model_tokens) {
    ids.push_back(static_cast<int64_t>(t));
  }
  return ids;
}

} // namespace gemma4
} // namespace examples
} // namespace executorch
