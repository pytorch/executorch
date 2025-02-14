/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tiktoken header
// Used by OpenAI, adapted from https://github.com/sewenew/tokenizer
#pragma once

// Standard
#include <cstdint>

// Third Party
#include "re2/re2.h"

// Local
#include <pytorch/tokenizers/bpe_tokenizer_base.h>
#include <pytorch/tokenizers/result.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace tokenizers {

static constexpr int32_t kSpecialTokensSize = 256;
static constexpr size_t kBOSTokenIndex = 0;
static constexpr size_t kEOSTokenIndex = 1;

class Tiktoken : public detail::BPETokenizerBase {
 public:
  explicit Tiktoken(
      std::unique_ptr<std::vector<std::string>> special_tokens,
      size_t bos_token_index,
      size_t eos_token_index)
      : _special_tokens(std::move(special_tokens)),
        _bos_token_index(bos_token_index),
        _eos_token_index(eos_token_index) {
    assert(_bos_token_index < _special_tokens->size());
    assert(_eos_token_index < _special_tokens->size());
  };

  explicit Tiktoken()
      : _special_tokens(_get_default_special_tokens()),
        _bos_token_index(kBOSTokenIndex),
        _eos_token_index(kEOSTokenIndex){};

  Error load(const std::string& tokenizer_path) override;

 private:
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

  template <typename T>
  std::pair<std::optional<std::string>, re2::StringPiece>
  _split_with_allowed_special_token(
      re2::StringPiece& input,
      const T& allowed_special) const;

  Error _encode(
      re2::StringPiece& input,
      std::vector<uint64_t>& ret,
      uint64_t& last_piece_token_len) const override;

  void _decode(re2::StringPiece input, std::string& ret) const override;

  template <typename T>
  Result<std::pair<std::vector<uint64_t>, uint64_t>> _encode_with_special_token(
      const std::string& text,
      const T& allowed_special) const;

  detail::Encoder _build_special_token_encoder(ssize_t num_base_tokens) const;

  std::unique_ptr<std::vector<std::string>> _special_tokens;
  size_t _bos_token_index;
  size_t _eos_token_index;

  // Removed negative lookahead \s+(?!\S) since it's not supported by RE2.
  const std::string _pattern =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)";

  detail::Re2UPtr _regex;
};

} // namespace tokenizers
