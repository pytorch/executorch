/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/models/llama2/tokenizer/base64.h>
#include <executorch/examples/models/llama2/tokenizer/tokenizer.h>
#include <re2/re2.h>
#include <cstdint>
#include <functional>
#include <optional>
#include <regex>
#include <unordered_map>

namespace torch {
namespace executor {

using Encoder = std::unordered_map<std::string, uint64_t>;
using Decoder = std::unordered_map<uint64_t, std::string>;
using Re2UPtr = std::unique_ptr<re2::RE2>;

class Tiktoken : public Tokenizer {
 public:
  explicit Tiktoken(int32_t vocab_size, uint64_t bos_tok, uint64_t eos_tok)
      : Tokenizer(vocab_size, bos_tok, eos_tok){};
  ~Tiktoken(){};

  Error load(const std::string& tokenizer_path);

  Result<std::vector<uint64_t>>
  encode(const std::string& input, int8_t bos, int8_t eos);

  Result<std::string> decode(uint64_t prev_token, uint64_t token);

 private:
  static inline const Encoder _get_special_tokens(ssize_t num_base_tokens) {
    Encoder special_tokens;
    special_tokens.emplace("<|begin_of_text|>", num_base_tokens++);
    special_tokens.emplace("<|end_of_text|>", num_base_tokens++);
    special_tokens.emplace("<|reserved_special_token_0|>", num_base_tokens++);
    special_tokens.emplace("<|reserved_special_token_1|>", num_base_tokens++);
    special_tokens.emplace("<|reserved_special_token_2|>", num_base_tokens++);
    special_tokens.emplace("<|reserved_special_token_3|>", num_base_tokens++);
    special_tokens.emplace("<|start_header_id|>", num_base_tokens++);
    special_tokens.emplace("<|end_header_id|>", num_base_tokens++);
    special_tokens.emplace("<|reserved_special_token_4|>", num_base_tokens++);
    special_tokens.emplace("<|eot_id|>", num_base_tokens++);
    for (auto i = 5; i < 251; ++i) {
      special_tokens.emplace(
          "<|reserved_special_token_" + std::to_string(i) + "|>",
          num_base_tokens++);
    }
    return special_tokens;
  }

  template <typename T>
  std::pair<std::optional<std::string>, re2::StringPiece>
  _split_with_allowed_special_token(
      re2::StringPiece& input,
      const T& allowed_special);

  void _encode(
      re2::StringPiece& input,
      std::vector<uint64_t>& ret,
      uint64_t& last_piece_token_len);

  template <typename T>
  std::pair<std::vector<uint64_t>, uint64_t> _encode_with_special_token(
      const std::string& text,
      const T& allowed_special);

  // Removed negative lookahead \s+(?!\S) since it's not supported by RE2.
  const std::string _pattern =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)";
  Encoder _encoder;
  Encoder _special_token_encoder;
  Decoder _decoder;
  Decoder _special_token_decoder;

  Re2UPtr _regex;
  Re2UPtr _special_token_regex;
};
} // namespace executor
} // namespace torch
