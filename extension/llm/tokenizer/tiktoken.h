/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/tokenizer/tokenizer.h>
#include <re2/re2.h>
#include <memory>
#include <optional>
#include <unordered_map>

namespace executorch {
namespace extension {
namespace llm {

using Encoder = std::unordered_map<std::string, uint64_t>;
using Decoder = std::unordered_map<uint64_t, std::string>;
using Re2UPtr = std::unique_ptr<re2::RE2>;

class ET_EXPERIMENTAL Tiktoken : public Tokenizer {
 public:
  /**
   * @param[in] special_tokens List of special tokens including bos, eos;
   * @param[in] bos_token_index Index of the bos token in special_tokens;
   * @param[in] eos_token_index Index of the eos token in special_tokens.
   */
  explicit Tiktoken(
      std::unique_ptr<std::vector<std::string>> special_tokens,
      size_t bos_token_index,
      size_t eos_token_index);

  ::executorch::runtime::Error load(const std::string& tokenizer_path) override;

  ::executorch::runtime::Result<std::vector<uint64_t>>
  encode(const std::string& input, int8_t bos, int8_t eos) const override;

  ::executorch::runtime::Result<std::string> decode(
      uint64_t prev_token,
      uint64_t token) const override;

 private:
  template <typename T>
  std::pair<std::optional<std::string>, re2::StringPiece>
  _split_with_allowed_special_token(
      re2::StringPiece& input,
      const T& allowed_special) const;

  void _encode(
      re2::StringPiece& input,
      std::vector<uint64_t>& ret,
      uint64_t& last_piece_token_len) const;

  template <typename T>
  std::pair<std::vector<uint64_t>, uint64_t> _encode_with_special_token(
      const std::string& text,
      const T& allowed_special) const;

  Encoder _build_special_token_encoder(ssize_t num_base_tokens) const;

  std::unique_ptr<std::vector<std::string>> _special_tokens;
  size_t _bos_token_index;
  size_t _eos_token_index;
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

} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::Decoder;
using ::executorch::extension::llm::Encoder;
using ::executorch::extension::llm::Re2UPtr;
using ::executorch::extension::llm::Tiktoken;
} // namespace executor
} // namespace torch
