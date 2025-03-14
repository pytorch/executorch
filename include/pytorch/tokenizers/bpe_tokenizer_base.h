/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

// Base class for all BPE tokenizer implementations
#pragma once

// Standard
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// Third Party
#include <re2/re2.h>

// Local
#include <pytorch/tokenizers/result.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace tokenizers {
namespace detail {

using Encoder = std::unordered_map<std::string, uint64_t>;
using Decoder = std::unordered_map<uint64_t, std::string>;
using Re2UPtr = std::unique_ptr<re2::RE2>;

class BPETokenizerBase : public Tokenizer {
 public:
  Result<std::vector<uint64_t>>
  encode(const std::string& input, int8_t bos, int8_t eos) const override;

  Result<std::string> decode(uint64_t prev_token, uint64_t token)
      const override;

 protected:
  explicit BPETokenizerBase() {}
  virtual ~BPETokenizerBase() {}

  std::pair<std::optional<std::string>, re2::StringPiece>
  split_with_allowed_special_token_(
      re2::StringPiece& input,
      const Encoder& allowed_special) const;

  Result<std::pair<std::vector<uint64_t>, uint64_t>> encode_with_special_token_(
      const std::string& text,
      const Encoder& allowed_special) const;

  Result<std::vector<uint64_t>> byte_pair_encode_(
      const std::string& piece,
      const Encoder& encoder) const;

  // Protected members that can be overloaded by other BPE tokenizers
  Re2UPtr special_token_regex_;
  Encoder encoder_;
  Encoder special_token_encoder_;
  Decoder decoder_;
  Decoder special_token_decoder_;

 private:
  virtual Error _encode(
      re2::StringPiece& input,
      std::vector<uint64_t>& ret,
      uint64_t& last_piece_token_len) const = 0;

  virtual void _decode(re2::StringPiece input, std::string& ret) const = 0;
};

} // namespace detail
} // namespace tokenizers
