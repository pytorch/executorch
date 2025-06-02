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
#include <type_traits>
#include <unordered_map>
#include <vector>

// Local
#include <pytorch/tokenizers/error.h>
#include <pytorch/tokenizers/regex.h>
#include <pytorch/tokenizers/result.h>
#include <pytorch/tokenizers/string_integer_map.h>
#include <pytorch/tokenizers/tokenizer.h>

#include "re2/re2.h"

namespace tokenizers {
namespace detail {

using TokenMap = StringIntegerMap<>;

template <typename TToken, typename TRank>
static Result<TokenMap> buildTokenMap(
    std::vector<std::pair<TToken, TRank>> container) {
  static_assert(
      std::is_same_v<TToken, std::string> ||
          std::is_same_v<TToken, std::string_view>,
      "TToken must be std::string or std::string_view");
  static_assert(
      std::is_integral_v<TRank> && std::is_unsigned_v<TRank>,
      "TRank must be an unsigned integer");

  std::sort(
      container.begin(), container.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
      });

  auto duplicate_begin = std::unique(
      container.begin(), container.end(), [](const auto& a, const auto& b) {
        return a.first == b.first;
      });

  TK_CHECK_OR_RETURN_ERROR(
      duplicate_begin == container.end(),
      ParseFailure,
      "duplicate token: %s rank: %llu",
      duplicate_begin->first.c_str(),
      static_cast<unsigned long long>(duplicate_begin->second));

  std::sort(
      container.begin(), container.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
      });

  duplicate_begin = std::unique(
      container.begin(), container.end(), [](const auto& a, const auto& b) {
        return a.second == b.second;
      });

  TK_CHECK_OR_RETURN_ERROR(
      duplicate_begin == container.end(),
      ParseFailure,
      "duplicate rank: %llu"
      " token: %s",
      static_cast<unsigned long long>(duplicate_begin->second),
      duplicate_begin->first.c_str());

  return TokenMap(container);
};

template <typename TContainer, typename TTokenAccessor, typename TRankAccessor>
static Result<TokenMap> buildTokenMap(
    const TContainer& container,
    TTokenAccessor token_accessor,
    TRankAccessor rank_accessor) {
  using TokenType = std::invoke_result_t<TTokenAccessor, const TContainer&>;
  using RankType = std::invoke_result_t<TRankAccessor, const TContainer&>;

  static_assert(
      std::is_same_v<TokenType, std::string> ||
          std::is_same_v<TokenType, std::string_view>,
      "TokenType must be std::string or std::string_view");
  static_assert(
      std::is_integral_v<RankType> && std::is_unsigned_v<RankType>,
      "RankType must be an unsigned integer");

  std::vector<std::pair<TokenType, RankType>> pairs;
  pairs.reserve(container.size());
  for (const auto& value : container) {
    pairs.emplace_back(token_accessor(value), rank_accessor(value));
  }

  return buildTokenMap(std::move(pairs));
}

inline Result<std::unique_ptr<IRegex>> build_special_token_regex(
    const TokenMap& special_token_map) {
  std::string special_pattern;
  const std::size_t count = special_token_map.size();

  for (std::size_t i = 0; i < count; ++i) {
    const auto& [token, _] = special_token_map.getElement(i);
    if (!special_pattern.empty()) {
      special_pattern += "|";
    }
    special_pattern += re2::RE2::QuoteMeta(std::string(token));
  }

  if (special_pattern.empty()) {
    return static_cast<std::unique_ptr<IRegex>>(nullptr);
  }
  return create_regex(special_pattern);
}

class BPETokenizerBase : public Tokenizer {
 public:
  Result<std::vector<uint64_t>>
  encode(const std::string& input, int8_t bos, int8_t eos) const override;

  Result<std::string> decode(uint64_t prev_token, uint64_t token)
      const override;

 protected:
  explicit BPETokenizerBase() {}
  virtual ~BPETokenizerBase() override {}

  std::pair<std::optional<std::string>, std::string>
  split_with_allowed_special_token_(
      const std::string& input,
      const TokenMap& allowed_special) const;

  std::pair<std::optional<std::string>, std::string>
  split_with_allowed_special_token_(
      const std::string& input,
      size_t offset,
      const TokenMap& allowed_special) const;

  Result<std::pair<std::vector<uint64_t>, uint64_t>> encode_with_special_token_(
      const std::string& text,
      const TokenMap& allowed_special) const;

  Result<std::vector<uint64_t>> byte_pair_encode_(
      const std::string& piece,
      const TokenMap& encoder) const;

  // Protected members that can be overloaded by other BPE tokenizers
  std::unique_ptr<IRegex> special_token_regex_;
  std::optional<TokenMap> token_map_;
  std::optional<TokenMap> special_token_map_;

 private:
  virtual Error _encode(
      const std::string& input,
      std::vector<uint64_t>& ret,
      uint64_t& last_piece_token_len) const = 0;

  virtual void _decode(const std::string& input, std::string& ret) const = 0;
};

} // namespace detail
} // namespace tokenizers
