/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/bpe_tokenizer_base.h>

// Standard
#include <inttypes.h>

namespace tokenizers {
namespace detail {

// ---- Helper utils start -----------------------------------------------------
namespace {

static uint64_t _max_size() { return std::numeric_limits<uint64_t>::max(); }

static std::vector<uint64_t>
_byte_pair_merge(const std::string &piece,
                 const std::unordered_map<std::string, uint64_t> &ranks,
                 std::function<uint64_t(uint64_t, uint64_t)> func) {
  // This is a vector of (start, rank).
  // The rank is of the byte pair starting at position start.
  // The rank of the last item in the vector is not a valid value.
  std::vector<std::pair<uint64_t, uint64_t>> parts;
  parts.reserve(piece.size() + 1);
  for (auto idx = 0U; idx < piece.size() + 1; ++idx) {
    parts.emplace_back(idx, _max_size());
  }

  auto get_rank =
      [&piece, &ranks](const std::vector<std::pair<uint64_t, uint64_t>> &parts,
                       uint64_t start_idx,
                       uint64_t skip) -> std::optional<uint64_t> {
    if (start_idx + skip + 2 < parts.size()) {
      auto s = parts[start_idx].first;
      auto e = parts[start_idx + skip + 2].first;
      auto key = piece.substr(s, e - s);
      auto iter = ranks.find(key);
      if (iter != ranks.end()) {
        return iter->second;
      }
    }
    return std::nullopt;
  };

  // We look up the ranks once in the beginning and iteratively update
  // them during each merge, which reduces the number of rank lookups.
  for (auto i = 0U; i < parts.size() - 2; ++i) {
    auto rank = get_rank(parts, i, 0);
    if (rank) {
      // usize::MAX is a sentinel value and cannot be a valid rank
      if (*rank == _max_size()) {
        TK_LOG(Error, "at %" PRIu32 " rank is too large\n", i);
      }
      parts[i].second = *rank;
    }
  }

  // If you have n parts and m merges, this does O(mn) work.
  // We could do something with a heap and do O(m log n) work.
  // It is important to consider that n is often small (<100), and as such
  // the cache-locality benefits outweigh the algorithmic complexity downsides
  // of the `parts` vector data structure above.

  // Note that we hash bytes, not token pairs. As long as we train BPE the way
  // we currently do, this is equivalent. An easy way to break this would be
  // to decouple merge priority from token index or to prevent specific token
  // merges.
  while (true) {
    if (parts.size() == 1) {
      break;
    }

    // usize::MAX is a sentinel rank value allowing us to
    // take the min more quickly
    auto min_rank = std::make_pair<uint64_t, uint64_t>(_max_size(), 0);
    for (auto i = 0U; i < parts.size() - 1; ++i) {
      auto rank = parts[i].second;
      if (rank < min_rank.first) {
        min_rank.first = rank;
        min_rank.second = i;
      }
    }

    if (min_rank.first != _max_size()) {
      auto i = min_rank.second;

      // NOTE: We are about to remove parts[i + 1]. We do not do it
      // yet because there are cache-locality benefits to updating
      // parts[i] and parts[i-1] before removing, which could thrash
      // the cache. Thus, we update the rank calculation by skipping over
      // parts[i + 1], by invoking `get_rank!` with `skip = 1`.
      auto rank = get_rank(parts, i, 1);
      if (rank) {
        parts[i].second = *rank;
      } else {
        parts[i].second = _max_size();
      }
      if (i > 0) {
        rank = get_rank(parts, i - 1, 1);
        if (rank) {
          parts[i - 1].second = *rank;
        } else {
          parts[i - 1].second = _max_size();
        }
      }

      parts.erase(parts.begin() + (i + 1));
    } else {
      break;
    }
  }
  std::vector<uint64_t> out;
  out.reserve(parts.size() - 1);
  for (auto i = 0U; i < parts.size() - 1; ++i) {
    auto s = parts[i].first;
    auto e = parts[i + 1].first;
    out.push_back(func(s, e));
  }
  return out;
}

} // namespace
// ---- Helper utils end -------------------------------------------------------
// ---- protected start --------------------------------------------------------

std::pair<std::optional<std::string>, re2::StringPiece>
BPETokenizerBase::split_with_allowed_special_token_(
    re2::StringPiece &input, const Encoder &allowed_special) const {
  if (!special_token_regex_) {
    return std::make_pair(std::nullopt, input);
  }

  auto start = input.begin();
  std::string special;
  while (true) {
    if (!re2::RE2::FindAndConsume(&input, *special_token_regex_, &special)) {
      // No special token.
      break;
    }

    if (allowed_special.count(special) == 1) {
      // Found an allowed special token, split the text with it.
      return std::make_pair(
          special,
          re2::StringPiece(start, input.begin() - start - special.size()));
    } // else try to find the next special token
  }

  return std::make_pair(std::nullopt, input);
}

Result<std::pair<std::vector<uint64_t>, uint64_t>>
BPETokenizerBase::encode_with_special_token_(
    const std::string &text, const Encoder &allowed_special) const {
  std::vector<uint64_t> tokens;
  uint64_t last_piece_token_len = 0;
  re2::StringPiece input(text);
  while (true) {
    auto [special, sub_input] =
        split_with_allowed_special_token_(input, allowed_special);

    _encode(sub_input, tokens, last_piece_token_len);

    if (special) {
      uint64_t token = 0;
      try {
        token = special_token_encoder_.at(*special);
      } catch (const std::out_of_range &) {
        // Should never go here, since special pattern includes all special
        // chars.
        TK_LOG(Error, "unknown special token: %s\n", special->c_str());
        return Error::EncodeFailure;
      }

      tokens.push_back(token);
      last_piece_token_len = 0;
    } else {
      break;
    }
  }

  // last_piece_token_len is how many tokens came from the last regex split.
  // This is used for determining unstable tokens, since you can't merge
  // across (stable) regex splits
  return std::make_pair(tokens, last_piece_token_len);
}

Result<std::vector<uint64_t>>
BPETokenizerBase::byte_pair_encode_(const std::string &piece,
                                    const Encoder &encoder) const {
  if (piece.size() == 1) {
    auto iter = encoder.find(piece);
    if (iter != encoder.end()) {
      return std::vector<uint64_t>({iter->second});
    } else {
      // TODO: is it possible?
      return Error::EncodeFailure;
    }
  }

  return _byte_pair_merge(piece, encoder,
                          [&piece, &encoder](uint64_t start, uint64_t stop) {
                            std::string key = piece.substr(start, stop - start);
                            auto iter = encoder.find(key);
                            if (iter != encoder.end()) {
                              return iter->second;
                            } else {
                              // TODO: what if key does not exist? Should we
                              // return `unknown`? assert(false); // ??
                              return uint64_t(0);
                            }
                          });
}

// ---- protected end ----------------------------------------------------------
// ---- public start -----------------------------------------------------------

Result<std::vector<uint64_t>> BPETokenizerBase::encode(const std::string &text,
                                                       int8_t bos,
                                                       int8_t eos) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  auto res =
      TK_UNWRAP(encode_with_special_token_(text, special_token_encoder_)).first;
  for (auto i = 0; i < bos; ++i) {
    res.insert(res.begin(), bos_tok_);
  }
  for (auto i = 0; i < eos; ++i) {
    res.push_back(eos_tok_);
  }
  return Result<std::vector<uint64_t>>(std::move(res));
}

Result<std::string> BPETokenizerBase::decode(uint64_t prev,
                                             uint64_t cur) const {
  (void)prev;
  if (!initialized_) {
    return Error::Uninitialized;
  }
  std::string ret;

  std::string token_bytes;
  auto iter = decoder_.find(cur);
  if (iter != decoder_.end()) {
    token_bytes = iter->second;
  } else {
    iter = special_token_decoder_.find(cur);
    if (iter != special_token_decoder_.end()) {
      token_bytes = iter->second;
    } else {
      TK_LOG(Error, "unknown token: %" PRIu64 "\n", cur);
      return Error::DecodeFailure;
    }
  }
  _decode(token_bytes, ret);

  return ret;
}

// ---- public end -------------------------------------------------------------

} // namespace detail
} // namespace tokenizers
