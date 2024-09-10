/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Adopted from https://github.com/sewenew/tokenizer

// @lint-ignore-every LICENSELINT
/**************************************************************************
   Copyright (c) 2023 sewenew

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 *************************************************************************/

#include <executorch/extension/llm/tokenizer/base64.h>
#include <executorch/extension/llm/tokenizer/tiktoken.h>
#include <executorch/runtime/core/result.h>
#include <fstream>
#include <limits>

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace executorch {
namespace extension {
namespace llm {

// ------------------------------Util start------------------------------------

static uint64_t _max_size() {
  return std::numeric_limits<uint64_t>::max();
}

static Re2UPtr _create_regex(const std::string& pattern) {
  assert(!pattern.empty());

  return std::make_unique<re2::RE2>("(" + pattern + ")");
}

static Re2UPtr _build_special_token_regex(const Encoder& special_encoder) {
  std::string special_pattern;
  for (const auto& ele : special_encoder) {
    if (!special_pattern.empty()) {
      special_pattern += "|";
    }
    special_pattern += re2::RE2::QuoteMeta(ele.first);
  }

  if (special_pattern.empty()) {
    return nullptr;
  }

  return _create_regex(special_pattern);
}

static Result<std::pair<std::string, uint64_t>> _parse(
    const std::string& line) {
  // Tiktoken format
  // https://github.com/openai/tiktoken/blob/main/tiktoken/load.py#L140 <base64
  // encoded token str> <rank>
  auto pos = line.find(" ");
  ET_CHECK_OR_RETURN_ERROR(
      pos != std::string::npos,
      InvalidArgument,
      "invalid tiktoken line: %s",
      line.c_str());

  auto token = ET_UNWRAP(base64::decode({line.data(), pos}));
  uint64_t rank = 0;
  try {
    rank = std::stoul(line.substr(pos + 1));
  } catch (const std::exception&) {
    ET_CHECK_OR_RETURN_ERROR(
        false, InvalidArgument, "invalid encoder rank: %s", line.c_str());
  }

  return std::pair{std::move(token), rank};
}

static Result<Encoder> _load_encoder(const std::string& path) {
  std::ifstream file(path);
  ET_CHECK_OR_RETURN_ERROR(
      file, InvalidArgument, "failed to open encoder file: %s", path.c_str());

  Encoder encoder;
  std::string line;
  while (std::getline(file, line)) {
    auto [token, rank] = ET_UNWRAP(_parse(line));

    ET_CHECK_OR_RETURN_ERROR(
        encoder.emplace(std::move(token), rank).second,
        InvalidArgument,
        "duplicate item: %s",
        line.c_str());
  }

  return encoder;
}

static Result<Decoder> _build_decoder(const Encoder& encoder) {
  Decoder decoder;
  for (const auto& [k, v] : encoder) {
    decoder.emplace(v, k);
  }

  ET_CHECK_OR_RETURN_ERROR(
      encoder.size() == decoder.size(),
      InvalidArgument,
      "duplicate items in encoder");

  return decoder;
}

static std::vector<uint64_t> _byte_pair_merge(
    const std::string& piece,
    const std::unordered_map<std::string, uint64_t>& ranks,
    std::function<uint64_t(uint64_t, uint64_t)> func) {
  // This is a vector of (start, rank).
  // The rank is of the byte pair starting at position start.
  // The rank of the last item in the vector is not a valid value.
  std::vector<std::pair<uint64_t, uint64_t>> parts;
  parts.reserve(piece.size() + 1);
  for (auto idx = 0U; idx < piece.size() + 1; ++idx) {
    parts.emplace_back(idx, _max_size());
  }

  auto get_rank = [&piece, &ranks](
                      const std::vector<std::pair<uint64_t, uint64_t>>& parts,
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
      ET_CHECK_MSG(*rank != _max_size(), "rank is too large");
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

static std::vector<uint64_t> _byte_pair_encode(
    const std::string& piece,
    const Encoder& encoder) {
  if (piece.size() == 1) {
    auto iter = encoder.find(piece);
    if (iter != encoder.end()) {
      return std::vector<uint64_t>({iter->second});
    } else {
      // TODO: is it possible?
      return {};
    }
  }

  return _byte_pair_merge(
      piece, encoder, [&piece, &encoder](uint64_t start, uint64_t stop) {
        std::string key = piece.substr(start, stop - start);
        auto iter = encoder.find(key);
        if (iter != encoder.end()) {
          return iter->second;
        } else {
          // TODO: what if key does not exist? Should we return `unknown`?
          // assert(false); // ??
          return uint64_t(0);
        }
      });
}
// ------------------------------Util end------------------------------------
// -------------------------private method start-------------------------------

template <typename T>
std::pair<std::optional<std::string>, re2::StringPiece>
Tiktoken::_split_with_allowed_special_token(
    re2::StringPiece& input,
    const T& allowed_special) const {
  if (!_special_token_regex) {
    return std::make_pair(std::nullopt, input);
  }

#if __cplusplus >= 202002L
  auto start = input.begin();
#else
  const char* start = input.data();
#endif
  std::string special;
  while (true) {
    if (!re2::RE2::FindAndConsume(&input, *_special_token_regex, &special)) {
      // No special token.
      break;
    }

    if (allowed_special.count(special) == 1) {
      // Found an allowed special token, split the text with it.
#if __cplusplus >= 202002L
      return std::make_pair(
          special,
          re2::StringPiece(start, input.begin() - start - special.size()));
#else
      return std::make_pair(
          special,
          re2::StringPiece(start, (input.data() - start) - special.size()));
#endif
    } // else try to find the next special token
  }

  return std::make_pair(std::nullopt, input);
}

void Tiktoken::_encode(
    re2::StringPiece& input,
    std::vector<uint64_t>& ret,
    uint64_t& last_piece_token_len) const {
  std::string piece;
  assert(_regex);
  while (re2::RE2::FindAndConsume(&input, *_regex, &piece)) {
    auto iter = _encoder.find(piece);
    if (iter != _encoder.end()) {
      last_piece_token_len = 1;
      ret.push_back(iter->second);
      continue;
    }
    auto tokens = _byte_pair_encode(piece, _encoder);
    last_piece_token_len = tokens.size();
    ret.insert(ret.end(), tokens.begin(), tokens.end());
  }
}

template <typename T>
std::pair<std::vector<uint64_t>, uint64_t> Tiktoken::_encode_with_special_token(
    const std::string& text,
    const T& allowed_special) const {
  std::vector<uint64_t> tokens;
  uint64_t last_piece_token_len = 0;
  re2::StringPiece input(text);
  while (true) {
    auto [special, sub_input] =
        _split_with_allowed_special_token(input, allowed_special);

    _encode(sub_input, tokens, last_piece_token_len);

    if (special) {
      uint64_t token = 0;
      try {
        token = _special_token_encoder.at(*special);
      } catch (const std::out_of_range&) {
        // Should never go here, since special pattern includes all special
        // chars.
        ET_CHECK_MSG(false, "unknown special token: %s", special->c_str());
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

Encoder Tiktoken::_build_special_token_encoder(ssize_t num_base_tokens) const {
  Encoder special_token_encoder;
  for (ssize_t i = 0; i < _special_tokens->size(); ++i) {
    special_token_encoder.emplace(_special_tokens->at(i), num_base_tokens + i);
  }
  return special_token_encoder;
}

// -------------------------private method end-------------------------------
// -------------------------public method start-------------------------------

Tiktoken::Tiktoken(
    std::unique_ptr<std::vector<std::string>> special_tokens,
    size_t bos_token_index,
    size_t eos_token_index)
    : Tokenizer(),
      _special_tokens(std::move(special_tokens)),
      _bos_token_index(bos_token_index),
      _eos_token_index(eos_token_index) {
  ET_CHECK_MSG(
      _bos_token_index < _special_tokens->size(),
      "invalid bos_token_index %zu",
      _bos_token_index);
  ET_CHECK_MSG(
      _eos_token_index < _special_tokens->size(),
      "invalid eos_token_index %zu",
      _eos_token_index);
}

Error Tiktoken::load(const std::string& path) {
  _encoder = ET_UNWRAP(_load_encoder(path));
  _special_token_encoder = _build_special_token_encoder(_encoder.size());

  _decoder = ET_UNWRAP(_build_decoder(_encoder));
  _special_token_decoder = ET_UNWRAP(_build_decoder(_special_token_encoder));

  _regex = _create_regex(_pattern);
  // Warmup re2 as it is slow on the first run, void the return value as it's
  // not needed Refer to
  // https://github.com/google/re2/blob/6dcd83d60f7944926bfd308cc13979fc53dd69ca/re2/fuzzing/re2_fuzzer.cc#L136-L141
  (void)_regex->ReverseProgramSize();

  _special_token_regex = _build_special_token_regex(_special_token_encoder);
  // Same as above, warm up re2
  (void)_special_token_regex->ReverseProgramSize();

  // initialize vocab_size, bos_tok, eos_tok
  vocab_size_ = _encoder.size() + _special_token_encoder.size();
  bos_tok_ = _special_token_encoder.at(_special_tokens->at(_bos_token_index));
  eos_tok_ = _special_token_encoder.at(_special_tokens->at(_eos_token_index));

  initialized_ = true;
  return Error::Ok;
}

Result<std::vector<uint64_t>>
Tiktoken::encode(const std::string& text, int8_t bos, int8_t eos) const {
  if (!initialized_) {
    return Error::NotSupported;
  }
  auto res = _encode_with_special_token(text, _special_token_encoder).first;
  for (auto i = 0; i < bos; ++i) {
    res.insert(res.begin(), bos_tok_);
  }
  for (auto i = 0; i < eos; ++i) {
    res.push_back(eos_tok_);
  }
  return Result<std::vector<uint64_t>>(std::move(res));
}

Result<std::string> Tiktoken::decode(uint64_t prev, uint64_t cur) const {
  (void)prev;
  ET_CHECK_OK_OR_RETURN_ERROR(Tokenizer::decode_verify(cur));
  std::string ret;

  std::string token_bytes;
  auto iter = _decoder.find(cur);
  if (iter != _decoder.end()) {
    token_bytes = iter->second;
  } else {
    iter = _special_token_decoder.find(cur);
    if (iter != _special_token_decoder.end()) {
      token_bytes = iter->second;
    } else {
      ET_CHECK_MSG(false, "unknown token: %" PRIu64, cur);
    }
  }
  ret += token_bytes;

  return ret;
}
// -------------------------public method end-------------------------------

} // namespace llm
} // namespace extension
} // namespace executorch
