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

#include <pytorch/tokenizers/base64.h>
#include <pytorch/tokenizers/tiktoken.h>
#include <cinttypes>
#include <fstream>
#include <limits>
#include "re2/re2.h"

namespace tokenizers {

using namespace detail;

// ------------------------------Util start------------------------------------
namespace {

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
  TK_CHECK_OR_RETURN_ERROR(
      pos != std::string::npos,
      ParseFailure,
      "invalid tiktoken line: %s",
      line.c_str());

  auto token = TK_UNWRAP(base64::decode({line.data(), pos}));
  uint64_t rank = 0;
  try {
    rank = std::stoul(line.substr(pos + 1));
  } catch (const std::exception&) {
    TK_CHECK_OR_RETURN_ERROR(
        false, EncodeFailure, "invalid encoder rank: %s", line.c_str());
  }

  return std::pair{std::move(token), rank};
}

static Result<Encoder> _load_encoder(const std::string& path) {
  std::ifstream file(path);
  TK_CHECK_OR_RETURN_ERROR(
      file, LoadFailure, "failed to open encoder file: %s", path.c_str());

  Encoder encoder;
  std::string line;
  while (std::getline(file, line)) {
    auto [token, rank] = TK_UNWRAP(_parse(line));

    TK_CHECK_OR_RETURN_ERROR(
        encoder.emplace(std::move(token), rank).second,
        ParseFailure,
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

  TK_CHECK_OR_RETURN_ERROR(
      encoder.size() == decoder.size(),
      LoadFailure,
      "duplicate items in encoder");

  return decoder;
}

} // namespace

// ------------------------------Util end------------------------------------
// -------------------------private method start-------------------------------

template <typename T>
std::pair<std::optional<std::string>, re2::StringPiece>
Tiktoken::_split_with_allowed_special_token(
    re2::StringPiece& input,
    const T& allowed_special) const {
  if (!special_token_regex_) {
    return std::make_pair(std::nullopt, input);
  }

#if __cplusplus >= 202002L
  auto start = input.begin();
#else
  const char* start = input.data();
#endif
  std::string special;
  while (true) {
    if (!re2::RE2::FindAndConsume(&input, *special_token_regex_, &special)) {
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

Error Tiktoken::_encode(
    re2::StringPiece& input,
    std::vector<uint64_t>& ret,
    uint64_t& last_piece_token_len) const {
  std::string piece;
  assert(_regex);
  while (re2::RE2::FindAndConsume(&input, *_regex, &piece)) {
    auto iter = encoder_.find(piece);
    if (iter != encoder_.end()) {
      last_piece_token_len = 1;
      ret.push_back(iter->second);
      continue;
    }
    auto tokens = TK_UNWRAP(byte_pair_encode_(piece, encoder_));
    last_piece_token_len = tokens.size();
    ret.insert(ret.end(), tokens.begin(), tokens.end());
  }
  return Error::Ok;
}

void Tiktoken::_decode(re2::StringPiece input, std::string& ret) const {
#ifdef _USE_INTERNAL_STRING_VIEW
  ret += input.as_string();
#else
  ret += input;
#endif
}

template <typename T>
Result<std::pair<std::vector<uint64_t>, uint64_t>>
Tiktoken::_encode_with_special_token(
    const std::string& text,
    const T& allowed_special) const {
  std::vector<uint64_t> tokens;
  uint64_t last_piece_token_len = 0;
  re2::StringPiece input(text);
  while (true) {
    auto [special, sub_input] =
        _split_with_allowed_special_token(input, allowed_special);

    TK_CHECK_OK_OR_RETURN_ERROR(
        _encode(sub_input, tokens, last_piece_token_len));

    if (special) {
      uint64_t token = 0;
      try {
        token = special_token_encoder_.at(*special);
      } catch (const std::out_of_range&) {
        // Should never go here, since special pattern includes all special
        // chars.
        TK_LOG(Error, "unknown special token: %s", special->c_str());
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

Encoder Tiktoken::_build_special_token_encoder(ssize_t num_base_tokens) const {
  Encoder special_token_encoder;
  for (ssize_t i = 0; i < _special_tokens->size(); ++i) {
    special_token_encoder.emplace(_special_tokens->at(i), num_base_tokens + i);
  }
  return special_token_encoder;
}

// -------------------------private method end-------------------------------
// -------------------------public method start-------------------------------

Error Tiktoken::load(const std::string& path) {
  encoder_ = TK_UNWRAP(_load_encoder(path));
  special_token_encoder_ = _build_special_token_encoder(encoder_.size());

  decoder_ = TK_UNWRAP(_build_decoder(encoder_));
  special_token_decoder_ = TK_UNWRAP(_build_decoder(special_token_encoder_));

  _regex = _create_regex(_pattern);
  // Warmup re2 as it is slow on the first run, void the return value as it's
  // not needed Refer to
  // https://github.com/google/re2/blob/6dcd83d60f7944926bfd308cc13979fc53dd69ca/re2/fuzzing/re2_fuzzer.cc#L136-L141
  (void)_regex->ReverseProgramSize();

  special_token_regex_ = _build_special_token_regex(special_token_encoder_);
  // Same as above, warm up re2
  (void)special_token_regex_->ReverseProgramSize();

  // initialize vocab_size, bos_tok, eos_tok
  vocab_size_ = encoder_.size() + special_token_encoder_.size();
  bos_tok_ = special_token_encoder_.at(_special_tokens->at(_bos_token_index));
  eos_tok_ = special_token_encoder_.at(_special_tokens->at(_eos_token_index));

  initialized_ = true;
  return Error::Ok;
}

// -------------------------public method end-------------------------------

} // namespace tokenizers
