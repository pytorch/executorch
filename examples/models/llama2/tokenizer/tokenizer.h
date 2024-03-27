/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple Byte Pair Encoding (BPE) Tokenizer. Note that the vanila tokenizer
// model won't work with this class, it needs to go through tokenizer.py first.
#pragma once

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/result.h>

namespace torch {
namespace executor {

struct TokenIndex {
  const char* str;
  int32_t id;
};

class Tokenizer {
 public:
  explicit Tokenizer(int32_t vocab_size, int32_t bos_tok, int32_t eos_tok);
  ~Tokenizer();

  Error load(const std::string& tokenizer_path);

  Error encode(
      const char* text,
      int8_t bos,
      int8_t eos,
      int32_t* tokens,
      int32_t* n_tokens);

  Result<const char*> decode(int prev_token, int token);

  // getters
  int32_t vocab_size() const {
    return vocab_size_;
  }

  int32_t bos_tok() const {
    return bos_tok_;
  }

  int32_t eos_tok() const {
    return eos_tok_;
  }

 private:
  bool initialized_;
  const int32_t vocab_size_;
  int32_t bos_tok_, eos_tok_;
  std::unique_ptr<char*[]> vocab_;
  std::unique_ptr<float[]> vocab_scores_;
  std::unique_ptr<TokenIndex[]> sorted_vocab_;
  unsigned int max_token_length_;
  unsigned char byte_pieces_[512]; // stores all single-byte strings
};

} // namespace executor
} // namespace torch
