/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/models/llama2/tokenizer/tokenizer.h>
#include <cstdint>

namespace torch {
namespace executor {

struct TokenIndex {
  const char* str;
  int32_t id;
};

class BPETokenizer : public Tokenizer {
 public:
  explicit BPETokenizer(int32_t vocab_size, uint64_t bos_tok, uint64_t eos_tok);
  ~BPETokenizer() override;

  Error load(const std::string& tokenizer_path) override;

  Result<std::vector<uint64_t>>
  encode(const std::string& input, int8_t bos, int8_t eos) override;

  Result<std::string> decode(uint64_t prev_token, uint64_t token) override;

 private:
  std::unique_ptr<char*[]> vocab_;
  std::unique_ptr<float[]> vocab_scores_;
  std::unique_ptr<TokenIndex[]> sorted_vocab_;
  unsigned int max_token_length_;
  unsigned char byte_pieces_[512]; // stores all single-byte strings
};
} // namespace executor
} // namespace torch
