/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/models/llama2/tokenizer/tokenizer.h>
#include <sentencepiece_processor.h>
#include <cstdint>

namespace torch {
namespace executor {
// ----------------------- SPTokenizer -----------------------
// Used by sentencepiece. Adapted from llama2.c.
struct TokenIndex {
  const char* str;
  int32_t id;
};

class SPTokenizer : public Tokenizer {
 public:
  explicit SPTokenizer()
      : Tokenizer(),
        _processor(std::make_unique<sentencepiece::SentencePieceProcessor>()){};
  ~SPTokenizer() override;

  Error load(const std::string& tokenizer_path) override;

  Result<std::vector<uint64_t>>
  encode(const std::string& input, int8_t bos, int8_t eos) override;

  Result<std::string> decode(uint64_t prev_token, uint64_t token) override;

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> _processor;
};
} // namespace executor
} // namespace torch
