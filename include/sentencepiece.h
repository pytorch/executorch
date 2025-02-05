/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A tokenizer that works with sentencepiece.
#pragma once

#include <memory>
#include <vector>
#include "sentencepiece_processor.h"
#include "tokenizer.h"
namespace tokenizers {

struct TokenIndex {
  const char* str;
  int32_t id;
};

class SPTokenizer : public Tokenizer {
 public:
  explicit SPTokenizer();
  ~SPTokenizer() override;

  Error load(const std::string& tokenizer_path) override;

  Result<std::vector<uint64_t>>
  encode(const std::string& input, int8_t bos, int8_t eos) const override;

  Result<std::string> decode(uint64_t prev_token, uint64_t token)
      const override;

 private:
  std::unique_ptr<sentencepiece::SentencePieceProcessor> _processor;
};

} // namespace tokenizers
