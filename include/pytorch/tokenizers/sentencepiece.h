/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

// A tokenizer that works with sentencepiece. Used by Llama2.
#pragma once

#include <pytorch/tokenizers/tokenizer.h>
#include <memory>
#include <vector>
#include "sentencepiece_processor.h"
namespace tokenizers {

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
