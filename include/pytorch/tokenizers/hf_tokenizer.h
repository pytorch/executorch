/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Used by many Huggingface models. Adapted from a combination of the original
// rust implementation (https://github.com/huggingface/tokenizers/tree/main)
// and the corresponding support in llama.cpp
// (https://github.com/ggerganov/llama.cpp)
#pragma once

// Standard
#include <string>

// Third Party
#include <re2/re2.h>

// Local
#include <pytorch/tokenizers/bpe_tokenizer_base.h>
#include <pytorch/tokenizers/error.h>
#include <pytorch/tokenizers/pre_tokenizer.h>
#include <pytorch/tokenizers/result.h>
#include <pytorch/tokenizers/token_decoder.h>

namespace tokenizers {
class HFTokenizer : public detail::BPETokenizerBase {
 public:
  /*-- Public Interface --*/

  /**
   * Default initialize with no loaded data
   */
  explicit HFTokenizer() {}
  ~HFTokenizer() {}

  /**
   * Load the model data into the
   */
  Error load(const std::string& tokenizer_path) override;

 private:
  Error _encode(
      re2::StringPiece& input,
      std::vector<uint64_t>& ret,
      uint64_t& last_piece_token_len) const override;

  void _decode(re2::StringPiece input, std::string& ret) const override;

  PreTokenizer::Ptr _pretokenizer;
  TokenDecoder::Ptr _decoder;
};

} // namespace tokenizers
