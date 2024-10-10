/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/tokenizer/tokenizer.h>
#include <memory>

namespace executorch {
namespace extension {
namespace llm {

struct TokenIndex {
  const char* str;
  int32_t id;
};

// A simple Byte Pair Encoding (BPE) Tokenizer. Note that the current C++ code
// won't work with this class, it needs to go through tokenizer.py first.
class ET_EXPERIMENTAL BPETokenizer : public Tokenizer {
 public:
  explicit BPETokenizer();
  ~BPETokenizer() override;

  ::executorch::runtime::Error load(const std::string& tokenizer_path) override;

  ::executorch::runtime::Result<std::vector<uint64_t>>
  encode(const std::string& input, int8_t bos, int8_t eos) const override;

  ::executorch::runtime::Result<std::string> decode(
      uint64_t prev_token,
      uint64_t token) const override;

 private:
  std::unique_ptr<char*[]> vocab_ = nullptr;
  std::unique_ptr<float[]> vocab_scores_ = nullptr;
  std::unique_ptr<TokenIndex[]> sorted_vocab_ = nullptr;
  unsigned int max_token_length_ = 0;
  unsigned char byte_pieces_[512]; // stores all single-byte strings
};

} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::BPETokenizer;
using ::executorch::extension::llm::TokenIndex;
} // namespace executor
} // namespace torch
