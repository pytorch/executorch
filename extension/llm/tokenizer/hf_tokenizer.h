/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/tokenizer/tokenizer.h>

namespace executorch {
namespace extension {
namespace llm {

class ET_EXPERIMENTAL HfTokenizer : public Tokenizer {
 public:
  explicit HfTokenizer(){};
  ~HfTokenizer() override;

  ::executorch::runtime::Error load(const std::string& tokenizer_path) override;

  ::executorch::runtime::Result<std::vector<uint64_t>>
  encode(const std::string& input, int8_t bos, int8_t eos) const override;

  ::executorch::runtime::Result<std::string> decode(
      uint64_t prev_token,
      uint64_t token) const override;
};

} // namespace llm
} // namespace extension
} // namespace executorch
