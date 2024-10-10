/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cinttypes>
// patternlint-disable-next-line executorch-cpp-nostdinc
#include <string>
// patternlint-disable-next-line executorch-cpp-nostdinc
#include <vector>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace llm {

// A tokenizer interface.
class ET_EXPERIMENTAL Tokenizer {
 public:
  explicit Tokenizer() {}
  virtual ~Tokenizer() {}

  virtual ::executorch::runtime::Error load(
      const std::string& tokenizer_path) = 0;

  virtual ::executorch::runtime::Result<std::vector<uint64_t>>
  encode(const std::string& input, int8_t bos, int8_t eos) const = 0;

  ::executorch::runtime::Error decode_verify(uint64_t token) const {
    if (!initialized_) {
      ET_LOG(Error, "Tokenizer not initialized");
      return ::executorch::runtime::Error::NotSupported;
    }
    if (token >= vocab_size_) {
      ET_LOG(
          Error,
          "token  %" PRIu64 " is out side of vacab range %d",
          token,
          vocab_size_);
      return ::executorch::runtime::Error::NotSupported;
    }
    return ::executorch::runtime::Error::Ok;
  }

  virtual ::executorch::runtime::Result<std::string> decode(
      uint64_t prev_token,
      uint64_t token) const = 0;

  // getters
  int32_t vocab_size() const {
    return vocab_size_;
  }

  uint64_t bos_tok() const {
    return bos_tok_;
  }

  uint64_t eos_tok() const {
    return eos_tok_;
  }

 protected:
  bool initialized_ = false;
  int32_t vocab_size_ = 0;
  uint64_t bos_tok_ = 0;
  uint64_t eos_tok_ = 0;
};

} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::Tokenizer;
} // namespace executor
} // namespace torch
