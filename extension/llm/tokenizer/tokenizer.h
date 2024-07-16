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
#include <vector>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/result.h>

namespace torch {
namespace executor {

class Tokenizer {
 public:
  explicit Tokenizer() {}
  virtual ~Tokenizer() {}

  virtual Error load(const std::string& tokenizer_path) = 0;

  virtual Result<std::vector<uint64_t>>
  encode(const std::string& input, int8_t bos, int8_t eos) const = 0;

  Error decode_verify(uint64_t token) const {
    if (!initialized_) {
      ET_LOG(Error, "Tokenizer not initialized");
      return Error::NotSupported;
    }
    if (token >= vocab_size_) {
      ET_LOG(
          Error,
          "token  %" PRIu64 " is out side of vacab range %d",
          token,
          vocab_size_);
      return Error::NotSupported;
    }
    return Error::Ok;
  }

  virtual Result<std::string> decode(uint64_t prev_token, uint64_t token)
      const = 0;

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

} // namespace executor
} // namespace torch
