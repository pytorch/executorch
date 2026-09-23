/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "api.h"

namespace tokenizers {
class Tokenizer;
}

namespace kev {

// Borrows the Module and tokenizer, which must outlive it. Calls must be
// serialized per Module, including calls through the prefix API below.
class Kev final : public SystemOne {
 public:
  Kev(executorch::extension::Module& module,
      const tokenizers::Tokenizer& tokenizer);

  executorch::runtime::Result<Answers> system_one(
      const std::string& state,
      const Questions& questions) override;

 private:
  executorch::extension::Module& module_;
  const tokenizers::Tokenizer& tokenizer_;
};

// Owns the prefix tensors; borrows the Module and tokenizer, which must outlive
// it. Prefixes can coexist. Call sequentially per Module, including prefill.
class Prefix {
 public:
  Prefix(Prefix&&) = default;
  Prefix& operator=(Prefix&&) = default;

 private:
  Prefix() = default;
  friend executorch::runtime::Result<Prefix> prefill(
      executorch::extension::Module&,
      const tokenizers::Tokenizer&,
      const std::string&);
  friend executorch::runtime::Result<Answers> evaluate(
      const Prefix&,
      const Questions&);

  executorch::extension::Module* module_ = nullptr;
  const tokenizers::Tokenizer* tokenizer_ = nullptr;
  std::array<executorch::extension::TensorPtr, 3> state_;
  std::array<int64_t, 5> special_{};
  int64_t pad_id_ = 0;
  size_t length_ = 0;
  size_t max_context_ = 0;
  size_t max_questions_ = 0;
  size_t max_options_ = 0;
};

executorch::runtime::Result<Prefix> prefill(
    executorch::extension::Module& module,
    const tokenizers::Tokenizer& tokenizer,
    const std::string& state);

// Every question starts from the same prefix. This call leaves it unchanged;
// later calls can ask different questions. Answers own their values.
// Requests are split into the program's batch limit and retain their order.
executorch::runtime::Result<Answers> evaluate(
    const Prefix& prefix,
    const Questions& questions);

} // namespace kev
