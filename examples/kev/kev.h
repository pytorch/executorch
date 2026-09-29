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

class Kev;

// Owns the snapshot. The Kev instance that created it must outlive it.
class Prefix {
 public:
  Prefix(Prefix&&) = default;
  Prefix& operator=(Prefix&&) = default;

 private:
  Prefix() = default;
  friend class Kev;

  const Kev* owner_ = nullptr;
  std::array<executorch::extension::TensorPtr, 3> state_;
  std::array<int64_t, 5> special_{};
  int64_t pad_id_ = 0;
  size_t length_ = 0;
  size_t max_context_ = 0;
  size_t max_questions_ = 0;
  size_t max_options_ = 0;
};

// Borrows the Module and tokenizer, which must outlive it. All calls must be
// serialized per Module. Multiple prefixes may coexist.
class Kev final : public SystemOne {
 public:
  Kev(executorch::extension::Module& module,
      const tokenizers::Tokenizer& tokenizer);

  executorch::runtime::Result<Answers> system_one(
      const std::string& state,
      const Questions& questions) override;

  executorch::runtime::Result<Prefix> prefill(const std::string& state);

  // Uses a prefix from this instance, leaving it unchanged. Answers own their
  // values. Requests are split into the program's batch limit and retain order.
  executorch::runtime::Result<Answers> evaluate(
      const Prefix& prefix,
      const Questions& questions);

 private:
  executorch::extension::Module& module_;
  const tokenizers::Tokenizer& tokenizer_;
};

} // namespace kev
