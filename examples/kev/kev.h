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
#include <optional>
#include <string>
#include <vector>

namespace tokenizers {
class Tokenizer;
}

namespace kev {

struct Option {
  std::string label;
  std::optional<std::string> description;
};

struct Question {
  std::string id;
  std::string instructions;
  std::vector<Option> options;
};

struct OptionScores {
  std::vector<double> logits;
  std::vector<double> probabilities;
  size_t selected_index;
};

struct Answer {
  std::string question_id;
  std::vector<std::string> labels;
  OptionScores scores;
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
  friend executorch::runtime::Result<std::vector<Answer>> evaluate(
      const Prefix&,
      const std::vector<Question>&);

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
executorch::runtime::Result<std::vector<Answer>> evaluate(
    const Prefix& prefix,
    const std::vector<Question>& questions);

} // namespace kev
