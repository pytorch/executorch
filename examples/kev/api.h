/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/result.h>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace kev {

struct Choice {
  std::string instructions;
  std::vector<std::pair<std::string, std::optional<std::string>>> criteria;
};

enum class NoulOutcome {
  False,
  True,
};

struct Noul {
  std::string instructions;
  std::map<NoulOutcome, std::string> criteria;
};

struct Score {
  std::string instructions;
  std::vector<std::string> criteria;
};

using Question = std::variant<Choice, Noul, Score>;
using Questions = std::vector<std::pair<std::string, Question>>;

struct ChoiceAnswer {
  std::string choice;
  std::map<std::string, double> probabilities;
  double confidence;
};

struct NoulAnswer {
  double noul;
};

struct ScoreAnswer {
  double score;
  std::vector<std::string> legend;
  std::vector<double> probabilities;
  double confidence;
};

using Answer = std::variant<ChoiceAnswer, NoulAnswer, ScoreAnswer>;
using Answers = std::vector<std::pair<std::string, Answer>>;

class SystemOne {
 public:
  virtual ~SystemOne() = default;

  virtual executorch::runtime::Result<Answers> system_one(
      const std::string& state,
      const Questions& questions) = 0;
};

} // namespace kev
