/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <regex>
#include <string>
#include "regex.h"

namespace tokenizers {

/**
 * @brief std::regex-based implementation of IRegex.
 */
class StdRegex : public IRegex {
 public:
  /**
   * @brief Construct a std::regex wrapper with the given pattern.
   *
   * @param pattern The regex pattern to compile.
   * @throws std::regex_error if the pattern is invalid.
   */
  explicit StdRegex(const std::string& pattern);

  /**
   * @brief Find all non-overlapping matches in the input string.
   */
  virtual std::vector<Match> find_all(const std::string& text) const override;

 private:
  std::regex regex_;
};

} // namespace tokenizers
