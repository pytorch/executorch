/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>

#include <re2/re2.h>

#include <pytorch/tokenizers/regex.h>

namespace tokenizers {

/**
 * @brief RE2-based implementation of IRegex.
 */
class Re2Regex : public IRegex {
 public:
  /**
   * @brief Construct a RE2 regex.
   */
  explicit Re2Regex() {}

  /**
   * @brief compile the given regex pattern.
   * @param pattern The regex pattern to compile.
   * @return An Error object indicating success or failure of the compilation.
   */
  virtual Error compile(const std::string& pattern) override;

  /**
   * @brief Return all non-overlapping matches found in the input string.
   */
  virtual std::vector<Match> find_all(const std::string& text) const override;

 private:
  std::unique_ptr<re2::RE2> regex_;
};

} // namespace tokenizers
