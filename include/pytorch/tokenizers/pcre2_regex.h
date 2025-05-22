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

// Define PCRE2 code unit width before including pcre2.h
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <pytorch/tokenizers/regex.h>

namespace tokenizers {

/**
 * @brief PCRE2-based implementation of IRegex.
 */
class Pcre2Regex : public IRegex {
 public:
  /**
   * @brief Construct a PCRE2 regex with the given pattern.
   *
   * @param pattern The regex pattern to compile.
   */
  explicit Pcre2Regex(const std::string& pattern);

  /**
   * @brief Destructor to clean up PCRE2 resources.
   */
  ~Pcre2Regex();

  /**
   * @brief Return all non-overlapping matches found in the input string.
   */
  virtual std::vector<Match> find_all(const std::string& text) const override;

 private:
  pcre2_code* regex_;
  pcre2_match_data* match_data_;

  friend Result<std::unique_ptr<IRegex>> create_fallback_regex(
      const std::string& pattern);
};

} // namespace tokenizers
