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
#include <vector>

#include <pytorch/tokenizers/result.h>

namespace tokenizers {

struct Match {
  size_t start; // starting index of the match
  size_t end; // ending index of the match (exclusive)
};

/**
 * @brief Abstract interface for regex wrappers.
 */
class IRegex {
 public:
  virtual ~IRegex() = default;

  /**
   * @brief Find all non-overlapping matches in the input string.
   *
   * @param text The input string to search.
   * @return A vector of strings containing all matched substrings.
   */
  virtual std::vector<Match> find_all(const std::string& text) const = 0;
};

/**
 * @brief Creates a regex instance. If no strong symbol defined, only
 * uses RE2. This is a weak symbol to allow other regex libraries to be
 * used.
 *
 * @param pattern The regex pattern to compile.
 * @return A unique pointer to an IRegex-compatible object.
 */
Result<std::unique_ptr<IRegex>> create_regex(const std::string& pattern);

/**
 * @brief Creates a fallback regex instance. If no strong symbol defined,
 * returns Error, otherwise uses PCRE2 and std::regex.
 * This is a weak symbol to allow other regex libraries to be used.
 *
 * @param pattern The regex pattern to compile.
 * @return A unique pointer to an IRegex-compatible object.
 */
Result<std::unique_ptr<IRegex>> create_fallback_regex(
    const std::string& pattern) TK_WEAK;

} // namespace tokenizers
