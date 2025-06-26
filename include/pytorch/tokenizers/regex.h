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
   * @brief Compile the given regex pattern.
   * @param pattern The regex pattern to compile.
   * @return An Error object indicating success or failure of the compilation.
   */
  virtual Error compile(const std::string& pattern) = 0;

  /**
   * @brief Find all non-overlapping matches in the input string.
   *
   * @param text The input string to search.
   * @return A vector of strings containing all matched substrings.
   */
  virtual std::vector<Match> find_all(const std::string& text) const = 0;

  /**
   * @brief Escape special regex characters in a string to treat it as literal.
   *
   * @param input The input string to escape.
   * @return The escaped string that can be used as a literal pattern in regex.
   */
  static std::string escape(const std::string& input);
};

// Function pointer type for create_fallback_regex implementations
using FallbackRegexFn = Result<std::unique_ptr<IRegex>> (*)(const std::string&);

/**
 * @brief Creates a regex instance. If no strong symbol defined, only
 * uses RE2. This is a weak symbol to allow other regex libraries to be
 * used.
 *
 * @param pattern The regex pattern to compile.
 * @return A unique pointer to an IRegex-compatible object.
 */
Result<std::unique_ptr<IRegex>> create_regex(const std::string& pattern);

bool register_override_fallback_regex(FallbackRegexFn fn);

FallbackRegexFn get_fallback_regex();

} // namespace tokenizers
