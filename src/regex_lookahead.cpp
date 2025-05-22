/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file contains the implementation of create_regex with lookahead support

#include <pytorch/tokenizers/pcre2_regex.h>
#include <pytorch/tokenizers/regex.h>
#include <pytorch/tokenizers/std_regex.h>

#include <iostream>
#include <memory>

namespace tokenizers {

/**
 * @brief Factory function that creates a regex object using RE2 if possible.
 *        Falls back to PCRE2 if RE2 rejects the pattern due to lookahead.
 *        Falls back to std::regex if PCRE2 also fails.
 */

#ifdef _MSC_VER
#pragma weak create_fallback_regex
#endif // _MSC_VER
Result<std::unique_ptr<IRegex>> create_fallback_regex(
    const std::string& pattern) {
  auto pcre2 = std::make_unique<Pcre2Regex>("(" + pattern + ")");

  if (pcre2->regex_ != nullptr && pcre2->match_data_ != nullptr) {
    std::cout
        << "RE2 is unable to support things such as negative lookaheads in "
        << pattern << ", using PCRE2 instead." << std::endl;
    return static_cast<std::unique_ptr<IRegex>>(std::move(pcre2));
  }

  // If PCRE2 also fails, fall back to std::regex
  try {
    std::cout << "PCRE2 failed to compile pattern, falling back to std::regex.";
    auto std_regex = std::make_unique<StdRegex>("(" + pattern + ")");
    return static_cast<std::unique_ptr<IRegex>>(std::move(std_regex));
  } catch (const std::regex_error& e) {
    std::cerr << "std::regex failed: " << e.what() << std::endl;
    return tokenizers::Error::LoadFailure;
  }
}

} // namespace tokenizers
