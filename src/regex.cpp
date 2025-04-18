/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pytorch/tokenizers/re2_regex.h>
#include <pytorch/tokenizers/regex.h>
#include <pytorch/tokenizers/std_regex.h>

#include <re2/re2.h>
#include <iostream>
#include <memory>

namespace tokenizers {

/**
 * @brief Factory function that creates a regex object using RE2 if possible.
 *        Falls back to std::regex if RE2 rejects the pattern with
 *        ErrorBadPerlOp.
 */
Result<std::unique_ptr<IRegex>> create_regex(const std::string& pattern) {
  // Try RE2 first
  auto re2 = std::make_unique<Re2Regex>("(" + pattern + ")");

  if (re2->regex_->ok()) {
    return static_cast<std::unique_ptr<IRegex>>(std::move(re2));
  }

  if (re2->regex_->error_code() == re2::RE2::ErrorBadPerlOp) {
    try {
      std::cout
          << "RE2 is unable to support things such as negative lookaheads in "
          << pattern << ", defaulting to std::regex.";
      auto std_regex = std::make_unique<StdRegex>("(" + pattern + ")");
      return static_cast<std::unique_ptr<IRegex>>(std::move(std_regex));
    } catch (const std::regex_error& e) {
      std::cerr << "std::regex failed: " << e.what() << std::endl;
      return tokenizers::Error::LoadFailure;
    }
  } else {
    std::cerr << "RE2 failed to compile pattern: " << pattern << "\n";
    std::cerr << "Error: " << (re2->regex_->error()) << std::endl;
    return tokenizers::Error::LoadFailure;
  }
}

} // namespace tokenizers
