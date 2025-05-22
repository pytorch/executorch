/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// A weak symbol for create_regex, only using RE2 regex library.
// regex_lookahead.cpp has the implementation of create_regex with lookahead
// support, backed by PCRE2 and std::regex.

#include <pytorch/tokenizers/re2_regex.h>
#include <pytorch/tokenizers/regex.h>

#include <iostream>

namespace tokenizers {

Result<std::unique_ptr<IRegex>> create_regex(const std::string& pattern) {
  // Try RE2 first
  auto re2 = std::make_unique<Re2Regex>("(" + pattern + ")");

  if (re2->regex_->ok()) {
    return static_cast<std::unique_ptr<IRegex>>(std::move(re2));
  }

  std::cerr << "RE2 failed to compile pattern: " << pattern << "\n";
  std::cerr << "Error: " << (re2->regex_->error()) << std::endl;

  if (re2->regex_->error_code() == re2::RE2::ErrorBadPerlOp) {
    auto res = create_fallback_regex(pattern);
    if (!res.ok()) {
      std::cerr
          << "RE2 doesn't support lookahead patterns. "
          << "Link with the lookahead-enabled version of this library to enable support."
          << std::endl;
    } else {
      return res;
    }
  }

  return tokenizers::Error::RegexFailure;
}

#ifdef _MSC_VER
#pragma weak create_fallback_regex
#endif // _MSC_VER
Result<std::unique_ptr<IRegex>> create_fallback_regex(
    const std::string& pattern) {
  (void)pattern;
  return tokenizers::Error::RegexFailure;
}

} // namespace tokenizers
