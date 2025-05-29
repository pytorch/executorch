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
 * @brief Implementation of the fallback regex function with lookahead support.
 *        Falls back to PCRE2 if RE2 rejects the pattern due to lookahead.
 *        Falls back to std::regex if PCRE2 also fails.
 */
Result<std::unique_ptr<IRegex>> create_fallback_regex(
    const std::string& pattern) {
  TK_LOG(Info, "Creating PCRE2 regex");
  auto pcre2 = std::make_unique<Pcre2Regex>();
  auto err = pcre2->compile(pattern);

  if (err == Error::Ok) {
    return static_cast<std::unique_ptr<IRegex>>(std::move(pcre2));
  }

  // If PCRE2 also fails, fall back to std::regex
  auto std_regex = std::make_unique<StdRegex>();
  err = std_regex->compile(pattern);
  if (err == Error::Ok) {
    TK_LOG(
        Info, "PCRE2 failed to compile pattern, falling back to std::regex.");
    return static_cast<std::unique_ptr<IRegex>>(std::move(std_regex));
  }

  return tokenizers::Error::RegexFailure;
}

static bool registered =
    register_override_fallback_regex(create_fallback_regex);

} // namespace tokenizers
