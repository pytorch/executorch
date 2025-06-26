/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Default implementation for create_regex, only using RE2 regex library.
// regex_lookahead.cpp has the implementation of create_regex with lookahead
// support, backed by PCRE2 and std::regex.

#include <pytorch/tokenizers/re2_regex.h>
#include <pytorch/tokenizers/regex.h>

namespace tokenizers {

// Default implementation that returns failure
static Result<std::unique_ptr<IRegex>> default_create_fallback_regex(
    const std::string& pattern) {
  (void)pattern;
  return tokenizers::Error::RegexFailure;
}

FallbackRegexFn fallback_regex = default_create_fallback_regex;

bool register_override_fallback_regex(FallbackRegexFn fn) {
  TK_LOG(Info, "Registering override fallback regex");
  fallback_regex = fn;
  return true;
}

FallbackRegexFn get_fallback_regex() {
  return fallback_regex;
}

std::string IRegex::escape(const std::string& input) {
  std::string result;
  result.reserve(input.size() * 2); // Reserve space for potential escaping

  for (char c : input) {
    // Escape regex special characters to treat them as literal strings
    if (c == '\\' || c == '^' || c == '$' || c == '.' || c == '|' || c == '?' ||
        c == '*' || c == '+' || c == '(' || c == ')' || c == '[' || c == ']' ||
        c == '{' || c == '}') {
      result += '\\';
    }
    result += c;
  }

  return result;
}

Result<std::unique_ptr<IRegex>> create_regex(const std::string& pattern) {
  // Try RE2 first
  auto re2 = std::make_unique<Re2Regex>();
  auto err = re2->compile("(" + pattern + ")");

  if (err == Error::Ok) {
    return static_cast<std::unique_ptr<IRegex>>(std::move(re2));
  }

  auto res = get_fallback_regex()(pattern);
  if (!res.ok()) {
    TK_LOG(
        Error,
        "RE2 doesn't support lookahead patterns. Link with `regex_lookahead` to enable support.");
  } else {
    return res;
  }

  return tokenizers::Error::RegexFailure;
}
} // namespace tokenizers
