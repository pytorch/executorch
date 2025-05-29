/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @lint-ignore-every LICENSELINT
 * @lint-ignore-every CLANGTIDY facebook-hte-StdRegexIsAwful
 */

#include <pytorch/tokenizers/std_regex.h>
#include <regex>

namespace tokenizers {

Error StdRegex::compile(const std::string& pattern) {
  try {
    regex_ = std::regex(pattern);
    return Error::Ok;
  } catch (std::regex_error) {
    TK_LOG(Error, "Failed to compile regex: %s", pattern.c_str());
    return Error::RegexFailure;
  }
}

std::vector<Match> StdRegex::find_all(const std::string& text) const {
  std::vector<Match> result;
  std::sregex_iterator iter(text.begin(), text.end(), regex_);
  std::sregex_iterator end;

  for (; iter != end; ++iter) {
    const auto& match = *iter;
    size_t start = match.position(1);
    result.push_back({start, start + match[1].length()});
  }

  return result;
}

} // namespace tokenizers
