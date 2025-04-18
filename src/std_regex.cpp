/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pytorch/tokenizers/std_regex.h>
#include <regex>

namespace tokenizers {

StdRegex::StdRegex(const std::string& pattern) : regex_(pattern) {}

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
