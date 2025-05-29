/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pytorch/tokenizers/re2_regex.h>

namespace tokenizers {

Error Re2Regex::compile(const std::string& pattern) {
  regex_ = std::make_unique<re2::RE2>(pattern);
  // Warmup re2 as it is slow on the first run, void the return value as it's
  // not needed Refer to
  // https://github.com/google/re2/blob/6dcd83d60f7944926bfd308cc13979fc53dd69ca/re2/fuzzing/re2_fuzzer.cc#L136-L141
  (void)regex_->ReverseProgramSize();
  if (regex_->ok()) {
    return Error::Ok;
  } else {
    TK_LOG(
        Error,
        "Failed to compile regex: %s, error: %s",
        pattern.c_str(),
        regex_->error().c_str());
    return Error::RegexFailure;
  }
}

std::vector<Match> Re2Regex::find_all(const std::string& text) const {
  if (!regex_ || !regex_->ok()) {
    TK_LOG(Error, "Regex is not compiled or invalid, run compile() first");
    return std::vector<Match>{};
  }
  std::vector<Match> result;
  re2::StringPiece input(text);
  re2::StringPiece piece;

  const char* base = input.data();

  while (RE2::FindAndConsume(&input, *regex_, &piece)) {
    size_t start = piece.data() - base;
    result.push_back({start, start + piece.size()});
  }

  return result;
}

} // namespace tokenizers
