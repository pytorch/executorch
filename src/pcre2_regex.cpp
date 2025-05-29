/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <vector>

#include <pytorch/tokenizers/pcre2_regex.h>

namespace tokenizers {

Error Pcre2Regex::compile(const std::string& pattern) {
  int error_code;
  PCRE2_SIZE error_offset;

  // Compile the pattern
  regex_ = pcre2_compile(
      reinterpret_cast<PCRE2_SPTR>(pattern.c_str()),
      pattern.length(),
      PCRE2_UCP | PCRE2_UTF, // Enable Unicode support and UTF-8 mode
      &error_code,
      &error_offset,
      nullptr);

  if (regex_ == nullptr) {
    PCRE2_UCHAR error_buffer[256];
    pcre2_get_error_message(error_code, error_buffer, sizeof(error_buffer));
    TK_LOG(
        Error,
        "PCRE2 compilation failed at offset %" PRId64 ": %s",
        static_cast<int64_t>(error_offset),
        error_buffer);
    return Error::RegexFailure;
  }

  // Create match data
  match_data_ = pcre2_match_data_create_from_pattern(regex_, nullptr);
  if (match_data_ == nullptr) {
    pcre2_code_free(regex_);
    regex_ = nullptr;
    TK_LOG(Error, "Failed to create PCRE2 match data");
    return Error::RegexFailure;
  }

  return Error::Ok;
}

Pcre2Regex::~Pcre2Regex() {
  if (match_data_) {
    pcre2_match_data_free(match_data_);
  }
  if (regex_) {
    pcre2_code_free(regex_);
  }
}

std::vector<Match> Pcre2Regex::find_all(const std::string& text) const {
  std::vector<Match> result;

  if (!regex_ || !match_data_) {
    TK_LOG(Error, "Regex is not compiled or invalid, run compile() first");
    return result;
  }

  PCRE2_SIZE* ovector;
  PCRE2_SPTR subject = reinterpret_cast<PCRE2_SPTR>(text.c_str());
  PCRE2_SIZE subject_length = text.length();
  PCRE2_SIZE offset = 0;

  while (offset < subject_length) {
    int rc = pcre2_match(
        regex_,
        subject,
        subject_length,
        offset,
        0, // Default options
        match_data_,
        nullptr);

    if (rc < 0) {
      if (rc == PCRE2_ERROR_NOMATCH) {
        break; // No more matches
      } else {
        // Error occurred
        PCRE2_UCHAR error_buffer[256];
        pcre2_get_error_message(rc, error_buffer, sizeof(error_buffer));
        std::cerr << "PCRE2 matching error: " << error_buffer << std::endl;
        break;
      }
    }

    ovector = pcre2_get_ovector_pointer(match_data_);

    // Add the match to the result
    result.push_back({ovector[0], ovector[1]});

    // Move to the next position after the match
    offset = ovector[1];

    // If the match was empty, move forward by one character to avoid infinite
    // loop
    if (ovector[0] == ovector[1]) {
      offset++;
    }
  }

  return result;
}

} // namespace tokenizers
