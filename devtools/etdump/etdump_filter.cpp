/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/devtools/etdump/etdump_filter.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/assert.h>

using ::executorch::runtime::DebugHandle;
using ::executorch::runtime::Error;

namespace executorch {
namespace etdump {

ETDumpFilter::ETDumpFilter()
    : regex_count_(0), range_start_(0), range_end_(0) {}

Result<bool> ETDumpFilter::add_regex(const char* pattern) {
  if (regex_count_ >= MAX_REGEX_PATTERNS) {
    return Error::OutOfResources; // Error code for exceeding max patterns
  }
  size_t len = strlen(pattern);
  if (len >= MAX_PATTERN_LENGTH) {
    return Error::InvalidArgument; // Pattern too long
  }
  strcpy(regex_patterns_[regex_count_], pattern);
  regex_patterns_[regex_count_][len] = '\0';
  regex_count_++;
  return true;
}

Result<bool> ETDumpFilter::set_debug_handle_range(size_t start, size_t end) {
  if (start >= end) {
    return Error::InvalidArgument; // Start is greater than end
  }
  if (start < 0 || end < 0) {
    return Error::InvalidArgument; // Start or end is negative
  }
  range_start_ = start;
  range_end_ = end;
  return true;
}

Result<bool> ETDumpFilter::filter_name_(const char* name) {
  if (name == nullptr) {
    return Error::InvalidArgument; // Name is null
  }
  if (regex_count_ == 0) {
    return true;
  }
  for (size_t i = 0; i < regex_count_; ++i) {
    if (RE2::FullMatch(name, regex_patterns_[i])) {
      return true;
    }
  }
  return false;
}
Result<bool> ETDumpFilter::filter_delegate_debug_index_(
    DebugHandle debug_handle) {
  if (debug_handle == runtime::kUnsetDebugHandle) {
    return Error::InvalidArgument; // Delegate debug index is unset
  }

  if (range_start_ == 0 && range_end_ == 0) {
    return true;
  }

  if (debug_handle < range_start_ || debug_handle >= range_end_) {
    return false;
  }

  return true;
}

Result<bool> ETDumpFilter::filter(
    const char* name,
    DebugHandle delegate_debug_index) {
  if (name) {
    return filter_name_(name);
  } else {
    return filter_delegate_debug_index_(delegate_debug_index);
  }
}

size_t ETDumpFilter::get_n_regex() const {
  return regex_count_;
}

} // namespace etdump
} // namespace executorch
