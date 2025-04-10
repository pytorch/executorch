/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/devtools/etdump/etdump_filter.h>

#include <executorch/runtime/core/error.h>

using ::executorch::runtime::DelegateDebugIntId;
using ::executorch::runtime::Error;
using ::executorch::runtime::kUnsetDelegateDebugIntId;

namespace executorch {
namespace etdump {

ETDumpFilter::ETDumpFilter() = default;

Result<bool> ETDumpFilter::add_regex(string_view pattern) {
  auto regex = std::make_unique<re2::RE2>(pattern.data());
  if (!regex->ok()) {
    return Error::InvalidArgument; // Error during regex compilation
  }
  regex_patterns_.emplace_back(std::move(regex));
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
    return Error::InvalidArgument;
  }
  if (regex_patterns_.empty()) {
    return true;
  }
  for (const auto& regex : regex_patterns_) {
    if (RE2::FullMatch(name, *regex)) {
      return true;
    }
  }
  return false;
}

Result<bool> ETDumpFilter::filter_delegate_debug_index_(
    DelegateDebugIntId debug_handle) {
  if (debug_handle == kUnsetDelegateDebugIntId) {
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
    DelegateDebugIntId delegate_debug_index) {
  if ((name == nullptr) == (delegate_debug_index == kUnsetDelegateDebugIntId)) {
    return Error::InvalidArgument; // Name and delegate debug index should be
                                   // both set or unset
  }

  if (name) {
    return filter_name_(name);
  } else {
    return filter_delegate_debug_index_(delegate_debug_index);
  }
}

size_t ETDumpFilter::get_n_regex() const {
  return regex_patterns_.size();
}

} // namespace etdump
} // namespace executorch
