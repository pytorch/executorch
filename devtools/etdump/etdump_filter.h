/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <re2/re2.h>
#include <memory>

#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/platform.h>

namespace executorch::etdump {

using ::executorch::runtime::Result;
using std::string_view;

/**
 * ETDumpFilter is a class that filters intermediate output based on output's
 * name by full regex filtering, or delegate debug indices by range-based
 * filtering.
 *
 * Note that this filter supports up to MAX_REGEX_PATTERNS regex patterns with a
 * maximum length of MAX_PATTERN_LENGTH characters each.
 */

class ETDumpFilter : public ::executorch::runtime::EventTracerFilterBase {
 public:
  ETDumpFilter();
  ~ETDumpFilter() override = default;
  /**
   * Adds a regex pattern to the filter.
   *
   * @param[in] pattern A c string representing the regex pattern to be added.
   *
   * @return A Result<bool> indicating the success or failure of adding the
   * regex pattern.
   *         - True if the pattern is successfully added.
   *         - False if the pattern could not be added or if the maximum number
   * of patterns is exceeded.
   *         - An error code if number of pattern has reached to cap, or any
   * error occurs during regex compilation.
   */
  Result<bool> add_regex(string_view pattern);
  /**
   * Sets the range for the delegate debug index filtering as [start, end).
   * Note that this function will flush the existing range.
   *
   * @param[in] start The start of the range for filtering.
   * @param[in] end The end of the range for filtering.
   *
   * @return A Result<bool> indicating the success or failure of setting the
   * range.
   *         - True if the range is successfully set.
   *         - An error code if an error occurs.
   */
  Result<bool> set_debug_handle_range(size_t start, size_t end);

  /**
   * Filters events based on the given name or delegate debug index.
   *
   * Note that everytime only one of either the name or delegate_debug_index
   * should be passed in.
   *
   * @param[in] name A pointer to a string representing the `name` of the
   * event. If `delegate_debug_index` is not set to kUnsetDebugHandle, `name`
   * should be set to nullptr.
   *
   * @param[in] delegate_debug_index A DebugHandle representing the debug index
   * of the delegate. If `name` is not nullptr, this should be set to
   * kUnsetDebugHandle.
   *
   * @return A Result<bool> indicating whether the event matches the filter
   * criteria.
   *         - True if the event matches the filter.
   *         - False if the event does not match, or is unknown, or filter is
   * unset.
   *         - An error code if an error occurs during filtering.
   */
  Result<bool> filter(
      const char* name,
      ::executorch::runtime::DelegateDebugIntId delegate_debug_index) override;

  /**
   * Returns the number of regex patterns in the filter.
   */
  size_t get_n_regex() const;

 private:
  std::vector<std::unique_ptr<re2::RE2>> regex_patterns_;
  size_t range_start_ = 0;
  size_t range_end_ = 0;
  Result<bool> filter_name_(const char* name);
  Result<bool> filter_delegate_debug_index_(
      ::executorch::runtime::DelegateDebugIntId delegate_debug_index);
};

} // namespace executorch::etdump
