/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/platform/log.h>

#include <cstdio>

#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/platform.h>

namespace executorch {
namespace runtime {
namespace internal {

/**
 * Get the current timestamp to construct a log event.
 *
 * @retval Monotonically non-decreasing timestamp in system ticks.
 */
et_timestamp_t get_log_timestamp() {
  return et_pal_current_ticks();
}

// Double-check that the log levels are ordered from lowest to highest severity.
static_assert(LogLevel::Debug < LogLevel::Info, "");
static_assert(LogLevel::Info < LogLevel::Error, "");
static_assert(LogLevel::Error < LogLevel::Fatal, "");
static_assert(LogLevel::Fatal < LogLevel::NumLevels, "");

/**
 * Maps LogLevel values to et_pal_log_level_t values.
 *
 * We don't share values because LogLevel values need to be ordered by severity,
 * and et_pal_log_level_t values need to be printable characters.
 */
static constexpr et_pal_log_level_t kLevelToPal[size_t(LogLevel::NumLevels)] = {
    et_pal_log_level_t::kDebug,
    et_pal_log_level_t::kInfo,
    et_pal_log_level_t::kError,
    et_pal_log_level_t::kFatal,
};

// Double-check that the indices are correct.
static_assert(
    kLevelToPal[size_t(LogLevel::Debug)] == et_pal_log_level_t::kDebug,
    "");
static_assert(
    kLevelToPal[size_t(LogLevel::Info)] == et_pal_log_level_t::kInfo,
    "");
static_assert(
    kLevelToPal[size_t(LogLevel::Error)] == et_pal_log_level_t::kError,
    "");
static_assert(
    kLevelToPal[size_t(LogLevel::Fatal)] == et_pal_log_level_t::kFatal,
    "");

/**
 * Log a string message.
 *
 * Note: This is an internal function. Use the `ET_LOG` macro instead.
 *
 * @param[in] level Log severity level.
 * @param[in] timestamp Timestamp (in system ticks) of the log event.
 * @param[in] filename Name of the source file creating the log event.
 * @param[in] function Name of the function creating the log event.
 * @param[in] line Source file line of the caller.
 * @param[in] format Format string.
 * @param[in] args Variable argument list.
 */
void vlogf(
    __ET_UNUSED LogLevel level,
    et_timestamp_t timestamp,
    const char* filename,
    __ET_UNUSED const char* function,
    size_t line,
    const char* format,
    va_list args) {
#if ET_LOG_ENABLED

  // Maximum length of a log message.
  static constexpr size_t kMaxLogMessageLength = 256;
  char buf[kMaxLogMessageLength];
  size_t len = vsnprintf(buf, kMaxLogMessageLength, format, args);
  if (len >= kMaxLogMessageLength - 1) {
    buf[kMaxLogMessageLength - 2] = '$';
    len = kMaxLogMessageLength - 1;
  }
  buf[kMaxLogMessageLength - 1] = 0;

  et_pal_log_level_t pal_level =
      (int(level) >= 0 && level < LogLevel::NumLevels)
      ? kLevelToPal[size_t(level)]
      : et_pal_log_level_t::kUnknown;

  et_pal_emit_log_message(
      timestamp, pal_level, filename, function, line, buf, len);

#endif // ET_LOG_ENABLED
}

} // namespace internal
} // namespace runtime
} // namespace executorch
