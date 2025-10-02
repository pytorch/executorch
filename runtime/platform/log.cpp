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
  return pal_current_ticks();
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

#if ET_LOG_ENABLED
static size_t get_valid_utf8_prefix_length(const char* bytes, size_t length) {
  if (!bytes || length == 0) {
    return 0;
  }
  const auto* data = reinterpret_cast<const unsigned char*>(bytes);
  size_t i = length;
  while (i > 0 && (data[i - 1] & 0xC0) == 0x80) {
    --i;
  }
  if (i == 0) {
    return 0;
  }
  const size_t lead_pos = i - 1;
  const unsigned char lead = data[lead_pos];
  size_t need = 0;

  if (lead < 0x80) {
    need = 1;
  } else if ((lead & 0xE0) == 0xC0) {
    need = 2;
  } else if ((lead & 0xF0) == 0xE0) {
    need = 3;
  } else if ((lead & 0xF8) == 0xF0) {
    need = 4;
  } else {
    return lead_pos;
  }
  return length - lead_pos == need ? length : lead_pos;
}
#endif // ET_LOG_ENABLED

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
    ET_UNUSED LogLevel level,
    et_timestamp_t timestamp,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* format,
    va_list args) {
#if ET_LOG_ENABLED

  // Maximum length of a log message.
  static constexpr size_t kMaxLogMessageLength = 256;
  char buffer[kMaxLogMessageLength];

  const auto write_count =
      vsnprintf(buffer, kMaxLogMessageLength, format, args);
  const size_t used_length = (write_count < 0)
      ? 0
      : (write_count >= static_cast<int>(kMaxLogMessageLength)
             ? kMaxLogMessageLength - 1
             : static_cast<size_t>(write_count));
  const auto valid_length = get_valid_utf8_prefix_length(buffer, used_length);
  buffer[valid_length] = '\0';

  const auto pal_level = (level < LogLevel::NumLevels)
      ? kLevelToPal[size_t(level)]
      : et_pal_log_level_t::kUnknown;

  pal_emit_log_message(
      timestamp, pal_level, filename, function, line, buffer, valid_length);

#endif // ET_LOG_ENABLED
}

} // namespace internal
} // namespace runtime
} // namespace executorch
