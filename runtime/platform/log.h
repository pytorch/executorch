/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * ExecuTorch logging API.
 */

#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdlib>

#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/types.h>

// Set minimum log severity if compiler option is not provided.
#ifndef ET_MIN_LOG_LEVEL
#define ET_MIN_LOG_LEVEL Info
#endif // !defined(ET_MIN_LOG_LEVEL)

/*
 * Enable logging by default if compiler option is not provided.
 * This should facilitate less confusion for those developing ExecuTorch.
 */
#ifndef ET_LOG_ENABLED
#define ET_LOG_ENABLED 1
#endif // !defined(ET_LOG_ENABLED)

namespace executorch {
namespace runtime {

/**
 * Severity level of a log message. Must be ordered from lowest to highest
 * severity.
 */
enum class LogLevel : uint8_t {
  /**
   * Log messages provided for highly granular debuggability.
   *
   * Log messages using this severity are unlikely to be compiled by default
   * into most debug builds.
   */
  Debug,

  /**
   * Log messages providing information about the state of the system
   * for debuggability.
   */
  Info,

  /**
   * Log messages about errors within ExecuTorch during runtime.
   */
  Error,

  /**
   * Log messages that precede a fatal error. However, logging at this level
   * does not perform the actual abort, something else needs to.
   */
  Fatal,

  /**
   * Number of supported log levels, with values in [0, NumLevels).
   */
  NumLevels,
};

namespace internal {

/**
 * Get the current timestamp to construct a log event.
 *
 * @retval Monotonically non-decreasing timestamp in system ticks.
 */
et_timestamp_t get_log_timestamp();

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
__ET_PRINTFLIKE(6, 0)
void vlogf(
    LogLevel level,
    et_timestamp_t timestamp,
    const char* filename,
    const char* function,
    size_t line,
    const char* format,
    va_list args);

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
 */
__ET_PRINTFLIKE(6, 7)
inline void logf(
    LogLevel level,
    et_timestamp_t timestamp,
    const char* filename,
    const char* function,
    size_t line,
    const char* format,
    ...) {
#if ET_LOG_ENABLED
  va_list args;
  va_start(args, format);
  internal::vlogf(level, timestamp, filename, function, line, format, args);
  va_end(args);
#endif // ET_LOG_ENABLED
}

} // namespace internal

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::LogLevel;
} // namespace executor
} // namespace torch

#if ET_LOG_ENABLED

/**
 * Log a message at the given log severity level.
 *
 * @param[in] _level Log severity level.
 * @param[in] _format Log message format string.
 */
#define ET_LOG(_level, _format, ...)                                 \
  ({                                                                 \
    const auto _log_level = ::executorch::runtime::LogLevel::_level; \
    if (static_cast<uint32_t>(_log_level) >=                         \
        static_cast<uint32_t>(                                       \
            ::executorch::runtime::LogLevel::ET_MIN_LOG_LEVEL)) {    \
      const auto _timestamp =                                        \
          ::executorch::runtime::internal::get_log_timestamp();      \
      ::executorch::runtime::internal::logf(                         \
          _log_level,                                                \
          _timestamp,                                                \
          __ET_SHORT_FILENAME,                                       \
          __ET_FUNCTION,                                             \
          __ET_LINE,                                                 \
          _format,                                                   \
          ##__VA_ARGS__);                                            \
    }                                                                \
  })

#else // ET_LOG_ENABLED

/**
 * Log a message at the given log severity level.
 *
 * @param[in] _level Log severity level.
 * @param[in] _format Log message format string.
 */
#define ET_LOG(_level, _format, ...) ((void)0)

#endif // ET_LOG_ENABLED
