/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Tokenizers logging API, adopted from ExecuTorch.
 */

#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Set minimum log severity if compiler option is not provided.
#ifndef TK_MIN_LOG_LEVEL
#define TK_MIN_LOG_LEVEL Info
#endif // !defined(TK_MIN_LOG_LEVEL)

/*
 * Enable logging by default if compiler option is not provided.
 * This should facilitate less confusion for those developing Tokenizers.
 */
#ifndef TK_LOG_ENABLED
#define TK_LOG_ENABLED 1
#endif // !defined(TK_LOG_ENABLED)

/**
 * Annotation marking a function as printf-like, providing compiler support
 * for format string argument checking.
 */
#ifdef _MSC_VER
#include <sal.h>
#define TK_PRINTFLIKE(_string_index, _va_index) _Printf_format_string_
#else
#define TK_PRINTFLIKE(_string_index, _va_index) \
  __attribute__((format(printf, _string_index, _va_index)))
#endif

/// Define a C symbol with weak linkage.
#ifdef _MSC_VER
// There currently doesn't seem to be a great way to do this in Windows and
// given that weak linkage is not really critical on Windows, we'll just leave
// it as a stub.
#define TK_WEAK
#else
#define TK_WEAK __attribute__((weak))
#endif

#ifndef __has_builtin
#define __has_builtin(x) (0)
#endif

#if __has_builtin(__builtin_strrchr)
/// Name of the source file without a directory string.
#define TK_SHORT_FILENAME (__builtin_strrchr("/" __FILE__, '/') + 1)
#else
#define TK_SHORT_FILENAME __FILE__
#endif

#if __has_builtin(__builtin_LINE)
/// Current line as an integer.
#define TK_LINE __builtin_LINE()
#else
#define TK_LINE __LINE__
#endif // __has_builtin(__builtin_LINE)

#if __has_builtin(__builtin_FUNCTION)
/// Name of the current function as a const char[].
#define TK_FUNCTION __builtin_FUNCTION()
#else
#define TK_FUNCTION __FUNCTION__
#endif // __has_builtin(__builtin_FUNCTION)

/**
 * Clients should neither define nor use this macro. Used to optionally declare
 * the tk_pal_*() functions as weak symbols.
 *
 * This provides a way to both:
 * - Include the header and define weak symbols (used by the internal default
 *   implementations)
 * - Include the header and define strong symbols (used by client overrides)
 */
#ifndef TK_INTERNAL_PLATFORM_WEAKNESS
#define TK_INTERNAL_PLATFORM_WEAKNESS TK_WEAK
#endif // !defined(TK_INTERNAL_PLATFORM_WEAKNESS)

// TODO: making an assumption that we have stderr
#define TK_LOG_OUTPUT_FILE stderr

namespace tokenizers {

extern "C" {
/**
 * Severity level of a log message. Values must map to printable 7-bit ASCII
 * uppercase letters.
 */
typedef enum {
  kDebug = 'D',
  kInfo = 'I',
  kError = 'E',
  kFatal = 'F',
  kUnknown = '?', // Exception to the "uppercase letter" rule.
} tk_pal_log_level_t;

/**
 * Emit a log message via platform output (serial port, console, etc).
 *
 * @param[in] level Severity level of the message. Must be a printable 7-bit
 *     ASCII uppercase letter.
 * @param[in] filename Name of the file that created the log event.
 * @param[in] function Name of the function that created the log event.
 * @param[in] line Line in the source file where the log event was created.
 * @param[in] message Message string to log.
 * @param[in] length Message string length.
 */
inline void TK_INTERNAL_PLATFORM_WEAKNESS tk_pal_emit_log_message(
    tk_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    size_t length) {
  // Use a format similar to glog and folly::logging, except:
  // - Print time since et_pal_init since we don't have wall time
  // - Don't include the thread ID, to avoid adding a threading dependency
  // - Add the string "tokenizers:" to make the logs more searchable
  //
  // Clients who want to change the format or add other fields can override this
  // weak implementation of et_pal_emit_log_message.
  fprintf(
      TK_LOG_OUTPUT_FILE,
      "%c tokenizers:%s:%zu] %s\n",
      level,
      filename,
      line,
      message);
  fflush(TK_LOG_OUTPUT_FILE);
}

} // extern "C"
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
   * Log messages about errors within Tokenizers during runtime.
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
 * Maps LogLevel values to et_pal_log_level_t values.
 *
 * We don't share values because LogLevel values need to be ordered by severity,
 * and et_pal_log_level_t values need to be printable characters.
 */
static constexpr tk_pal_log_level_t kLevelToPal[size_t(LogLevel::NumLevels)] = {
    tk_pal_log_level_t::kDebug,
    tk_pal_log_level_t::kInfo,
    tk_pal_log_level_t::kError,
    tk_pal_log_level_t::kFatal,
};

// TODO: add timestamp support

/**
 * Log a string message.
 *
 * Note: This is an internal function. Use the `ET_LOG` macro instead.
 *
 * @param[in] level Log severity level.
 * @param[in] filename Name of the source file creating the log event.
 * @param[in] function Name of the function creating the log event.
 * @param[in] line Source file line of the caller.
 * @param[in] format Format string.
 * @param[in] args Variable argument list.
 */
TK_PRINTFLIKE(5, 0)
inline void vlogf(
    LogLevel level,
    const char* filename,
    const char* function,
    size_t line,
    const char* format,
    va_list args) {
  // Maximum length of a log message.
  static constexpr size_t kMaxLogMessageLength = 256;
  char buf[kMaxLogMessageLength];
  size_t len = vsnprintf(buf, kMaxLogMessageLength, format, args);
  if (len >= kMaxLogMessageLength - 1) {
    buf[kMaxLogMessageLength - 2] = '$';
    len = kMaxLogMessageLength - 1;
  }
  buf[kMaxLogMessageLength - 1] = 0;

  tk_pal_log_level_t pal_level =
      (int(level) >= 0 && level < LogLevel::NumLevels)
      ? kLevelToPal[size_t(level)]
      : tk_pal_log_level_t::kUnknown;

  tk_pal_emit_log_message(pal_level, filename, function, line, buf, len);
}

/**
 * Log a string message.
 *
 * Note: This is an internal function. Use the `ET_LOG` macro instead.
 *
 * @param[in] level Log severity level.
 * @param[in] filename Name of the source file creating the log event.
 * @param[in] function Name of the function creating the log event.
 * @param[in] line Source file line of the caller.
 * @param[in] format Format string.
 */
TK_PRINTFLIKE(5, 6)
inline void logf(
    LogLevel level,
    const char* filename,
    const char* function,
    size_t line,
    const char* format,
    ...) {
#if TK_LOG_ENABLED
  va_list args;
  va_start(args, format);
  internal::vlogf(level, filename, function, line, format, args);
  va_end(args);
#endif // TK_LOG_ENABLED
}

} // namespace internal

} // namespace tokenizers

#if TK_LOG_ENABLED

/**
 * Log a message at the given log severity level.
 *
 * @param[in] _level Log severity level.
 * @param[in] _format Log message format string.
 */
#define TK_LOG(_level, _format, ...)                                       \
  do {                                                                     \
    const auto _log_level = ::tokenizers::LogLevel::_level;                \
    if (static_cast<uint32_t>(_log_level) >=                               \
        static_cast<uint32_t>(::tokenizers::LogLevel::TK_MIN_LOG_LEVEL)) { \
      ::tokenizers::internal::logf(                                        \
          _log_level,                                                      \
          TK_SHORT_FILENAME,                                               \
          TK_FUNCTION,                                                     \
          TK_LINE,                                                         \
          _format,                                                         \
          ##__VA_ARGS__);                                                  \
    }                                                                      \
  } while (0)
#else // TK_LOG_ENABLED

/**
 * Log a message at the given log severity level.
 *
 * @param[in] _level Log severity level.
 * @param[in] _format Log message format string.
 */
#define TK_LOG(_level, _format, ...) ((void)0)

#endif // TK_LOG_ENABLED
