/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Fallback PAL implementations for POSIX-compatible systems.
 *
 * Note that this assumes that the platform defines the symbols used in this
 * file (like fprintf()), because this file will still be built even if the
 * functions are later overridden. When building for a platform that does not
 * provide the necessary symbols, clients can use Minimal.cpp instead, but they
 * will need to override all of the functions.
 */

// This cpp file will provide weak implementations of the symbols declared in
// Platform.h. Client users can strongly define any or all of the functions to
// override them.
#define ET_INTERNAL_PLATFORM_WEAKNESS ET_WEAK
#include <executorch/runtime/platform/platform.h>

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

#include <executorch/runtime/platform/compiler.h>

// The FILE* to write logs to.
#define ET_LOG_OUTPUT_FILE stderr

/**
 * On debug builds, ensure that `et_pal_init` has been called before
 * other PAL functions which depend on initialization.
 */
#ifdef NDEBUG

/**
 * Assert that the PAL has been initialized.
 */
#define _ASSERT_PAL_INITIALIZED() ((void)0)

#else // NDEBUG

/**
 * Assert that the PAL has been initialized.
 */
#define _ASSERT_PAL_INITIALIZED()                                   \
  ({                                                                \
    if (!initialized) {                                             \
      fprintf(                                                      \
          ET_LOG_OUTPUT_FILE,                                       \
          "ExecuTorch PAL must be initialized before call to %s()", \
          ET_FUNCTION);                                             \
      fflush(ET_LOG_OUTPUT_FILE);                                   \
      et_pal_abort();                                               \
    }                                                               \
  })

#endif // NDEBUG

/// Start time of the system (used to zero the system timestamp).
static std::chrono::time_point<std::chrono::steady_clock> systemStartTime;

/// Flag set to true if the PAL has been successfully initialized.
static bool initialized = false;

/**
 * Initialize the platform abstraction layer.
 *
 * This function should be called before any other function provided by the PAL
 * to initialize any global state. Typically overridden by PAL implementer.
 */
void et_pal_init(void) {
  if (initialized) {
    return;
  }

  systemStartTime = std::chrono::steady_clock::now();
  initialized = true;
}

/**
 * Immediately abort execution, setting the device into an error state, if
 * available.
 */
ET_NORETURN void et_pal_abort(void) {
  std::abort();
}

/**
 * Return a monotonically non-decreasing timestamp in system ticks.
 *
 * @retval Timestamp value in system ticks.
 */
et_timestamp_t et_pal_current_ticks(void) {
  _ASSERT_PAL_INITIALIZED();
  auto systemCurrentTime = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             systemCurrentTime - systemStartTime)
      .count();
}

/**
 * Return the conversion rate from system ticks to nanoseconds, as a fraction.
 * To convert an interval from system ticks to nanoseconds, multiply the tick
 * count by the numerator and then divide by the denominator:
 *   nanoseconds = ticks * numerator / denominator
 *
 * @retval The ratio of nanoseconds to system ticks.
 */
et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  // The system tick interval is 1 nanosecond, so the conversion factor is 1.
  return {1, 1};
}

/**
 * Emit a log message via platform output (serial port, console, etc).
 *
 * @param[in] timestamp Timestamp of the log event in system ticks since boot.
 * @param[in] level Severity level of the message. Must be a printable 7-bit
 *     ASCII uppercase letter.
 * @param[in] filename Name of the file that created the log event.
 * @param[in] function Name of the function that created the log event.
 * @param[in] line Line in the source file where the log event was created.
 * @param[in] message Message string to log.
 * @param[in] length Message string length.
 */
void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  _ASSERT_PAL_INITIALIZED();

  // Not all platforms have ticks == nanoseconds, but this one does.
  timestamp /= 1000; // To microseconds
  unsigned long int us = timestamp % 1000000;
  timestamp /= 1000000; // To seconds
  unsigned int sec = timestamp % 60;
  timestamp /= 60; // To minutes
  unsigned int min = timestamp % 60;
  timestamp /= 60; // To hours
  unsigned int hour = timestamp;

  // Use a format similar to glog and folly::logging, except:
  // - Print time since et_pal_init since we don't have wall time
  // - Don't include the thread ID, to avoid adding a threading dependency
  // - Add the string "executorch:" to make the logs more searchable
  //
  // Clients who want to change the format or add other fields can override this
  // weak implementation of et_pal_emit_log_message.
  fprintf(
      ET_LOG_OUTPUT_FILE,
      "%c %02u:%02u:%02u.%06lu executorch:%s:%zu] %s\n",
      level,
      hour,
      min,
      sec,
      us,
      filename,
      line,
      message);
  fflush(ET_LOG_OUTPUT_FILE);
}

/**
 * NOTE: Core runtime code must not call this directly. It may only be called by
 * a MemoryAllocator wrapper.
 *
 * Allocates size bytes of memory via malloc.
 *
 * @param[in] size Number of bytes to allocate.
 * @returns the allocated memory, or nullptr on failure. Must be freed using
 *     et_pal_free().
 */
void* et_pal_allocate(size_t size) {
  return malloc(size);
}

/**
 * Frees memory allocated by et_pal_allocate().
 *
 * @param[in] ptr Pointer to memory to free. May be nullptr.
 */
void et_pal_free(void* ptr) {
  free(ptr);
}
