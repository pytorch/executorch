/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Default PAL implementations for Android system.
 */

// This cpp file will provide weak implementations of the symbols declared in
// Platform.h. Client users can strongly define any or all of the functions to
// override them.
#define ET_INTERNAL_PLATFORM_WEAKNESS ET_WEAK
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/platform.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <android/log.h>

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
  do {                                                              \
    if (!initialized) {                                             \
      __android_log_print(                                          \
          ANDROID_LOG_FATAL,                                        \
          "ExecuTorch",                                             \
          "ExecuTorch PAL must be initialized before call to %s()", \
          ET_FUNCTION);                                             \
    }                                                               \
  } while (0)

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
#ifdef _MSC_VER
#pragma weak et_pal_init
#endif // _MSC_VER
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
#ifdef _MSC_VER
#pragma weak et_pal_abort
#endif // _MSC_VER
ET_NORETURN void et_pal_abort(void) {
  std::abort();
}

/**
 * Return a monotonically non-decreasing timestamp in system ticks.
 *
 * @retval Timestamp value in system ticks.
 */
#ifdef _MSC_VER
#pragma weak et_pal_current_ticks
#endif // _MSC_VER
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
#ifdef _MSC_VER
#pragma weak et_pal_ticks_to_ns_multiplier
#endif // _MSC_VER
et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  // The system tick interval is 1 nanosecond, so the conversion factor is 1.
  return {1, 1};
}

/**
 * Emit a log message to adb logcat.
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
#ifdef _MSC_VER
#pragma weak et_pal_emit_log_message
#endif // _MSC_VER
void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    ET_UNUSED const char* filename,
    ET_UNUSED const char* function,
    ET_UNUSED size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  _ASSERT_PAL_INITIALIZED();

  int android_log_level = ANDROID_LOG_UNKNOWN;
  if (level == 'D') {
    android_log_level = ANDROID_LOG_DEBUG;
  } else if (level == 'I') {
    android_log_level = ANDROID_LOG_INFO;
  } else if (level == 'E') {
    android_log_level = ANDROID_LOG_ERROR;
  } else if (level == 'F') {
    android_log_level = ANDROID_LOG_FATAL;
  }

  __android_log_print(android_log_level, "ExecuTorch", "%s", message);
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
#ifdef _MSC_VER
#pragma weak et_pal_allocate
#endif // _MSC_VER
void* et_pal_allocate(size_t size) {
  return malloc(size);
}

/**
 * Frees memory allocated by et_pal_allocate().
 *
 * @param[in] ptr Pointer to memory to free. May be nullptr.
 */
#ifdef _MSC_VER
#pragma weak et_pal_free
#endif // _MSC_VER
void et_pal_free(void* ptr) {
  free(ptr);
}
