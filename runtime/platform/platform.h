/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Platform abstraction layer to allow individual platform libraries to override
 * symbols in ExecuTorch. PAL functions are defined as C functions so a platform
 * library implementer can use C in lieu of C++.
 */

#pragma once

// Use C-style includes so that C code can include this header.
#include <stddef.h>
#include <stdint.h>

#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/types.h>

/**
 * Clients should neither define nor use this macro. Used to optionally declare
 * the et_pal_*() functions as weak symbols.
 *
 * This provides a way to both:
 * - Include the header and define weak symbols (used by the internal default
 *   implementations)
 * - Include the header and define strong symbols (used by client overrides)
 */
#ifndef ET_INTERNAL_PLATFORM_WEAKNESS
#define ET_INTERNAL_PLATFORM_WEAKNESS
#endif

extern "C" {

/**
 * Represents the conversion ratio from system ticks to nanoseconds.
 * To convert, use nanoseconds = ticks * numerator / denominator.
 */
typedef struct {
  uint64_t numerator;
  uint64_t denominator;
} et_tick_ratio_t;

/**
 * Initialize the platform abstraction layer.
 *
 * This function should be called before any other function provided by the PAL
 * to initialize any global state. Typically overridden by PAL implementer.
 */
void et_pal_init(void) ET_INTERNAL_PLATFORM_WEAKNESS;

/**
 * Immediately abort execution, setting the device into an error state, if
 * available.
 */
ET_NORETURN void et_pal_abort(void) ET_INTERNAL_PLATFORM_WEAKNESS;

/**
 * Return a monotonically non-decreasing timestamp in system ticks.
 *
 * @retval Timestamp value in system ticks.
 */
et_timestamp_t et_pal_current_ticks(void) ET_INTERNAL_PLATFORM_WEAKNESS;

/**
 * Return the conversion rate from system ticks to nanoseconds as a fraction.
 * To convert a system ticks to nanoseconds, multiply the tick count by the
 * numerator and then divide by the denominator:
 *   nanoseconds = ticks * numerator / denominator
 *
 * The utility method executorch::runtime::ticks_to_ns(et_timestamp_t) can also
 * be used to perform the conversion for a given tick count. It is defined in
 * torch/executor/runtime/platform/clock.h.
 *
 * @retval The ratio of nanoseconds to system ticks.
 */
et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void)
    ET_INTERNAL_PLATFORM_WEAKNESS;

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
} et_pal_log_level_t;

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
    const char* function,
    size_t line,
    const char* message,
    size_t length) ET_INTERNAL_PLATFORM_WEAKNESS;

/**
 * NOTE: Core runtime code must not call this directly. It may only be called by
 * a MemoryAllocator wrapper.
 *
 * Allocates size bytes of memory.
 *
 * @param[in] size Number of bytes to allocate.
 * @returns the allocated memory, or nullptr on failure. Must be freed using
 *     et_pal_free().
 */
void* et_pal_allocate(size_t size) ET_INTERNAL_PLATFORM_WEAKNESS;

/**
 * Frees memory allocated by et_pal_allocate().
 *
 * @param[in] ptr Pointer to memory to free. May be nullptr.
 */
void et_pal_free(void* ptr) ET_INTERNAL_PLATFORM_WEAKNESS;

} // extern "C"
