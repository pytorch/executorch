/*
 * Copyright (c) 2025 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bare-metal Platform Abstraction Layer (PAL) for ExecuTorch.
 *
 * Provides weak symbol implementations of the ExecuTorch PAL for bare-metal
 * ARM Cortex-M targets. The default posix.cpp uses std::chrono::steady_clock
 * which is not available in ARM bare-metal libc++.
 *
 * Applications can override these weak symbols with their own implementations
 * (e.g., using DWT cycle counter, SysTick, or other timing mechanisms).
 */

#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <cstdint>
#include <cstdio>

namespace torch {
namespace executor {

__attribute__((weak)) void et_pal_init(void) {}

__attribute__((weak)) et_timestamp_t et_pal_current_ticks(void) {
    return 0;
}

__attribute__((weak)) et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
    return {1, 1};
}

__attribute__((weak)) void et_pal_emit_log_message(
    et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    const char* function,
    size_t line,
    const char* message,
    size_t length) {
    (void)timestamp;
    (void)level;
    (void)filename;
    (void)function;
    (void)line;
    (void)length;
#ifdef EXECUTORCH_PAL_ENABLE_SEMIHOSTING
    fprintf(stderr, "%s\n", message);
#else
    (void)message;
#endif
}

__attribute__((weak)) __attribute__((noreturn)) void et_pal_abort(void) {
#ifdef EXECUTORCH_PAL_ENABLE_SEMIHOSTING
    fprintf(stderr, "ExecuTorch: abort() called\n");
#endif
    while (1) {
        __asm__ volatile("bkpt #0");
    }
}

} // namespace executor
} // namespace torch
