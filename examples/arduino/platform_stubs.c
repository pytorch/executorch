/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Platform stubs for Arduino LLEXT environment.
// Weak symbols — only used when the platform doesn't provide them.

#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>

__attribute__((weak)) void _Exit(int status) {
    (void)status;
    while (1) {}
}

// Intentionally discards output — last-resort stub for boards without fprintf.
__attribute__((weak)) int fprintf(FILE* stream, const char* fmt, ...) {
    (void)stream;
    (void)fmt;
    return 0;
}

#if defined(__ARM_EABI__)
// Use double intermediate to avoid the compiler lowering (long long)f back
// into a call to __aeabi_f2lz, which would cause infinite recursion.
__attribute__((weak)) long long __aeabi_f2lz(float f) {
    double d = (double)f;
    if (d < 0) {
        d = -d;
        uint32_t hi = (uint32_t)(d / 4294967296.0);
        uint32_t lo = (uint32_t)(d - (double)hi * 4294967296.0);
        return -(long long)(((uint64_t)hi << 32) | (uint64_t)lo);
    }
    uint32_t hi = (uint32_t)(d / 4294967296.0);
    uint32_t lo = (uint32_t)(d - (double)hi * 4294967296.0);
    return (long long)(((uint64_t)hi << 32) | (uint64_t)lo);
}
#endif
