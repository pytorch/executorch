/*
 * Copyright (c) 2025 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * C++ runtime stubs for bare-metal ARM targets.
 *
 * Provides minimal stub implementations for C++ runtime functions
 * required by ARM toolchains that are not needed in bare-metal contexts.
 * These are typically needed when linking GCC-compiled ExecuTorch libraries
 * with ARM Clang applications.
 */

#include <cstdint>

extern "C" {

__attribute__((weak)) int __aeabi_atexit(
    void* object, void (*destructor)(void*), void* dso_handle) {
    (void)object;
    (void)destructor;
    (void)dso_handle;
    return 0;
}

// GCC libstdc++ std::random_device stubs (mangled names for GCC's ABI).
// These provide deterministic values using a simple LCG —
// override with hardware RNG for security-sensitive applications.

// std::random_device::_M_getval()
__attribute__((weak)) unsigned int _ZNSt13random_device9_M_getvalEv() {
    static unsigned int seed = 0x12345678;
    seed = (seed * 1103515245U + 12345U) & 0x7FFFFFFF;
    return seed;
}

// std::random_device::_M_fini()
__attribute__((weak)) void _ZNSt13random_device7_M_finiEv() {}

// std::random_device::_M_init(std::string const&)
__attribute__((weak)) void
_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(
    const void* token) {
    (void)token;
}

__attribute__((weak)) char* getenv(const char* name) {
    (void)name;
    return nullptr;
}

} // extern "C"
