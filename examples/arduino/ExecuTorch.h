/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Arduino's custom <new> header omits <exception>, which breaks
// std::bad_variant_access in <variant>. Include it first.
#include <exception>

#ifndef C10_USING_CUSTOM_GENERATED_MACROS
#define C10_USING_CUSTOM_GENERATED_MACROS
#endif
#ifndef ET_ENABLE_DEPRECATED_CONSTANT_BUFFER
#define ET_ENABLE_DEPRECATED_CONSTANT_BUFFER 0
#endif
#ifndef FLATBUFFERS_MAX_ALIGNMENT
#define FLATBUFFERS_MAX_ALIGNMENT 1024
#endif

// ExecuTorch's C++ needs a compiler newer than some Arduino cores ship. The
// Renesas cores are on arm-none-eabi-gcc 7.2.1 from 2017, which cannot resolve
// EValue::to()'s overload set and fails with pages of template errors that say
// nothing about the real cause. Fail with one line instead.
//
// Verified on GCC 12.2 (Zephyr SDK 0.16.8). 7.2.1 is known bad. The versions
// in between are untested, so 9 is a conservative floor rather than a measured
// one.
//
// library.properties still says architectures=zephyr, because every other
// Arduino core currently ships arm-none-eabi-gcc 7. Advertising the library to
// boards that cannot build it would trade a clear error for a wasted install.
// This guard is for anyone who copies the library in by hand, and for whenever
// a second core becomes usable.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 9
#error \
    "ExecuTorch needs GCC 9 or newer. This board core ships an older \
arm-none-eabi-gcc (Renesas cores are on 7.2.1 from 2017). Nothing in the \
library can work around it; the core's toolchain has to be updated."
#endif

// Several Arduino cores define abs/min/max/round as macros in Arduino.h. They
// expand inside c10's templates and inside <complex>, turning `inline T abs(T
// a)` into a syntax error. The Zephyr core happens not to define abs, which is
// the only reason this went unnoticed. Undefine them before the ExecuTorch
// headers; std::abs, std::min and std::max are available to sketches instead.
#undef abs
#undef min
#undef max
#undef round

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>
