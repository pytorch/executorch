/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Compiler utility macros.
 */

#pragma once

// Compiler support checks.

#if !defined(__cplusplus)
#error ExecuTorch must be compiled using a C++ compiler.
#endif

#if __cplusplus < 201103L && (!defined(_MSC_VER) || _MSC_VER < 1600) && \
    (!defined(__GNUC__) ||                                              \
     (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__ < 40400))
#error ExecuTorch must use a compiler supporting at least the C++11 standard.
#error __cplusplus _MSC_VER __GNUC__  __GNUC_MINOR__  __GNUC_PATCHLEVEL__
#endif

/*
 * Define annotations aliasing C++ declaration attributes.
 * See all C++ declaration attributes here:
 *   https://en.cppreference.com/w/cpp/language/attributes
 *
 * Note that ExecuTorch supports a lower C++ standard version than all standard
 * attributes. Therefore, some annotations are defined using their Clang/GNU
 * counterparts.
 *
 * GNU attribute definitions:
 *   https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html
 */

#define __ET_NORETURN [[noreturn]]
#define __ET_NOINLINE __attribute__((noinline))
#define __ET_INLINE __attribute__((always_inline)) inline

#if defined(__GNUC__)

#define __ET_UNREACHABLE() __builtin_unreachable()

#elif defined(_MSC_VER)

#define __ET_UNREACHABLE() __assume(0)

#else // defined(__GNUC__)

#define __ET_UNREACHABLE() \
  while (1)                \
    ;

#endif // defined(__GNUC__)

#if (__cplusplus) >= 201703L

#define __ET_DEPRECATED [[deprecated]]
#define __ET_FALLTHROUGH [[fallthrough]]
#define __ET_NODISCARD [[nodiscard]]
#define __ET_UNUSED [[maybe_unused]]

#else

#define __ET_DEPRECATED __attribute__((deprecated))
#define __ET_FALLTHROUGH __attribute__((fallthrough))
#define __ET_NODISCARD __attribute__((warn_unused_result))
#define __ET_UNUSED __attribute__((unused))

#endif // (__cplusplus) >= 201703L

// UNLIKELY Macro
// example
// if __ET_UNLIKELY(a > 10 && b < 5) {
//   do something
// }
#if (__cplusplus) >= 202002L

#define __ET_LIKELY(expr) (expr) [[likely]]
#define __ET_UNLIKELY(expr) (expr) [[unlikely]]

#else

#define __ET_LIKELY(expr) (expr)
#define __ET_UNLIKELY(expr) (expr)

#endif // (__cplusplus) >= 202002L

/// Define a C symbol with weak linkage.
#define __ET_WEAK __attribute__((weak))

/**
 * Annotation marking a function as printf-like, providing compiler support
 * for format string argument checking.
 */
#define __ET_PRINTFLIKE(_string_index, _va_index) \
  __attribute__((format(printf, _string_index, _va_index)))

/// Name of the source file without a directory string.
#define __ET_SHORT_FILENAME (__builtin_strrchr("/" __FILE__, '/') + 1)

#ifndef __has_builtin
#define __has_builtin(x) (0)
#endif

#if __has_builtin(__builtin_LINE)
/// Current line as an integer.
#define __ET_LINE __builtin_LINE()
#else
#define __ET_LINE __LINE__
#endif // __has_builtin(__builtin_LINE)

#if __has_builtin(__builtin_FUNCTION)
/// Name of the current function as a const char[].
#define __ET_FUNCTION __builtin_FUNCTION()
#else
#define __ET_FUNCTION __FUNCTION__
#endif // __has_builtin(__builtin_FUNCTION)

// Whether the compiler supports GNU statement expressions.
// https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html
#ifndef __ET_HAVE_GNU_STATEMENT_EXPRESSIONS
#if (defined(__GNUC__) && __GNUC__ >= 3) || defined(__clang__)
#define __ET_HAVE_GNU_STATEMENT_EXPRESSIONS 1
#else
#define __ET_HAVE_GNU_STATEMENT_EXPRESSIONS 0
#endif
#endif // ifndef
