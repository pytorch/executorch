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

/*
 * Compiler support checks. Follows the logic used by pytorch/c10/util/C++17.h
 * but may support older versions.
 */

// https://gcc.gnu.org/projects/cxx-status.html#cxx17
#if !defined(__clang__) && !defined(_MSC_VER) && defined(__GNUC__) && \
    __GNUC__ < 7
#error \
    "You're trying to build ExecuTorch with a too old version of GCC. We need GCC 7 or later."
#endif

// https://clang.llvm.org/cxx_status.html#cxx17
#if defined(__clang__) && __clang_major__ < 5
#error \
    "You're trying to build ExecuTorch with a too old version of Clang. We need Clang 5 or later."
#endif

#if (defined(_MSC_VER) && (!defined(_MSVC_LANG) || _MSVC_LANG < 201703L)) || \
    (!defined(_MSC_VER) && __cplusplus < 201703L)
#error "You need C++17 to compile ExecuTorch"
#endif

#if defined(_WIN32) && (defined(min) || defined(max))
#error \
    "Macro clash with min and max -- define NOMINMAX when compiling your program on Windows"
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

#define ET_NORETURN [[noreturn]]
#define ET_NOINLINE __attribute__((noinline))
#define ET_INLINE __attribute__((always_inline)) inline
#define ET_INLINE_ATTRIBUTE __attribute__((always_inline))

#if defined(__GNUC__)

#define ET_UNREACHABLE() __builtin_unreachable()

#elif defined(_MSC_VER)

#define ET_UNREACHABLE() __assume(0)

#else // defined(__GNUC__)

#define ET_UNREACHABLE() \
  while (1)              \
    ;

#endif // defined(__GNUC__)

#if (__cplusplus) >= 201703L

#define ET_DEPRECATED [[deprecated]]
#define ET_EXPERIMENTAL \
  [[deprecated("This API is experimental and may change without notice.")]]
#define ET_FALLTHROUGH [[fallthrough]]
#define ET_NODISCARD [[nodiscard]]
#define ET_UNUSED [[maybe_unused]]

#else

#define ET_DEPRECATED __attribute__((deprecated))
#define ET_EXPERIMENTAL \
  __attribute__((       \
      deprecated("This API is experimental and may change without notice.")))
#define ET_FALLTHROUGH __attribute__((fallthrough))
#define ET_NODISCARD __attribute__((warn_unused_result))
#define ET_UNUSED __attribute__((unused))

#endif // (__cplusplus) >= 201703L

// UNLIKELY Macro
// example
// if ET_UNLIKELY(a > 10 && b < 5) {
//   do something
// }
#if (__cplusplus) >= 202002L

#define ET_LIKELY(expr) (expr) [[likely]]
#define ET_UNLIKELY(expr) (expr) [[unlikely]]

#else

#define ET_LIKELY(expr) (expr)
#define ET_UNLIKELY(expr) (expr)

#endif // (__cplusplus) >= 202002L

/// Define a C symbol with weak linkage.
#define ET_WEAK __attribute__((weak))

/**
 * Annotation marking a function as printf-like, providing compiler support
 * for format string argument checking.
 */
#define ET_PRINTFLIKE(_string_index, _va_index) \
  __attribute__((format(printf, _string_index, _va_index)))

/// Name of the source file without a directory string.
#define ET_SHORT_FILENAME (__builtin_strrchr("/" __FILE__, '/') + 1)

#ifndef __has_builtin
#define __has_builtin(x) (0)
#endif

#if __has_builtin(__builtin_LINE)
/// Current line as an integer.
#define ET_LINE __builtin_LINE()
#else
#define ET_LINE __LINE__
#endif // __has_builtin(__builtin_LINE)

#if __has_builtin(__builtin_FUNCTION)
/// Name of the current function as a const char[].
#define ET_FUNCTION __builtin_FUNCTION()
#else
#define ET_FUNCTION __FUNCTION__
#endif // __has_builtin(__builtin_FUNCTION)

// Whether the compiler supports GNU statement expressions.
// https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html
#ifndef ET_HAVE_GNU_STATEMENT_EXPRESSIONS
#if (defined(__GNUC__) && __GNUC__ >= 3) || defined(__clang__)
#define ET_HAVE_GNU_STATEMENT_EXPRESSIONS 1
#else
#define ET_HAVE_GNU_STATEMENT_EXPRESSIONS 0
#endif
#endif // ifndef

// Define size_t and ssize_t.
#ifndef _WIN32
#include <sys/types.h>
#else
#include <stddef.h>
using ssize_t = ptrdiff_t;
#endif

// DEPRECATED: Use the non-underscore-prefixed versions instead.
// TODO(T199005537): Remove these once all users have stopped using them.
#define __ET_DEPRECATED ET_DEPRECATED
#define __ET_FALLTHROUGH ET_FALLTHROUGH
#define __ET_FUNCTION ET_FUNCTION
#define __ET_HAVE_GNU_STATEMENT_EXPRESSIONS ET_HAVE_GNU_STATEMENT_EXPRESSIONS
#define __ET_INLINE ET_INLINE
#define __ET_LIKELY ET_LIKELY
#define __ET_LINE ET_LINE
#define __ET_NODISCARD ET_NODISCARD
#define __ET_NOINLINE ET_NOINLINE
#define __ET_NORETURN ET_NORETURN
#define __ET_PRINTFLIKE ET_PRINTFLIKE
#define __ET_SHORT_FILENAME ET_SHORT_FILENAME
#define __ET_UNLIKELY ET_UNLIKELY
#define __ET_UNREACHABLE ET_UNREACHABLE
#define __ET_UNUSED ET_UNUSED
#define __ET_WEAK ET_WEAK
