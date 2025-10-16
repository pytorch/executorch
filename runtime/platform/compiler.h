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

// -----------------------------------------------------------------------------
// Compiler version checks
// -----------------------------------------------------------------------------

// GCC version check
#if !defined(__clang__) && !defined(_MSC_VER) && defined(__GNUC__) && __GNUC__ < 7
#error "You're trying to build ExecuTorch with a too old version of GCC. We need GCC 7 or later."
#endif

// Clang version check
#if defined(__clang__) && __clang_major__ < 5
#error "You're trying to build ExecuTorch with a too old version of Clang. We need Clang 5 or later."
#endif

// C++17 check
#if (defined(_MSC_VER) && (!defined(_MSVC_LANG) || _MSVC_LANG < 201703L)) || \
    (!defined(_MSC_VER) && __cplusplus < 201703L)
#error "You need C++17 to compile ExecuTorch"
#endif

// Windows min/max macro clash
#if defined(_MSC_VER) && (defined(min) || defined(max))
#error "Macro clash with min and max -- define NOMINMAX when compiling your program on Windows"
#endif

// -----------------------------------------------------------------------------
// Attribute macros
// -----------------------------------------------------------------------------

// [[noreturn]]
#define ET_NORETURN [[noreturn]]

// [[deprecated]]
#define ET_DEPRECATED [[deprecated]]
#define ET_EXPERIMENTAL [[deprecated("This API is experimental and may change without notice.")]]

// [[fallthrough]]
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 7)
#define ET_FALLTHROUGH [[fallthrough]]
#else
#define ET_FALLTHROUGH
#endif

// [[nodiscard]]
#define ET_NODISCARD [[nodiscard]]

// [[maybe_unused]]
#define ET_UNUSED [[maybe_unused]]

// Inline/NoInline
#if defined(_MSC_VER)
#define ET_NOINLINE __declspec(noinline)
#define ET_INLINE __forceinline
#define ET_INLINE_ATTRIBUTE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ET_NOINLINE __attribute__((noinline))
#define ET_INLINE __attribute__((always_inline)) inline
#define ET_INLINE_ATTRIBUTE __attribute__((always_inline))
#else
#define ET_NOINLINE
#define ET_INLINE inline
#define ET_INLINE_ATTRIBUTE
#endif

// Unreachable
#if defined(__GNUC__) || defined(__clang__)
#define ET_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#define ET_UNREACHABLE() __assume(0)
#else
#define ET_UNREACHABLE() do {} while (1)
#endif

// Likely/Unlikely
#if (__cplusplus) >= 202002L
#define ET_LIKELY(expr) (expr) [[likely]]
#define ET_UNLIKELY(expr) (expr) [[unlikely]]
#else
#define ET_LIKELY(expr) (expr)
#define ET_UNLIKELY(expr) (expr)
#endif

// Weak linkage
#if defined(_MSC_VER)
// No weak linkage in MSVC
#define ET_WEAK
#elif defined(__GNUC__) || defined(__clang__)
#define ET_WEAK __attribute__((weak))
#else
#define ET_WEAK
#endif

// Printf-like format checking
#if defined(_MSC_VER)
#include <sal.h>
#define ET_PRINTFLIKE(_string_index, _va_index) _Printf_format_string_
#elif defined(__GNUC__) || defined(__clang__)
#define ET_PRINTFLIKE(_string_index, _va_index) \
  __attribute__((format(printf, _string_index, _va_index)))
#else
#define ET_PRINTFLIKE(_string_index, _va_index)
#endif

// -----------------------------------------------------------------------------
// Builtin/Source location helpers
// -----------------------------------------------------------------------------

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if __has_builtin(__builtin_strrchr)
#define ET_SHORT_FILENAME (__builtin_strrchr("/" __FILE__, '/') + 1)
#else
#define ET_SHORT_FILENAME __FILE__
#endif

#if __has_builtin(__builtin_LINE)
#define ET_LINE __builtin_LINE()
#else
#define ET_LINE __LINE__
#endif

#if __has_builtin(__builtin_FUNCTION)
#define ET_FUNCTION __builtin_FUNCTION()
#else
#define ET_FUNCTION __FUNCTION__
#endif

// -----------------------------------------------------------------------------
// Format specifiers for size_t/ssize_t
// -----------------------------------------------------------------------------

#if defined(__XTENSA__)
#define ET_PRIsize_t "lu"
#define ET_PRIssize_t "ld"
#else
#define ET_PRIsize_t "zu"
#define ET_PRIssize_t "zd"
#endif

// -----------------------------------------------------------------------------
// GNU statement expressions
// -----------------------------------------------------------------------------

#ifndef ET_HAVE_GNU_STATEMENT_EXPRESSIONS
#if (defined(__GNUC__) && __GNUC__ >= 3) || defined(__clang__)
#define ET_HAVE_GNU_STATEMENT_EXPRESSIONS 1
#else
#define ET_HAVE_GNU_STATEMENT_EXPRESSIONS 0
#endif
#endif

// -----------------------------------------------------------------------------
// ssize_t definition
// -----------------------------------------------------------------------------

#ifndef _MSC_VER
#include <sys/types.h>
#else
#include <stddef.h>
using ssize_t = ptrdiff_t;
#endif

// -----------------------------------------------------------------------------
// Exception support
// -----------------------------------------------------------------------------

#ifdef __EXCEPTIONS
#define ET_HAS_EXCEPTIONS 1
#elif defined(_MSC_VER) && defined(_HAS_EXCEPTIONS) && _HAS_EXCEPTIONS
#define ET_HAS_EXCEPTIONS 1
#else
#define ET_HAS_EXCEPTIONS 0
#endif

// -----------------------------------------------------------------------------
// Deprecated legacy macros (to be removed)
// -----------------------------------------------------------------------------

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
