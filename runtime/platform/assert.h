/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/platform/abort.h>
#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/log.h>

/**
 * Assertion failure message emit method.
 *
 * @param[in] _format Printf-style message format string.
 * @param[in] ... Format string arguments.
 */
#define ET_ASSERT_MESSAGE_EMIT(_format, ...)     \
  ET_LOG(                                        \
      Fatal,                                     \
      "In function %s(), assert failed" _format, \
      ET_FUNCTION,                               \
      ##__VA_ARGS__)

/**
 * Abort the runtime if the condition is not true.
 * This check will be performed even in release builds.
 *
 * @param[in] _cond Condition asserted as true.
 * @param[in] _format Printf-style message format string.
 * @param[in] ... Format string arguments.
 */
#define ET_CHECK_MSG(_cond, _format, ...)                               \
  ({                                                                    \
    if ET_UNLIKELY (!(_cond)) {                                         \
      ET_ASSERT_MESSAGE_EMIT(" (%s): " _format, #_cond, ##__VA_ARGS__); \
      ::executorch::runtime::runtime_abort();                           \
    }                                                                   \
  })

/**
 * Abort the runtime if the condition is not true.
 * This check will be performed even in release builds.
 *
 * @param[in] _cond Condition asserted as true.
 */
#define ET_CHECK(_cond)                       \
  ({                                          \
    if ET_UNLIKELY (!(_cond)) {               \
      ET_ASSERT_MESSAGE_EMIT(": %s", #_cond); \
      ::executorch::runtime::runtime_abort(); \
    }                                         \
  })

#ifdef NDEBUG

/**
 * Abort the runtime if the condition is not true.
 * This check will be performed in debug builds, but not release builds.
 *
 * @param[in] _cond Condition asserted as true.
 * @param[in] _format Printf-style message format string.
 * @param[in] ... Format string arguments.
 */
#define ET_DCHECK_MSG(_cond, _format, ...) ((void)0)

/**
 * Abort the runtime if the condition is not true.
 * This check will be performed in debug builds, but not release builds.
 *
 * @param[in] _cond Condition asserted as true.
 */
#define ET_DCHECK(_cond) ((void)0)

#else // NDEBUG

/**
 * Abort the runtime if the condition is not true.
 * This check will be performed in debug builds, but not release builds.
 *
 * @param[in] _cond Condition asserted as true.
 * @param[in] _format Printf-style message format string.
 * @param[in] ... Format string arguments.
 */
#define ET_DCHECK_MSG(_cond, _format, ...) \
  ET_CHECK_MSG(_cond, _format, ##__VA_ARGS__)

/**
 * Abort the runtime if the condition is not true.
 * This check will be performed in debug builds, but not release builds.
 *
 * @param[in] _cond Condition asserted as true.
 */
#define ET_DCHECK(_cond) ET_CHECK(_cond)

#endif // NDEBUG

/**
 * Assert that this code location is unreachable during execution.
 */
#define ET_ASSERT_UNREACHABLE()                                   \
  ({                                                              \
    ET_CHECK_MSG(false, "Execution should not reach this point"); \
    ET_UNREACHABLE();                                             \
  })

/**
 * Assert that this code location is unreachable during execution.
 *
 * @param[in] _message Message on how to avoid this assertion error.
 */
#define ET_ASSERT_UNREACHABLE_MSG(_message)                            \
  ({                                                                   \
    ET_CHECK_MSG(                                                      \
        false, "Execution should not reach this point. %s", _message); \
    ET_UNREACHABLE();                                                  \
  })
