/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Result type to be used in conjunction with ExecuTorch Error type.
 */

#pragma once

#include <new>
#include <utility>

#include "executorch/runtime/core/error.h"
#include "executorch/runtime/platform/assert.h"

namespace executorch {
namespace runtime {

/**
 * Result type wrapping either a value of type T or an error.
 *
 * Example use case:
 * @code
 *   Result<OpFn> getOp(int opcode) {
 *     if (isValidOpCode(opcode)) {
 *       return opFns[opcode];
 *     }
 *     return Error::NotFound;
 *   }
 *
 *   Error useOp(int opcode) {
 *     Result<OpFn> op = getOp(opcode);
 *     if (!op.ok()) {
 *       return op.error();
 *     }
 *     print(op->toString());
 *     execute(*op);
 *     return Error::Ok;
 *   }
 * @endcode
 */
template <typename T>
class Result final {
 public:
  /// `value_type` member for generic programming.
  typedef T value_type;

  /**
   * Creates a Result object from an Error.
   *
   * To preserve the invariant that `(result.error() == Error::Ok) ==
   * result.ok()`, an `error` parameter value of `Error:Ok` will be converted to
   * a non-Ok value.
   */
  /* implicit */ Result(Error error)
      : error_(error == Error::Ok ? Error::Internal : error),
        hasValue_(false) {}

  /// Value copy constructor.
  /* implicit */ Result(const T& val) : value_(val), hasValue_(true) {}

  /// Value move constructor.
  /* implicit */ Result(T&& val) : value_(std::move(val)), hasValue_(true) {}

  /// Result move constructor.
  /* implicit */ Result(Result&& rhs) noexcept : hasValue_(rhs.hasValue_) {
    if (hasValue_) {
      // Use the value type's move constructor.
      new (&value_) T(std::move(rhs.value_));
    } else {
      error_ = rhs.error_;
    }
  }

  ~Result() {
    if (hasValue_) {
      // Manual value destruction.
      // Result "owns" the memory, so `delete` would segfault.
      value_.~T();
    }
  }

  /**
   * Returns true if this Result has a value.
   *
   * If true, it is guaranteed that `error()` will return `Error::Ok`.
   * If false, it is guaranteed that `error()` will not return `Error::Ok`.
   */
  __ET_NODISCARD bool ok() const {
    return hasValue_;
  }

  /**
   * Returns the error code of this Result.
   *
   * If this returns `Error::Ok`, it is guaranteed that `ok()` will return true.
   * If this does not return `Error:Ok`, it is guaranteed that `ok()` will
   * return false.
   */
  __ET_NODISCARD Error error() const {
    if (hasValue_) {
      return Error::Ok;
    } else {
      return error_;
    }
  }

  /**
   * Returns a reference to the Result's value; longhand for operator*().
   *
   * Only legal to call if `ok()` returns true.
   */
  T& get() {
    CheckOk();
    return value_;
  }

  /**
   * Returns a reference to the Result's value; longhand for operator*().
   *
   * Only legal to call if `ok()` returns true.
   */
  const T& get() const {
    CheckOk();
    return value_;
  }

  /*
   * Returns a reference to the Result's value; shorthand for get().
   *
   * Only legal to call if `ok()` returns true.
   */
  const T& operator*() const&;
  T& operator*() &;

  /*
   * Returns a pointer to the Result's value.
   *
   * Only legal to call if `ok()` returns true.
   */
  const T* operator->() const;
  T* operator->();

 private:
  /**
   * Delete default constructor since all Results should contain a value or
   * error.
   */
  Result() = delete;
  /// Delete copy constructor since T may not be copyable.
  Result(const Result&) = delete;
  /// Delete copy assignment since T may not be copyable.
  Result& operator=(const Result&) = delete;
  /// Delete move assignment since it's not a supported pattern to reuse Result.
  Result& operator=(Result&& rhs) = delete;

  // Panics if ok() would return false;
  void CheckOk() const {
    ET_CHECK(hasValue_);
  }

  union {
    T value_; // Used if hasValue_ is true.
    Error error_; // Used if hasValue_ is false.
  };

  /// True if the Result contains a value.
  const bool hasValue_;
};

template <typename T>
const T& Result<T>::operator*() const& {
  CheckOk();
  return value_;
}

template <typename T>
T& Result<T>::operator*() & {
  CheckOk();
  return value_;
}

template <typename T>
const T* Result<T>::operator->() const {
  CheckOk();
  return &value_;
}

template <typename T>
T* Result<T>::operator->() {
  CheckOk();
  return &value_;
}

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::Result;
} // namespace executor
} // namespace torch

/**
 * Unwrap a Result to obtain its value. If the Result contains an error,
 * propogate the error via trivial function return.
 *
 * Note: A function using ET_UNWRAP should itself return a Result or Error.
 *
 * @param[in] result__ Expression yielding the result to unwrap.
 * @param[in] ... Optional format string for the log error message and its
 * arguments.
 */
#define ET_UNWRAP(result__, ...) ET_INTERNAL_UNWRAP(result__, ##__VA_ARGS__)

// Internal only: Use ET_UNWRAP() instead.
#define ET_INTERNAL_UNWRAP(...)                                         \
  ET_INTERNAL_UNWRAP_SELECT(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1) \
  (__VA_ARGS__)

// Internal only: Use ET_UNWRAP() instead.
#define ET_INTERNAL_UNWRAP_SELECT(                   \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) \
  ET_INTERNAL_UNWRAP_##N

// Internal only: Use ET_UNWRAP() instead.
#define ET_INTERNAL_UNWRAP_1(result__) \
  ({                                   \
    auto et_result__ = (result__);     \
    if (!et_result__.ok()) {           \
      return et_result__.error();      \
    }                                  \
    std::move(*et_result__);           \
  })

// Internal only: Use ET_UNWRAP() instead.
#define ET_INTERNAL_UNWRAP_2(result__, message__, ...) \
  ({                                                   \
    auto et_result__ = (result__);                     \
    if (!et_result__.ok()) {                           \
      ET_LOG(Error, message__, ##__VA_ARGS__);         \
      return et_result__.error();                      \
    }                                                  \
    std::move(*et_result__);                           \
  })

// Internal only: Use ET_UNWRAP() instead.
#define ET_INTERNAL_UNWRAP_3 ET_INTERNAL_UNWRAP_2
#define ET_INTERNAL_UNWRAP_4 ET_INTERNAL_UNWRAP_2
#define ET_INTERNAL_UNWRAP_5 ET_INTERNAL_UNWRAP_2
#define ET_INTERNAL_UNWRAP_6 ET_INTERNAL_UNWRAP_2
#define ET_INTERNAL_UNWRAP_7 ET_INTERNAL_UNWRAP_2
#define ET_INTERNAL_UNWRAP_8 ET_INTERNAL_UNWRAP_2
#define ET_INTERNAL_UNWRAP_9 ET_INTERNAL_UNWRAP_2
#define ET_INTERNAL_UNWRAP_10 ET_INTERNAL_UNWRAP_2
