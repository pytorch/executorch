/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Result type to be used in conjunction with Tokenizers Error type.
 */

#pragma once

#include <pytorch/tokenizers/error.h>
#include <cassert>
#include <new>
#include <utility>

namespace tokenizers {

/**
 * Result type wrapping either a value of type T or an error.
 *
 * Example use case:
 * @code
 *   Result<std::string> decode(uint64_t token);
 *
 *   void generate()
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
  bool ok() const {
    return hasValue_;
  }

  /**
   * Returns the error code of this Result.
   *
   * If this returns `Error::Ok`, it is guaranteed that `ok()` will return true.
   * If this does not return `Error:Ok`, it is guaranteed that `ok()` will
   * return false.
   */
  Error error() const {
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
    assert(hasValue_ && "Result must be ok to access value.");
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

} // namespace tokenizers

/**
 * Unwraps a Result<T> value, throwing a runtime_error if the result contains an
 * error.
 *
 * @param[in] result__ The Result<T> to unwrap
 */
#define TK_UNWRAP_THROW(result__)                                     \
  ({                                                                  \
    auto unwrap_result__ = (result__);                                \
    if (!unwrap_result__.ok()) {                                      \
      throw std::runtime_error(                                       \
          "Error: " +                                                 \
          std::to_string(static_cast<int>(unwrap_result__.error()))); \
    }                                                                 \
    std::move(unwrap_result__.get());                                 \
  })

/**
 * Unwrap a Result to obtain its value. If the Result contains an error,
 * propogate the error via trivial function return.
 *
 * Note: A function using TK_UNWRAP should itself return a Result or Error.
 *
 * @param[in] result__ Expression yielding the result to unwrap.
 * @param[in] ... Optional format string for the log error message and its
 * arguments.
 */
#define TK_UNWRAP(result__, ...) TK_INTERNAL_UNWRAP(result__, ##__VA_ARGS__)

// Internal only: Use TK_UNWRAP() instead.
#define TK_INTERNAL_UNWRAP(...)                                         \
  TK_INTERNAL_UNWRAP_SELECT(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1) \
  (__VA_ARGS__)

// Internal only: Use TK_UNWRAP() instead.
#define TK_INTERNAL_UNWRAP_SELECT(                   \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) \
  TK_INTERNAL_UNWRAP_##N

// Internal only: Use TK_UNWRAP() instead.
#define TK_INTERNAL_UNWRAP_1(result__) \
  ({                                   \
    auto et_result__ = (result__);     \
    if (!et_result__.ok()) {           \
      return et_result__.error();      \
    }                                  \
    std::move(*et_result__);           \
  })

// Internal only: Use TK_UNWRAP() instead.
#define TK_INTERNAL_UNWRAP_2(result__, message__, ...) \
  ({                                                   \
    auto et_result__ = (result__);                     \
    if (!et_result__.ok()) {                           \
      TK_LOG(Error, message__, ##__VA_ARGS__);         \
      return et_result__.error();                      \
    }                                                  \
    std::move(*et_result__);                           \
  })

// Internal only: Use TK_UNWRAP() instead.
#define TK_INTERNAL_UNWRAP_3 TK_INTERNAL_UNWRAP_2
#define TK_INTERNAL_UNWRAP_4 TK_INTERNAL_UNWRAP_2
#define TK_INTERNAL_UNWRAP_5 TK_INTERNAL_UNWRAP_2
#define TK_INTERNAL_UNWRAP_6 TK_INTERNAL_UNWRAP_2
#define TK_INTERNAL_UNWRAP_7 TK_INTERNAL_UNWRAP_2
#define TK_INTERNAL_UNWRAP_8 TK_INTERNAL_UNWRAP_2
#define TK_INTERNAL_UNWRAP_9 TK_INTERNAL_UNWRAP_2
#define TK_INTERNAL_UNWRAP_10 TK_INTERNAL_UNWRAP_2
