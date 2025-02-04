/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/platform/assert.h>
#include <new>
#include <utility> // std::forward and other template magic checks

namespace executorch {
namespace runtime {
namespace etensor {

/// Used to indicate an optional type with uninitialized state.
struct nullopt_t final {
  constexpr explicit nullopt_t(int32_t) {}
};

/// A constant of type nullopt_t that is used to indicate an optional type with
/// uninitialized state.
constexpr nullopt_t nullopt{0};

/// Leaner optional class, subset of c10, std, and boost optional APIs.
template <class T>
class optional final {
 public:
  /// The type wrapped by the optional class.
  using value_type = T;

  /// Constructs an optional object that does not contain a value.
  /* implicit */ optional() noexcept : storage_(trivial_init), init_(false) {}

  /// Constructs an optional object that does not contain a value.
  /* implicit */ optional(nullopt_t) noexcept
      : storage_(trivial_init), init_(false) {}

  /// Constructs an optional object that matches the state of v.
  /* implicit */ optional(const optional<T>& v)
      : storage_(trivial_init), init_(v.init_) {
    if (init_) {
      new (&storage_.value_) T(v.storage_.value_);
    }
  }

  /// Constructs an optional object that contains the specified value.
  /* implicit */ optional(const T& v) : storage_(v), init_(true) {}

  /// Constructs an optional object from v.
  /* implicit */ optional(optional<T>&& v) noexcept(
      std::is_nothrow_move_constructible<T>::value)
      : storage_(trivial_init), init_(v.init_) {
    if (init_) {
      new (&storage_.value_) T(std::forward<T>(v.storage_.value_));
    }
  }

  /// Constructs an optional object that contains the specified value.
  /* implicit */ optional(T&& v) : storage_(std::forward<T>(v)), init_(true) {}

  optional& operator=(const optional& rhs) {
    if (init_ && !rhs.init_) {
      clear();
    } else if (!init_ && rhs.init_) {
      init_ = true;
      new (&storage_.value_) T(rhs.storage_.value_);
    } else if (init_ && rhs.init_) {
      storage_.value_ = rhs.storage_.value_;
    }
    return *this;
  }

  optional& operator=(optional&& rhs) noexcept(
      std::is_nothrow_move_assignable<T>::value &&
      std::is_nothrow_move_constructible<T>::value) {
    if (init_ && !rhs.init_) {
      clear();
    } else if (!init_ && rhs.init_) {
      init_ = true;
      new (&storage_.value_) T(std::forward<T>(rhs.storage_.value_));
    } else if (init_ && rhs.init_) {
      storage_.value_ = std::forward<T>(rhs.storage_.value_);
    }
    return *this;
  }

  /// Destroys the stored value if there is one
  ~optional() {
    if (init_) {
      storage_.value_.~T();
    }
  }

  optional& operator=(nullopt_t) noexcept {
    clear();
    return *this;
  }

  /// Returns true if the object contains a value, false otherwise
  explicit operator bool() const noexcept {
    return init_;
  }

  /// Returns true if the object contains a value, false otherwise
  bool has_value() const noexcept {
    return init_;
  }

  /// Returns a constant reference to the contained value. Calls ET_CHECK if
  /// the object does not contain a value.
  T const& value() const& {
    ET_CHECK(init_);
    return contained_val();
  }

  /// Returns a mutable reference to the contained value. Calls ET_CHECK if the
  /// object does not contain a value.
  T& value() & {
    ET_CHECK(init_);
    return contained_val();
  }

  /// Returns an rvalue of the contained value. Calls ET_CHECK if the object
  /// does not contain a value.
  T&& value() && {
    ET_CHECK(init_);
    return std::forward<T>(contained_val());
  }

 private:
  // Used to invoke the dummy ctor of storage_t in the initializer lists of
  // optional_base as default ctor is implicitly deleted because T is nontrivial
  struct trivial_init_t {
  } trivial_init{};

  /**
   * A wrapper type that lets us avoid constructing a T when there is no value.
   * If there is a value present, the optional class must destroy it.
   */
  union storage_t {
    /// A small, trivially-constructable alternative to T.
    unsigned char dummy_;
    /// The constructed value itself, if optional::has_value_ is true.
    T value_;

    /* implicit */ storage_t(trivial_init_t) {
      dummy_ = 0;
    }

    template <class... Args>
    storage_t(Args&&... args) : value_(std::forward<Args>(args)...) {}

    ~storage_t() {}
  };

  const T& contained_val() const& {
    return storage_.value_;
  }
  T&& contained_val() && {
    return std::move(storage_.value_);
  }
  T& contained_val() & {
    return storage_.value_;
  }

  void clear() noexcept {
    if (init_) {
      storage_.value_.~T();
    }
    init_ = false;
  }

  storage_t storage_;
  bool init_;
};

} // namespace etensor
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::nullopt;
using ::executorch::runtime::etensor::nullopt_t;
using ::executorch::runtime::etensor::optional;
} // namespace executor
} // namespace torch
