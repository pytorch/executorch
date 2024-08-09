/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <executorch/runtime/platform/assert.h>

namespace executorch {
namespace runtime {

/**
 * Represent a reference to an array (0 or more elements
 * consecutively in memory), i.e. a start pointer and a length.  It allows
 * various APIs to take consecutive elements easily and conveniently.
 *
 * This class does not own the underlying data, it is expected to be used in
 * situations where the data resides in some other buffer, whose lifetime
 * extends past that of the Span.
 *
 * Span and ArrayRef are extrememly similar with the difference being ArrayRef
 * views a list of constant elements and Span views a list of mutable elements.
 * Clients should decide between the two based on if the list elements for their
 * use case should be mutable.
 *
 * This is intended to be trivially copyable, so it should be passed by
 * value.
 */
template <typename T>
class Span final {
 public:
  using iterator = T*;
  using size_type = size_t;

 public:
  /// Construct an empty Span.
  /* implicit */ constexpr Span() noexcept : data_(nullptr), length_(0) {}

  /// Construct a Span from a pointer and length.
  Span(T* data, size_t length) : data_(data), length_(length) {
    ET_DCHECK(data_ != nullptr || length_ == 0);
  }

  /// Construct a Span from a range.
  Span(T* begin, T* end) : data_(begin), length_(end - begin) {}

  /// Construct a Span from a C array.
  template <size_t N>
  /* implicit */ constexpr Span(T (&Arr)[N]) : data_(Arr), length_(N) {}

  /// @returns a pointer to the start of the underlying element buffer.
  iterator begin() const noexcept {
    return data_;
  }

  /// @returns a pointer to the end of the underlying element buffer.
  iterator end() const noexcept {
    return data_ + length_;
  }

  /// @retval a boolean indicating if the Span is empty.
  constexpr bool empty() const noexcept {
    return length_ == 0;
  }

  /// @returns a pointer to the start of the underlying element buffer.
  constexpr T* data() const noexcept {
    return data_;
  }

  /// @returns the number of elements in the Span.
  constexpr size_t size() const noexcept {
    return length_;
  }

  /// Unchecked index into the array according to the argument index.
  /// @returns a reference to the element at the specified index.
  T& operator[](size_t index) const {
    return data_[index];
  }

 private:
  /// The start of the array, in an external buffer.
  T* data_;

  /// The number of elements.
  size_type length_;
};

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::Span;
} // namespace executor
} // namespace torch
