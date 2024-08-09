/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===--- ArrayRef.h - Array Reference Wrapper -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// removed llvm-specific functionality
// removed some implicit const -> non-const conversions that rely on
// complicated std::enable_if meta-programming
// removed a bunch of slice variants for simplicity...
// remove constructors for std::array
// remove constructors and operators for std::vector
// removed some prevention of accidental assignments from temporary that
// required std::enable_if meta-programming
// removed reverse iterator

#pragma once

#include <cstdint>

#include <executorch/runtime/platform/assert.h>

namespace executorch {
namespace runtime {

/**
 * Represents a constant reference to an array (0 or more elements
 * consecutively in memory), i.e. a start pointer and a length.  It allows
 * various APIs to take consecutive elements easily and conveniently.
 *
 * This class does not own the underlying data, it is expected to be used in
 * situations where the data resides in some other buffer, whose lifetime
 * extends past that of the ArrayRef. For this reason, it is not in general
 * safe to store an ArrayRef.
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
class ArrayRef final {
 public:
  using iterator = const T*;
  using const_iterator = const T*;
  using size_type = size_t;
  using value_type = T;

 private:
  /// The start of the array, in an external buffer.
  const T* Data;

  /// The number of elements.
  size_type Length;

 public:
  /// @name Constructors
  /// @{

  /// Construct an empty ArrayRef.
  /* implicit */ constexpr ArrayRef() : Data(nullptr), Length(0) {}

  /// Construct a ArrayRef from a single element. Implicitly convert element
  /// type. It is aligned with PyTorch's c10::ArrayRef.
  /* implicit */ constexpr ArrayRef(const T& OneElt)
      : Data(&OneElt), Length(1) {}

  /// Construct a ArrayRef from a pointer and length.
  ArrayRef(const T* data, size_t length) : Data(data), Length(length) {
    ET_DCHECK(Data != nullptr || Length == 0);
  }

  /// Construct a ArrayRef from a range.
  ArrayRef(const T* begin, const T* end) : Data(begin), Length(end - begin) {}

  /// Construct a ArrayRef from a C array.
  template <size_t N>
  /* implicit */ constexpr ArrayRef(const T (&Arr)[N]) : Data(Arr), Length(N) {}

  /// @}
  /// @name Simple Operations
  /// @{

  constexpr iterator begin() const {
    return Data;
  }
  constexpr iterator end() const {
    return Data + Length;
  }

  // These are actually the same as iterator, since ArrayRef only
  // gives you const iterators.
  constexpr const_iterator cbegin() const {
    return Data;
  }
  constexpr const_iterator cend() const {
    return Data + Length;
  }

  /// empty - Check if the array is empty.
  constexpr bool empty() const {
    return Length == 0;
  }

  constexpr const T* data() const {
    return Data;
  }

  /// size - Get the array size.
  constexpr size_t size() const {
    return Length;
  }

  /// front - Get the first element.
  const T& front() const {
    // ArrayRef: attempted to access front() of empty list
    ET_CHECK(!empty());
    return Data[0];
  }

  /// back - Get the last element.
  const T& back() const {
    // ArrayRef: attempted to access back() of empty list
    ET_CHECK(!empty());
    return Data[Length - 1];
  }

  /// equals - Check for element-wise equality.
  bool equals(ArrayRef RHS) const {
    if (Length != RHS.Length) {
      return false;
    }
    for (size_t i = 0; i < this->Length; i++) {
      if (Data[i] != RHS.Data[i]) {
        return false;
      }
    }
    return true;
  }

  /// slice(n, m) - Take M elements of the array starting at element N
  ArrayRef<T> slice(size_t N, size_t M) const {
    // cant slice longer then the array
    ET_CHECK(N + M <= size());
    return ArrayRef<T>(data() + N, M);
  }

  /// slice(n) - Chop off the first N elements of the array.
  constexpr ArrayRef<T> slice(size_t N) const {
    return slice(N, size() - N);
  }

  /// @}
  /// @name Operator Overloads
  /// @{
  constexpr const T& operator[](size_t Index) const {
    return Data[Index];
  }

  /// Vector compatibility
  const T& at(size_t Index) const {
    // invalid index
    ET_CHECK(Index < Length);
    return Data[Index];
  }

  /// @}
};

/// @name ArrayRef Convenience constructors
/// @{

/// Construct an ArrayRef from a single element.
template <typename T>
ArrayRef<T> makeArrayRef(const T& OneElt) {
  return OneElt;
}

/// Construct an ArrayRef from a pointer and length.
template <typename T>
ArrayRef<T> makeArrayRef(const T* data, size_t length) {
  return ArrayRef<T>(data, length);
}

/// Construct an ArrayRef from a range.
template <typename T>
ArrayRef<T> makeArrayRef(const T* begin, const T* end) {
  return ArrayRef<T>(begin, end);
}

/// Construct an ArrayRef from an ArrayRef (no-op) (const)
template <typename T>
ArrayRef<T> makeArrayRef(const ArrayRef<T>& Vec) {
  return Vec;
}

/// Construct an ArrayRef from an ArrayRef (no-op)
template <typename T>
ArrayRef<T>& makeArrayRef(ArrayRef<T>& Vec) {
  return Vec;
}

/// Construct an ArrayRef from a C array.
template <typename T, size_t N>
ArrayRef<T> makeArrayRef(const T (&Arr)[N]) {
  return ArrayRef<T>(Arr);
}

// WARNING: Template instantiation will NOT be willing to do an implicit
// conversions to get you to an ArrayRef, which is why we need so
// many overloads.

template <typename T>
bool operator==(ArrayRef<T> a1, ArrayRef<T> a2) {
  return a1.equals(a2);
}

template <typename T>
bool operator!=(ArrayRef<T> a1, ArrayRef<T> a2) {
  return !a1.equals(a2);
}

using IntArrayRef = ArrayRef<int64_t>;

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::IntArrayRef;
using ::executorch::runtime::makeArrayRef;
} // namespace executor
} // namespace torch
