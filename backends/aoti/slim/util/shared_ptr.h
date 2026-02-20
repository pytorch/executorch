/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <utility>

#include <executorch/runtime/platform/assert.h>

namespace executorch::backends::aoti::slim {

/**
 * SharedPtr - A lightweight shared pointer implementation optimized for
 * single-threaded execution contexts.
 *
 * This class provides shared ownership semantics similar to std::shared_ptr but
 * without atomic operations, making it faster in single-threaded contexts.
 * ExecuTorch AOTI-drive backends operate in a single-threaded context, so
 * this optimization is safe and provides better performance.
 *
 * Primary Use Cases:
 * 1. Intermediate SlimTensor Storage Management:
 *    - Manages temporary tensors created during model execution
 *    - Avoids the overhead of atomic reference counting in std::shared_ptr
 *
 * 2. Input/Output Tensor References:
 *    - Provides reference counting for input/output tensors
 *    - Uses dummy deleters to prevent premature deallocation when needed
 */
template <typename T>
class SharedPtr {
 private:
  struct ControlBlock {
    int count = 1;
    T* ptr;
    using Deleter = void (*)(T*);
    Deleter deleter;

    ControlBlock(T* p, Deleter d) : ptr(p), deleter(d) {}
    ControlBlock(const ControlBlock&) = delete;
    ControlBlock& operator=(const ControlBlock&) = delete;
    ControlBlock(ControlBlock&&) = delete;
    ControlBlock& operator=(ControlBlock&&) = delete;

    ~ControlBlock() {
      if (ptr) {
        deleter(ptr);
      }
    }
  };

  ControlBlock* cb_;

  static void default_deleter(T* p) {
    delete p;
  }

  void cleanup() {
    if (cb_ && --cb_->count == 0) {
      delete cb_;
    }
    cb_ = nullptr;
  }

 public:
  /// Default constructor - creates an empty shared pointer.
  SharedPtr() noexcept : cb_(nullptr) {}

  /// Constructor from raw pointer.
  explicit SharedPtr(T* p, typename ControlBlock::Deleter d = default_deleter)
      : cb_(p ? new ControlBlock(p, d) : nullptr) {}

  /// Copy constructor.
  SharedPtr(const SharedPtr& other) noexcept : cb_(other.cb_) {
    if (cb_) {
      ++cb_->count;
    }
  }

  /// Move constructor.
  SharedPtr(SharedPtr&& other) noexcept : cb_(other.cb_) {
    other.cb_ = nullptr;
  }

  /// Destructor.
  ~SharedPtr() {
    cleanup();
  }

  /// Copy assignment.
  SharedPtr& operator=(const SharedPtr& other) noexcept {
    if (this != &other) {
      cleanup();
      cb_ = other.cb_;
      if (cb_) {
        ++cb_->count;
      }
    }
    return *this;
  }

  /// Move assignment.
  SharedPtr& operator=(SharedPtr&& other) noexcept {
    if (this != &other) {
      cleanup();
      cb_ = other.cb_;
      other.cb_ = nullptr;
    }
    return *this;
  }

  /// Resets the shared pointer to manage a new object.
  void reset(
      T* p = nullptr,
      typename ControlBlock::Deleter d = default_deleter) {
    *this = SharedPtr(p, d);
  }

  /// Swaps the contents with another shared pointer.
  void swap(SharedPtr& other) noexcept {
    std::swap(cb_, other.cb_);
  }

  /// Returns the managed pointer.
  T* get() const noexcept {
    return cb_ ? cb_->ptr : nullptr;
  }

  /// Dereferences the managed pointer.
  T& operator*() const {
    ET_CHECK_MSG(cb_, "Dereferencing null SharedPtr");
    return *cb_->ptr;
  }

  /// Accesses members of the managed object.
  T* operator->() const {
    ET_CHECK_MSG(cb_, "Accessing member of null SharedPtr");
    return cb_->ptr;
  }

  /// Returns the reference count.
  long use_count() const noexcept {
    return cb_ ? cb_->count : 0;
  }

  /// Returns true if the shared pointer is not null.
  explicit operator bool() const noexcept {
    return cb_ != nullptr;
  }

  friend void swap(SharedPtr& a, SharedPtr& b) noexcept {
    a.swap(b);
  }

  friend bool operator==(const SharedPtr& lhs, const SharedPtr& rhs) noexcept {
    return lhs.get() == rhs.get();
  }

  friend bool operator!=(const SharedPtr& lhs, const SharedPtr& rhs) noexcept {
    return !(lhs == rhs);
  }

  friend bool operator==(const SharedPtr& lhs, std::nullptr_t) noexcept {
    return lhs.get() == nullptr;
  }

  friend bool operator!=(const SharedPtr& lhs, std::nullptr_t) noexcept {
    return lhs.get() != nullptr;
  }

  friend bool operator==(std::nullptr_t, const SharedPtr& rhs) noexcept {
    return rhs.get() == nullptr;
  }

  friend bool operator!=(std::nullptr_t, const SharedPtr& rhs) noexcept {
    return rhs.get() != nullptr;
  }
};

/// Creates a SharedPtr managing a new object constructed with the given args.
template <typename T, typename... Args>
SharedPtr<T> make_shared(Args&&... args) {
  return SharedPtr<T>(new T(std::forward<Args>(args)...));
}

} // namespace executorch::backends::aoti::slim
