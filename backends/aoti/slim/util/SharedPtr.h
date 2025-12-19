#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <executorch/runtime/platform/assert.h>

namespace executorch::backends::aoti::slim {

/**
 * NonAtomicSharedPtr - A lightweight, non-thread-safe shared pointer
 * implementation
 *
 * This class provides shared ownership semantics similar to std::shared_ptr but
 * without atomic operations, making it faster in single-threaded contexts where
 * thread safety is not required.
 *
 * Primary Use Cases:
 * 1. Intermediate SlimTensor Storage Management:
 *    - Manages temporary tensors created during model execution
 *    - These tensors are confined to single-threaded execution contexts
 *    - Avoids the overhead of atomic reference counting in std::shared_ptr
 *
 * 2. Input/Output Tensor References:
 *    - Provides reference counting for input/output tensors
 *    - Tensor lifetimes are externally managed (not by AOTI-generated code)
 *    - Uses dummy deleters to prevent premature deallocation
 *    - Reference counting still occurs but actual cleanup is deferred
 *
 * Performance Benefits:
 * - Non-atomic reference counting reduces CPU overhead
 * - Smaller memory footprint compared to std::shared_ptr
 * - Optimized for single-threaded tensor operations
 *
 * Thread Safety: NOT THREAD-SAFE
 * - Must only be used in single-threaded contexts
 * - Concurrent access will result in undefined behavior
 * - Define the USE_MULTI_THREAD macro to use std::shared_ptr instead when
 * thread safety is required
 */
template <typename T>
class NonAtomicSharedPtr {
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
  // Default constructor
  NonAtomicSharedPtr() noexcept : cb_(nullptr) {}

  // Constructor from raw pointer
  explicit NonAtomicSharedPtr(
      T* p,
      typename ControlBlock::Deleter d = default_deleter)
      : cb_(p ? new ControlBlock(p, d) : nullptr) {}

  // Copy constructor
  NonAtomicSharedPtr(const NonAtomicSharedPtr& other) noexcept
      : cb_(other.cb_) {
    if (cb_) {
      ++cb_->count;
    }
  }

  // Move constructor
  NonAtomicSharedPtr(NonAtomicSharedPtr&& other) noexcept : cb_(other.cb_) {
    other.cb_ = nullptr;
  }

  // Destructor
  ~NonAtomicSharedPtr() {
    cleanup();
  }

  // Copy assignment
  NonAtomicSharedPtr& operator=(const NonAtomicSharedPtr& other) noexcept {
    if (this != &other) {
      cleanup();
      cb_ = other.cb_;
      if (cb_) {
        ++cb_->count;
      }
    }
    return *this;
  }

  // Move assignment
  NonAtomicSharedPtr& operator=(NonAtomicSharedPtr&& other) noexcept {
    if (this != &other) {
      cleanup();
      cb_ = other.cb_;
      other.cb_ = nullptr;
    }
    return *this;
  }

  // Modifiers
  void reset(
      T* p = nullptr,
      typename ControlBlock::Deleter d = default_deleter) {
    *this = NonAtomicSharedPtr(p, d);
  }

  void swap(NonAtomicSharedPtr& other) noexcept {
    std::swap(cb_, other.cb_);
  }

  // Observers
  T* get() const noexcept {
    return cb_ ? cb_->ptr : nullptr;
  }
  T& operator*() const {
    ET_CHECK_MSG(cb_, "Dereferencing null NonAtomicSharedPtr");
    return *cb_->ptr;
  }
  T* operator->() const {
    ET_CHECK_MSG(cb_, "Accessing member of null NonAtomicSharedPtr");
    return cb_->ptr;
  }
  long use_count() const noexcept {
    return cb_ ? cb_->count : 0;
  }
  explicit operator bool() const noexcept {
    return cb_ != nullptr;
  }

  // Friend swap for ADL
  friend void swap(NonAtomicSharedPtr& a, NonAtomicSharedPtr& b) noexcept {
    a.swap(b);
  }

  // Comparison operators
  friend bool operator==(
      const NonAtomicSharedPtr& lhs,
      const NonAtomicSharedPtr& rhs) noexcept {
    return lhs.get() == rhs.get();
  }

  friend bool operator!=(
      const NonAtomicSharedPtr& lhs,
      const NonAtomicSharedPtr& rhs) noexcept {
    return !(lhs == rhs);
  }

  friend bool operator==(
      const NonAtomicSharedPtr& lhs,
      std::nullptr_t) noexcept {
    return lhs.get() == nullptr;
  }

  friend bool operator!=(
      const NonAtomicSharedPtr& lhs,
      std::nullptr_t) noexcept {
    return lhs.get() != nullptr;
  }

  friend bool operator==(
      std::nullptr_t,
      const NonAtomicSharedPtr& rhs) noexcept {
    return rhs.get() == nullptr;
  }

  friend bool operator!=(
      std::nullptr_t,
      const NonAtomicSharedPtr& rhs) noexcept {
    return rhs.get() != nullptr;
  }
};

#ifdef USE_MULTI_THREAD
template <typename T>
using SharedPtr = ::std::shared_ptr<T>;

// make_shared for std::shared_ptr
template <typename T, typename... Args>
std::shared_ptr<T> make_shared(Args&&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

#else
template <typename T>
using SharedPtr = ::executorch::backends::aoti::slim::NonAtomicSharedPtr<T>;

// make_shared for NonAtomicSharedPtr
template <typename T, typename... Args>
NonAtomicSharedPtr<T> make_shared(Args&&... args) {
  return NonAtomicSharedPtr<T>(new T(std::forward<Args>(args)...));
}

#endif // USE_MULTI_THREAD
} // namespace executorch::backends::aoti::slim
