/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/span.h>
#include <array>
#include <cstddef>
#include <cstring>
#include <variant>

namespace executorch {
namespace runtime {

static constexpr size_t kMaxOptionKeyLength = 64;
static constexpr size_t kMaxOptionValueLength = 256;

// Keys and string values are copied into the fixed-size arrays below,
// including their NUL terminators. BackendOptions rejects a key or value that
// does not fit instead of shortening it.
using OptionValue =
    std::variant<bool, int, std::array<char, kMaxOptionValueLength>>;

struct BackendOption {
  // key is the name of the backend option, like num_threads, enable_profiling,
  // etc
  char key[kMaxOptionKeyLength]{};
  // value is the value of the backend option, like 4, true, etc
  OptionValue value;
};

/**
 * A template class for storing and managing backend-specific configuration
 * options.
 *
 * This class provides a type-safe way to store key-value pairs for backend
 * configuration, with compile-time capacity limits and runtime type checking.
 * It supports bool, int, and const char* value types.
 *
 * @tparam MaxCapacity The maximum number of options that can be stored
 */
template <size_t MaxCapacity>
class BackendOptions {
 public:
  /**
   * Copy constructor
   */
  BackendOptions(const BackendOptions& other) : size_(other.size_) {
    for (size_t i = 0; i < size_; ++i) {
      options_[i] = other.options_[i];
    }
  }

  /**
   * Copy assignment operator
   */
  BackendOptions& operator=(const BackendOptions& other) {
    if (this != &other) {
      size_ = other.size_;
      for (size_t i = 0; i < size_; ++i) {
        options_[i] = other.options_[i];
      }
    }
    return *this;
  }

  /**
   * Default constructor - initializes with zero options.
   */
  BackendOptions() : size_(0) {}

  /**
   * Returns a mutable view of all stored options as a Span.
   *
   * @return A mutable Span containing all BackendOption entries
   */
  executorch::runtime::Span<BackendOption> view() {
    return executorch::runtime::Span<BackendOption>(options_, size_);
  }

  /**
   * Sets a boolean option value for the given key.
   * If the key already exists, updates its value. Otherwise, adds a new option.
   *
   * @tparam N The length of the key string (automatically deduced)
   * @param key The option key (must be a string literal or array)
   * @param value The boolean value to set
   * @return Error::Ok on success, Error::InvalidArgument if storage is full
   */
  template <size_t N>
  Error set_option(const char (&key)[N], bool value) noexcept {
    static_assert(N <= kMaxOptionKeyLength, "Option key is too long");
    return set_option_impl(key, N, value);
  }

  /**
   * Sets an integer option value for the given key.
   * If the key already exists, updates its value. Otherwise, adds a new option.
   *
   * @tparam N The length of the key string (automatically deduced)
   * @param key The option key (must be a string literal or array)
   * @param value The integer value to set
   * @return Error::Ok on success, Error::InvalidArgument if storage is full
   */
  template <size_t N>
  Error set_option(const char (&key)[N], int value) noexcept {
    static_assert(N <= kMaxOptionKeyLength, "Option key is too long");
    return set_option_impl(key, N, value);
  }

  /**
   * Sets a string option value for the given key.
   * If the key already exists, updates its value. Otherwise, adds a new option.
   *
   * The string is copied, so it only needs to live for the duration of the
   * call. A value longer than kMaxOptionValueLength - 1 bytes is rejected
   * rather than cut: a shortened value, such as a file path, can silently
   * name something else.
   *
   * @tparam N The length of the key string (automatically deduced)
   * @param key The option key (must be a string literal or array)
   * @param value The NUL-terminated string value to set
   * @return Error::Ok on success, Error::InvalidArgument if storage is full,
   * the value is null, or the value does not fit. On error the options are
   * left unchanged.
   */
  template <size_t N>
  Error set_option(const char (&key)[N], const char* value) noexcept {
    static_assert(N <= kMaxOptionKeyLength, "Option key is too long");
    if (value == nullptr) {
      ET_LOG(
          Error,
          "Backend option '%.*s' has a null value",
          static_cast<int>(N),
          key);
      return Error::InvalidArgument;
    }
    const size_t value_len = std::strlen(value);
    if (value_len >= kMaxOptionValueLength) {
      ET_LOG(
          Error,
          "Backend option '%.*s' value is %zu bytes, longer than the %zu byte limit; not set",
          static_cast<int>(N),
          key,
          value_len,
          kMaxOptionValueLength - 1);
      return Error::InvalidArgument;
    }
    std::array<char, kMaxOptionValueLength> arr{};
    std::memcpy(arr.data(), value, value_len);
    return set_option_impl(key, N, arr);
  }
  /**
   * Retrieves an option value by key and type.
   *
   * @tparam T The expected type of the option value (bool, int, or const char*)
   * @tparam KeyLen The length of the key string (automatically deduced)
   * @param key The option key to look up
   * @param out Reference to store the retrieved value
   * @return Error::Ok if found and type matches, Error::NotFound if key doesn't
   * exist, Error::InvalidArgument if type doesn't match
   */
  template <typename T, size_t KeyLen>
  Error get_option(const char (&key)[KeyLen], T& out) const {
    static_assert(KeyLen <= kMaxOptionKeyLength, "Option key is too long");
    for (size_t i = 0; i < size_; ++i) {
      if (std::strcmp(options_[i].key, key) == 0) {
        // Special handling for string (convert array to const char*)
        if constexpr (std::is_same_v<T, const char*>) {
          if (auto* arr = std::get_if<std::array<char, kMaxOptionValueLength>>(
                  &options_[i].value)) {
            out = arr->data(); // Return pointer to stored array
            return Error::Ok;
          }
        }
        // Default handling for bool/int
        else if (auto* val = std::get_if<T>(&options_[i].value)) {
          out = *val;
          return Error::Ok;
        }
        return Error::InvalidArgument;
      }
    }
    return Error::NotFound;
  }

 private:
  BackendOption options_[MaxCapacity]{}; // Storage for backend options
  size_t size_; // Current number of options

  /**
   * Internal implementation for setting option values.
   * Handles both updating existing options and adding new ones.
   *
   * @tparam T The type of the value (bool, int, or const char*)
   * @param key The option key
   * @param key_capacity The size of the caller's key array, at most
   * kMaxOptionKeyLength
   * @param value The value to set
   * @return Error::Ok on success, Error::InvalidArgument if storage is full
   * or the key is not NUL-terminated within its array
   */
  template <typename T>
  Error set_option_impl(const char* key, size_t key_capacity, T value) {
    if (std::memchr(key, '\0', key_capacity) == nullptr) {
      ET_LOG(
          Error,
          "Backend option key '%.*s' is not NUL-terminated within %zu bytes; not set",
          static_cast<int>(key_capacity),
          key,
          key_capacity);
      return Error::InvalidArgument;
    }
    // Update existing if found
    for (size_t i = 0; i < size_; ++i) {
      if (strcmp(options_[i].key, key) == 0) {
        options_[i].value = value;
        return Error::Ok;
      }
    }
    if (size_ < MaxCapacity) {
      BackendOption new_option;
      // The NUL check above bounds key_len to key_capacity - 1, which is at
      // most kMaxOptionKeyLength - 1, so the key and its NUL always fit.
      const size_t key_len = std::strlen(key);
      std::memcpy(new_option.key, key, key_len + 1);
      new_option.value = value; // Restored value assignment
      options_[size_++] = new_option; // Store option and increment size
      return Error::Ok;
    }
    return Error::InvalidArgument;
  }
};

} // namespace runtime
} // namespace executorch
