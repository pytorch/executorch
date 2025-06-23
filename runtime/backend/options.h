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
#include <cstddef>
#include <cstring>
#include <variant>

namespace executorch {
namespace runtime {

// Strongly-typed option key template
template <typename T>
struct OptionKey {
  using value_type = T;
  const char* key;
  constexpr explicit OptionKey(const char* k) : key(k) {}
};

// All string keys and values must have static storage duration (string
// literals, static const char arrays, or global constants). The BackendOptions
// class does NOT take ownership of strings.
using OptionValue = std::variant<bool, int, const char*>;
static constexpr size_t kMaxOptionKeyLength = 64;

struct BackendOption {
  // key is the name of the backend option, like num_threads, enable_profiling,
  // etc
  char key[kMaxOptionKeyLength];
  // value is the value of the backend option, like 4, true, etc
  OptionValue value;
};

/**
 * A template class for storing and managing backend-specific configuration options.
 * 
 * This class provides a type-safe way to store key-value pairs for backend configuration,
 * with compile-time capacity limits and runtime type checking. It supports bool, int, and
 * const char* value types.
 * 
 * @tparam MaxCapacity The maximum number of options that can be stored
 */
template <size_t MaxCapacity>
class BackendOptions {
 public:
  /**
   * Default constructor - initializes with zero options.
   */
  BackendOptions() : size_(0) {}

  /**
   * Returns a const view of all stored options as a Span.
   * 
   * @return A const Span containing all BackendOption entries
   */
  executorch::runtime::Span<BackendOption> view() const {
    return executorch::runtime::Span<BackendOption>(options_, size_);
  }

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
    return set_option_impl(key, value);
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
    return set_option_impl(key, value);
  }

  /**
   * Sets a string option value for the given key.
   * If the key already exists, updates its value. Otherwise, adds a new option.
   * 
   * Note: The string value must have static storage duration. This class does NOT
   * take ownership of the string - it only stores the pointer.
   * 
   * @tparam N The length of the key string (automatically deduced)
   * @param key The option key (must be a string literal or array)
   * @param value The string value to set (must have static storage duration)
   * @return Error::Ok on success, Error::InvalidArgument if storage is full
   */
  template <size_t N>
  Error set_option(const char (&key)[N], const char* value) noexcept {
    static_assert(N <= kMaxOptionKeyLength, "Option key is too long");
    return set_option_impl(key, value);
  }

  /**
   * Retrieves an option value by key and type.
   * 
   * @tparam T The expected type of the option value (bool, int, or const char*)
   * @tparam KeyLen The length of the key string (automatically deduced)
   * @param key The option key to look up
   * @param out Reference to store the retrieved value
   * @return Error::Ok if found and type matches, Error::NotFound if key doesn't exist,
   *         Error::InvalidArgument if type doesn't match
   */
  template <typename T, size_t KeyLen>
  Error get_option(const char (&key)[KeyLen], T& out) const {
    static_assert(KeyLen <= kMaxOptionKeyLength, "Option key is too long");

    for (size_t i = 0; i < size_; ++i) {
      if (std::strcmp(options_[i].key, key) == 0) {
        if (auto* val = std::get_if<T>(&options_[i].value)) {
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
   * @param value The value to set
   * @return Error::Ok on success, Error::InvalidArgument if storage is full
   */
  template <typename T>
  Error set_option_impl(const char* key, T value) {
    // Update existing if found
    for (size_t i = 0; i < size_; ++i) {
      if (strcmp(options_[i].key, key) == 0) {
        options_[i].value = value;
        return Error::Ok;
      }
    }
    // Add new option if space available
    if (size_ < MaxCapacity) {
      BackendOption new_option;
      strncpy(new_option.key, key, kMaxOptionKeyLength - 1);
      new_option.key[kMaxOptionKeyLength - 1] = '\0';
      new_option.value = value;
      options_[size_++] = new_option;
      return Error::Ok;
    }
    // Return error when full
    return Error::InvalidArgument;
  }
};

} // namespace runtime
} // namespace executorch
