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
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <variant>

namespace executorch {
namespace runtime {

static constexpr size_t kMaxOptionKeyLength = 64;
static constexpr size_t kMaxOptionValueLength = 256;

// String keys: must fit in kMaxOptionKeyLength characters (including null
// terminator). The public set_option() templates enforce this at compile time
// via static_assert on the key array's length, so overlong keys fail to
// compile rather than being silently truncated.
// String values: COPIED into the internal `std::array<char, ...>` variant arm
// and truncated at kMaxOptionValueLength - 1 characters (null-terminated).
// Callers do NOT need to keep the source strings alive after the set_option()
// call returns.
// The int64_t arm lets callers pass pointer-sized opaque handles (e.g.,
// driver handles like CUgreenCtx, cudaStream_t). Round-trip the pointer
// through uintptr_t so the cast is well-defined on all platforms:
//   opts.set_option("cuda_stream",
//       static_cast<int64_t>(reinterpret_cast<uintptr_t>(stream)));
//   auto* stream = reinterpret_cast<cudaStream_t>(
//       static_cast<uintptr_t>(value));
// On 32-bit platforms the int64_t arm is wider than necessary for pointers
// but remains correct.
using OptionValue =
    std::variant<bool, int, int64_t, std::array<char, kMaxOptionValueLength>>;

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
 * It supports bool, int, int64_t, and const char* value types. The int64_t
 * arm allows callers to pass pointer-sized opaque handles (e.g., driver
 * handles like CUgreenCtx) by reinterpret_cast to/from int64_t.
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
   * Sets an int64_t option value for the given key.
   *
   * Useful for pointer-sized opaque handles. Round-trip the pointer through
   * uintptr_t so the cast is well-defined on all platforms:
   *   opts.set_option("cuda_stream",
   *       static_cast<int64_t>(reinterpret_cast<uintptr_t>(stream)));
   *
   * Note: bare integer literals like `42` resolve to `int` (NOT `int64_t`),
   * and `42L` resolves to `int64_t` only on platforms where `long` is 64-bit
   * (Linux) but `int` on Windows. To target the int64_t arm unambiguously,
   * use `static_cast<int64_t>(value)` at the call site.
   *
   * If the key already exists, updates its value. Otherwise, adds a new option.
   *
   * @tparam N The length of the key string (automatically deduced)
   * @param key The option key (must be a string literal or array)
   * @param value The int64_t value to set
   * @return Error::Ok on success, Error::InvalidArgument if storage is full
   */
  template <size_t N>
  Error set_option(const char (&key)[N], int64_t value) noexcept {
    static_assert(N <= kMaxOptionKeyLength, "Option key is too long");
    return set_option_impl(key, value);
  }

  /**
   * Sets a string option value for the given key.
   * If the key already exists, updates its value. Otherwise, adds a new option.
   *
   * The string value is copied into an internal fixed-size buffer (truncated
   * at kMaxOptionValueLength - 1 characters and null-terminated). The caller
   * does NOT need to keep the source string alive after this call returns.
   *
   * @tparam N The length of the key string (automatically deduced)
   * @param key The option key (must be a string literal or array)
   * @param value The string value to set (copied; truncated at
   *              kMaxOptionValueLength - 1 characters)
   * @return Error::Ok on success, Error::InvalidArgument if storage is full
   */
  template <size_t N>
  Error set_option(const char (&key)[N], const char* value) noexcept {
    static_assert(N <= kMaxOptionKeyLength, "Option key is too long");
    // Create a fixed-size array and copy the string
    std::array<char, kMaxOptionValueLength> arr{};
    strncpy(arr.data(), value, kMaxOptionValueLength - 1);
    arr[kMaxOptionValueLength - 1] = '\0'; // Ensure null termination
    return set_option_impl(key, arr);
  }
  /**
   * Retrieves an option value by key and type.
   *
   * @tparam T The expected type of the option value (bool, int, int64_t, or
   * const char*)
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
        // Default handling for bool/int/int64_t
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
   * @tparam T The type of the value (bool, int, int64_t, or const char*)
   * @param key The option key
   * @param value The value to set
   * @return Error::Ok on success, Error::InvalidArgument if storage is full
   */
  template <typename T>
  Error set_option_impl(const char* key, T value) {
    static_assert(
        std::variant_size_v<OptionValue> == 4,
        "OptionValue arm count changed; audit set_option_impl + get_option");
    static_assert(
        std::is_same_v<T, bool> || std::is_same_v<T, int> ||
            std::is_same_v<T, int64_t> ||
            std::is_same_v<T, std::array<char, kMaxOptionValueLength>>,
        "set_option_impl<T> only supports the variant arms: bool, int, "
        "int64_t, and the fixed-size string array");
    // Update existing if found
    for (size_t i = 0; i < size_; ++i) {
      if (strcmp(options_[i].key, key) == 0) {
        options_[i].value = value;
        return Error::Ok;
      }
    }
    if (size_ < MaxCapacity) {
      BackendOption new_option;
      const size_t key_len = std::strlen(key);
      const size_t copy_len = std::min(key_len, kMaxOptionKeyLength - 1);
      std::memcpy(new_option.key, key, copy_len);
      new_option.key[copy_len] = '\0';
      new_option.value = value; // Restored value assignment
      options_[size_++] = new_option; // Store option and increment size
      return Error::Ok;
    }
    return Error::InvalidArgument;
  }
};

} // namespace runtime
} // namespace executorch
