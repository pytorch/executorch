/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/core/error.h>
#include <cstddef>
#include <cstring>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/array_ref.h>

namespace executorch {
namespace runtime {

// Strongly-typed option key template
template <typename T>
struct OptionKey {
  const char* key;
  constexpr explicit OptionKey(const char* k) : key(k) {}
};

enum class OptionType { BOOL, INT, STRING };

// Union for option values
union OptionValue {
  bool bool_value;
  int64_t int_value;
  const char* string_value;
};

struct BackendOption {
  const char* key;
  OptionType type;
  OptionValue value;
};

template <size_t MaxCapacity>
class BackendOptions {
 public:
  // Initialize with zero options
  BackendOptions() : size_(0) {}

  // Type-safe setters ---------------------------------------------------

  /// Sets or updates a boolean option
  /// @param key: Typed option key
  /// @param value: Boolean value to set
  void set_option(OptionKey<bool> key, bool value) {
    OptionValue v;
    v.bool_value = value; // Direct member assignment
    set_option_internal(key.key, OptionType::BOOL, v);
  }

  /// Sets or updates an integer option
  /// @param key: Typed option key
  /// @param value: Integer value to set
  void set_option(OptionKey<int64_t> key, int64_t value) {
    OptionValue v;
    v.int_value = value; // Direct member assignment
    set_option_internal(key.key, OptionType::INT, v);
  }

  /// Sets or updates a string option
  /// @param key: Typed option key
  /// @param value: Null-terminated string value to set
  void set_option(OptionKey<const char*> key, const char* value) {
    OptionValue v;
    v.string_value = value; // Direct member assignment
    set_option_internal(key.key, OptionType::STRING, v);
  }

  // Type-safe getters ---------------------------------------------------

  /// Retrieves a boolean option value
  /// @param key: Typed option key
  /// @param out_value: Reference to store retrieved value
  /// @return: Error code (Ok on success)
  executorch::runtime::Error get_option(OptionKey<bool> key, bool& out_value)
      const {
    OptionValue val;
    executorch::runtime::Error err =
        get_option_internal(key.key, OptionType::BOOL, val);
    if (err == executorch::runtime::Error::Ok) {
      out_value = val.bool_value;
    }
    return err;
  }

  /// Retrieves an integer option value
  /// @param key: Typed option key
  /// @param out_value: Reference to store retrieved value
  /// @return: Error code (Ok on success)
  executorch::runtime::Error get_option(
      OptionKey<int64_t> key,
      int64_t& out_value) const {
    OptionValue val;
    executorch::runtime::Error err =
        get_option_internal(key.key, OptionType::INT, val);
    if (err == executorch::runtime::Error::Ok) {
      out_value = val.int_value;
    }
    return err;
  }

  /// Retrieves an string option value
  /// @param key: Typed option key
  /// @param out_value: Reference to store retrieved value
  /// @return: Error code (Ok on success)
  executorch::runtime::Error get_option(
      OptionKey<const char*> key,
      const char*& out_value) const {
    OptionValue val;
    executorch::runtime::Error err =
        get_option_internal(key.key, OptionType::STRING, val);
    if (err == executorch::runtime::Error::Ok) {
      out_value = val.string_value;
    }
    return err;
  }

  executorch::runtime::ArrayRef<BackendOption> view() const {
    return executorch::runtime::ArrayRef<BackendOption>(options, size);
  }
    
 private:
  BackendOption options_[MaxCapacity]{};
  size_t size_;

  // Internal helper to set/update an option
  void
  set_option_internal(const char* key, OptionType type, OptionValue value) {
    // Update existing key if found
    for (size_t i = 0; i < size_; ++i) {
      if (strcmp(options_[i].key, key) == 0) {
        options_[i].type = type;
        options_[i].value = value;
        return;
      }
    }
    // Add new option if capacity allows
    if (size_ < MaxCapacity) {
      options_[size_] = BackendOption{key, type, value};
      size_++;
    }
  }

  // Internal helper to get an option value with type checking
  executorch::runtime::Error get_option_internal(
      const char* key,
      OptionType expected_type,
      OptionValue& out) const {
    for (size_t i = 0; i < size_; ++i) {
      if (strcmp(options_[i].key, key) == 0) {
        if (options_[i].type != expected_type) {
          return executorch::runtime::Error::InvalidArgument;
        }
        out = options_[i].value;
        return executorch::runtime::Error::Ok;
      }
    }
    return executorch::runtime::Error::NotFound;
  }
};

// Helper functions for creating typed option keys --------------------------
constexpr OptionKey<bool> BoolKey(const char* k) {
  return OptionKey<bool>(k);
}

constexpr OptionKey<int64_t> IntKey(const char* k) {
  return OptionKey<int64_t>(k);
}

constexpr OptionKey<const char*> StrKey(const char* k) {
  return OptionKey<const char*>(k);
}
} // namespace runtime
} // namespace executorch
