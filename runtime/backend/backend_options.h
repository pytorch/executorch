/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/error.h>
#include <cstddef>
#include <cstring>
#include <variant>

namespace executorch {
namespace runtime {

// Strongly-typed option key template
template <typename T>
struct OptionKey {
  const char* key;
  constexpr explicit OptionKey(const char* k) : key(k) {}
};

// Union replaced with std::variant
using OptionValue = std::variant<bool, int, const char*>;

struct BackendOption {
  const char* key; // key is the name of the backend option, like num_threads,
                   // enable_profiling, etc
  OptionValue
      value; // value is the value of the backend option, like 4, true, etc
};

template <size_t MaxCapacity>
class BackendOptions {
 public:
  // Initialize with zero options
  BackendOptions() : size_(0) {}

  // Type-safe setters
  template <typename T>
  void set_option(OptionKey<T> key, T value) {
    const char* k = key.key;
    // Update existing if found
    for (size_t i = 0; i < size_; ++i) {
      if (strcmp(options_[i].key, k) == 0) {
        options_[i].value = value;
        return;
      }
    }
    // Add new option if space available
    if (size_ < MaxCapacity) {
      options_[size_++] = BackendOption{k, value};
    }
  }

  // Type-safe getters
  template <typename T>
  Error get_option(OptionKey<T> key, T& out) const {
    const char* k = key.key;
    for (size_t i = 0; i < size_; ++i) {
      if (strcmp(options_[i].key, k) == 0) {
        if (auto* val = std::get_if<T>(&options_[i].value)) {
          out = *val;
          return Error::Ok;
        }
        return Error::InvalidArgument;
      }
    }
    return Error::NotFound;
  }
  executorch::runtime::ArrayRef<BackendOption> view() const {
    return executorch::runtime::ArrayRef<BackendOption>(options_, size_);
  }

 private:
  BackendOption options_[MaxCapacity]{}; // Storage for backend options
  size_t size_; // Current number of options
};

// Helper functions for creating typed option keys (unchanged)
constexpr OptionKey<bool> BoolKey(const char* k) {
  return OptionKey<bool>(k);
}

constexpr OptionKey<int> IntKey(const char* k) {
  return OptionKey<int>(k);
}

constexpr OptionKey<const char*> StrKey(const char* k) {
  return OptionKey<const char*>(k);
}

} // namespace runtime
} // namespace executorch
