/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/backend/backend_options.h>
#include <executorch/runtime/core/error.h>
#include <cstring>

#pragma once
namespace executorch {
namespace runtime {

struct Entry {
  const char* backend_name;
  ArrayRef<BackendOption> options;
};

template <size_t MaxBackends>
class BackendOptionsMap {
 public:
  // Default constructor
  BackendOptionsMap() : size_(0) {}

  // Add a new backend configuration
  Error add(
      const char* backend_name,
      ::executorch::runtime::ArrayRef<BackendOption> options) {
    if (size_ < MaxBackends) {
      entries_[size_] = {backend_name, options};
      ++size_;
      return Error::Ok;
    } else {
      ET_LOG(Error, "Maximum number of backends %lu reached", MaxBackends);
    }
    return Error::InvalidArgument;
  }

  // Get options for a specific backend
  ::executorch::runtime::ArrayRef<BackendOption> get(
      const char* backend_name) const {
    for (size_t i = 0; i < size_; ++i) {
      if (std::strcmp(entries_[i].backend_name, backend_name) == 0) {
        return entries_[i].options;
      }
    }
    return {}; // Return empty ArrayRef if not found
  }

  // Get a view of the entries (const version)
  ::executorch::runtime::ArrayRef<const Entry> entries() const {
    return ::executorch::runtime::ArrayRef<const Entry>(entries_, size_);
  }

  // Get a view of the entries (non-const version)
  ::executorch::runtime::ArrayRef<Entry> entries() {
    return ::executorch::runtime::ArrayRef<Entry>(entries_, size_);
  }

  // Get number of entries
  size_t size() const {
    return size_;
  }

 private:
  Entry entries_[MaxBackends]; // Storage for backend entries
  size_t size_ = 0; // Current number of entries
};

} // namespace runtime
} // namespace executorch
