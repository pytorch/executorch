// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace ptn {

// Accounts for constants materialized by Vulkan executables in one engine
// context.
//
// A package identity plus its canonical owner key remains stable when raw
// weight storage is acquired and released repeatedly.
//
// The tracker does not own or deduplicate Vulkan resources. Each record() call
// represents one executable materializing a constant from its source bytes.
//
// Not thread-safe: the owning engine must not compile methods concurrently.
class VulkanConstantMaterializationTracker {
 private:
  struct Key {
    uint64_t package_id = 0;
    std::string owner;

    bool operator==(const Key&) const = default;
  };

  struct KeyHash {
    size_t operator()(const Key& key) const;
  };

  struct Entry {
    size_t nbytes = 0;
  };

  std::unordered_map<Key, Entry, KeyHash> entries_;
  size_t unique_constant_bytes_ = 0;
  size_t materialized_constant_bytes_ = 0;

 public:
  // Source bytes across distinct package constants.
  size_t unique_constant_bytes() const {
    return unique_constant_bytes_;
  }

  // Source bytes handed to Vulkan materialization, counting repetitions.
  size_t materialized_constant_bytes() const {
    return materialized_constant_bytes_;
  }

  size_t num_constants() const {
    return entries_.size();
  }

  // True when an earlier executable materialized the same package constant.
  bool record(uint64_t package_id, const std::string& owner, size_t nbytes);
};

} // namespace ptn
