// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/VulkanConstantMaterializationTracker.h>

#include <functional>
#include <stdexcept>

namespace ptn {

size_t VulkanConstantMaterializationTracker::KeyHash::operator()(
    const Key& key) const {
  const size_t package_hash = std::hash<uint64_t>()(key.package_id);
  const size_t owner_hash = std::hash<std::string>()(key.owner);
  return package_hash ^
      (owner_hash + 0x9e3779b9 + (package_hash << 6) + (package_hash >> 2));
}

bool VulkanConstantMaterializationTracker::record(
    uint64_t package_id,
    const std::string& owner,
    size_t nbytes) {
  const auto inserted = entries_.emplace(Key{package_id, owner}, Entry{});
  Entry& entry = inserted.first->second;
  if (!inserted.second) {
    if (entry.nbytes != nbytes) {
      throw std::runtime_error(
          "vulkan constant tracker: byte count changed for package " +
          std::to_string(package_id) + " constant '" + owner + "': recorded " +
          std::to_string(entry.nbytes) + ", got " + std::to_string(nbytes));
    }
    materialized_constant_bytes_ += nbytes;
    return true;
  }

  entry.nbytes = nbytes;
  unique_constant_bytes_ += nbytes;
  materialized_constant_bytes_ += nbytes;
  return false;
}

} // namespace ptn
