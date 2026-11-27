/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace qnn {

// HTP fails a graph execute that binds some tensors as registered shared
// memory and others as plain client buffers (QNN SDK 2.48: prefill DMA error
// 1100, decode transport error 1003, or wrong results). A graph that binds any
// tensor in shared memory therefore stages its remaining tensors through
// persistent shared buffers, so every execute is either all shared or all
// client memory. A graph's staged tensors share one allocation, so they are
// registered through one shared-memory file descriptor.
inline bool NeedsStaging(size_t shared_tensors, size_t total_tensors) {
  return shared_tensors > 0 && shared_tensors < total_tensors;
}

// Offsets of tensors of `sizes` packed into one block, each aligned to
// `alignment`; returns the block's size.
inline size_t PackStaging(
    const std::vector<size_t>& sizes,
    size_t alignment,
    std::vector<size_t>& offsets) {
  offsets.clear();
  size_t total = 0;
  for (size_t nbytes : sizes) {
    total = (total + alignment - 1) / alignment * alignment;
    offsets.push_back(total);
    total += nbytes;
  }
  return total;
}

// One persistent staging block per key (a graph), holding all its staged
// tensors so they share one shared-memory allocation; freed on destruction.
class StagingBlocks {
 public:
  using Allocate = std::function<void*(size_t nbytes)>;
  using Free = std::function<void(void* data)>;

  StagingBlocks(Allocate allocate, Free free)
      : allocate_(std::move(allocate)), free_(std::move(free)) {}
  StagingBlocks(const StagingBlocks&) = delete;
  StagingBlocks& operator=(const StagingBlocks&) = delete;
  ~StagingBlocks() {
    for (const auto& [key, block] : blocks_) {
      free_(block.data);
    }
  }

  // The key's block of nbytes, allocated on first use; nullptr if the
  // allocation fails or the key already holds a block of another size.
  void* Get(const std::string& key, size_t nbytes) {
    auto it = blocks_.find(key);
    if (it != blocks_.end()) {
      return it->second.nbytes == nbytes ? it->second.data : nullptr;
    }
    void* data = allocate_(nbytes);
    if (data != nullptr) {
      blocks_.emplace(key, Block{data, nbytes});
    }
    return data;
  }

  size_t size() const {
    return blocks_.size();
  }

 private:
  struct Block {
    void* data;
    size_t nbytes;
  };
  Allocate allocate_;
  Free free_;
  std::unordered_map<std::string, Block> blocks_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
