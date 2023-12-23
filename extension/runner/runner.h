/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/runtime/executor/program.h>

namespace torch::executor {

class Runner {
 public:
  explicit Runner(
      std::unique_ptr<DataLoader> dataLoader,
      std::unique_ptr<MemoryAllocator> memoryAllocator);
  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  Error run(
      const std::string& methodName,
      const std::vector<EValue>& inputs,
      std::vector<EValue>& outputs);

  Result<const char*> getMethodName(size_t methodIndex) const;

 private:
  Error loadMethod(const std::string& methodName);

 private:
  class Memory {
   public:
    explicit Memory(
        std::vector<size_t> sizes,
        std::shared_ptr<MemoryAllocator> memoryAllocator) {
      plannedBuffers.resize(sizes.size());
      plannedSpans.resize(sizes.size());
      for (size_t i = 0; i < sizes.size(); ++i) {
        plannedBuffers[i].resize(sizes[i]);
        plannedSpans[i] = {plannedBuffers[i].data(), sizes[i]};
      }
      plannedMemory = std::make_unique<HierarchicalAllocator>(
          Span(plannedSpans.data(), plannedSpans.size()));
      memoryManager = std::make_unique<MemoryManager>(
          memoryAllocator.get(), plannedMemory.get());
    }
    /// Returns a pointer to the internal memory manager, the Memory instance
    /// must outlive this pointer.
    std::shared_ptr<MemoryManager> inline getMemoryManager() {
      return memoryManager;
    }

   private:
    std::vector<std::vector<uint8_t>> plannedBuffers;
    std::vector<Span<uint8_t>> plannedSpans;
    std::unique_ptr<HierarchicalAllocator> plannedMemory;
    std::shared_ptr<MemoryManager> memoryManager;
  };

  struct MethodHolder {
    std::unique_ptr<Memory> memory_;
    std::unique_ptr<Method> method_;
  };

 private:
  std::unique_ptr<DataLoader> dataLoader_;
  std::unique_ptr<Program> program_;
  std::shared_ptr<MemoryAllocator> memoryAllocator_;
  std::unordered_map<std::string, MethodHolder> methods_;
};

} // namespace torch::executor
