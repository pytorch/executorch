/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

#include <executorch/runtime/executor/program.h>

namespace torch::executor::demo {

class Module {
 public:
  explicit Module(const std::string& filePath);
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;

  Error forward(
      const std::vector<EValue>& inputs,
      std::vector<EValue>& outputs);

 private:
  std::unique_ptr<DataLoader> dataLoader_;
  std::unique_ptr<Program> program_;
  std::unique_ptr<MemoryAllocator> methodAllocator_;
  std::vector<std::vector<uint8_t>> plannedBuffers_;
  std::vector<Span<uint8_t>> plannedSpans_;
  std::unique_ptr<HierarchicalAllocator> plannedMemory_;
  std::unique_ptr<MemoryManager> memoryManager_;
  std::unique_ptr<Method> method_;
};

} // namespace torch::executor::demo
