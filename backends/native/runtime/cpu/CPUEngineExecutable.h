// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <executorch/backends/cpu/runtime/KernelProvider.h>
#include <executorch/backends/native/runtime/Method.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>

namespace ptn {

class EngineExecutable;
class Package;

class CPUConstantStore {
 public:
  executorch::backends::cpu::Buffer acquire(
      const Package& package,
      const std::string& key,
      const executorch::backends::cpu::BufferRequirements& requirements);

  size_t unique_bytes() const {
    return unique_bytes_;
  }
  size_t materialized_bytes() const {
    return materialized_bytes_;
  }

 private:
  struct Constant {
    executorch::extension::MallocMemoryAllocator allocator;
    executorch::backends::cpu::Buffer buffer;
  };
  std::map<std::string, std::vector<std::unique_ptr<Constant>>> constants_;
  size_t unique_bytes_ = 0;
  size_t materialized_bytes_ = 0;
};

std::unique_ptr<EngineExecutable> create_cpu_engine_executable(
    const Method& method,
    const Package& package,
    CPUConstantStore& constants,
    executorch::backends::cpu::RuntimeConfiguration configuration,
    const executorch::backends::cpu::ExecutionContext& execution);

} // namespace ptn
