// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/cpu/runtime/KernelProvider.h>

#include <string>

namespace executorch::backends::cpu {

struct PlanStep {
  KernelProvider* provider;
  KernelImplementation* implementation;
  KernelRegion region;
  std::unique_ptr<Executable> executable;
  StorageRequirements storage;
};

/// One static prepared plan. Graph, bindings and allocator outlive the plan.
/// select is a metadata-only phase; prepare consumes validated constant
/// bindings. The host serializes preparation and invocation, including
/// workspace use.
class CPUPlan {
 public:
  CPUPlan(
      const ptn::Graph& graph,
      std::vector<Buffer>& buffers,
      RuntimeConfiguration configuration,
      ExecutionContext execution);
  ~CPUPlan();
  CPUPlan(const CPUPlan&) = delete;
  CPUPlan& operator=(const CPUPlan&) = delete;
  CPUPlan(CPUPlan&&) = delete;
  CPUPlan& operator=(CPUPlan&&) = delete;

  runtime::Error select();
  runtime::Error prepare(runtime::MemoryAllocator& allocator);
  runtime::Error execute(const ExecutionContext& context);
  std::string describe(double preparation_ms) const;

  const std::vector<PlanStep>& steps() const {
    return steps_;
  }
  const std::vector<BufferRequirements>& requirements() const {
    return requirements_;
  }
  size_t arena_bytes() const {
    return arena_bytes_;
  }
  size_t scratch_bytes() const {
    return scratch_.readable_bytes;
  }
  size_t private_constant_bytes() const {
    return private_constant_bytes_;
  }

 private:
  runtime::Error allocate_activations(runtime::MemoryAllocator& allocator);
  const ptn::Graph& graph_;
  std::vector<Buffer>& buffers_;
  RuntimeConfiguration configuration_;
  ExecutionContext execution_;
  std::vector<std::unique_ptr<KernelProvider>> providers_;
  std::vector<std::vector<KernelImplementation*>> implementations_;
  std::vector<PlanStep> steps_;
  std::vector<BufferRequirements> requirements_;
  std::string decisions_;
  Buffer scratch_;
  size_t arena_bytes_ = 0;
  size_t private_constant_bytes_ = 0;
  bool prepared_ = false;
  bool preparation_started_ = false;
};
} // namespace executorch::backends::cpu
