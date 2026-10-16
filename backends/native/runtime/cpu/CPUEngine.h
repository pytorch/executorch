// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <executorch/backends/cpu/runtime/KernelProvider.h>
#include <executorch/backends/native/runtime/engine/Engine.h>

namespace ptn {

class CPUEngineContext;

// Reuses CPUPlan and its linked providers. The host must outlive its contexts
// and executables, as required by EngineHost.
class CPUEngineHost final : public EngineHost {
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit CPUEngineHost(std::unique_ptr<Impl> impl);

 public:
  ~CPUEngineHost() override;

  // An empty provider list selects XNNPACK and Reference. Copies the supplied
  // factory list and preferences; their source storage need not outlive this
  // call.
  static std::unique_ptr<CPUEngineHost> create(
      executorch::backends::cpu::RuntimeConfiguration configuration = {});

  const std::string& name() const override;
  const std::string& device_name() const override;

  std::unique_ptr<EngineContext> create_context(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package) override;

  std::unique_ptr<CPUEngineContext> create_cpu_context(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package);
};

// Static, unquantized graphs with contiguous constants and public I/O. Native
// view/permute/transpose aliases and contiguous full-span as_strided reshapes
// are materialized; relu_/gelu_/sigmoid_ are functionalized when all live
// aliases retain identical logical indexing.
// Persistent mutation, dynamic shapes and control flow are rejected. Operator
// and dtype support comes from the selected providers. Contexts are isolated.
class CPUEngineContext final : public EngineContext {
 private:
  friend class CPUEngineHost;
  struct Impl;
  std::unique_ptr<Impl> impl_;

  CPUEngineContext(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package,
      std::unique_ptr<Impl> impl);

  std::unique_ptr<EngineExecutable> compile_method(
      const Method& method) override;

 public:
  ~CPUEngineContext() override;

  // Logical source bytes, excluding allocator padding and provider prepacking.
  // Constants with compatible storage requirements are shared across methods.
  size_t unique_constant_bytes() const;
  size_t materialized_constant_bytes() const;
};

} // namespace ptn
