// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <executorch/backends/native/runtime/engine/Engine.h>

namespace ptn {

class VulkanEngineContext;

// Process-wide Vulkan runtime and adapter selection. The host must outlive the
// contexts and executables it creates.
class VulkanEngineHost final : public EngineHost {
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit VulkanEngineHost(std::unique_ptr<Impl> impl);

 public:
  ~VulkanEngineHost() override;

  // Bring up (or reuse) the process-wide Vulkan runtime and select its default
  // adapter. Throws std::runtime_error when no Vulkan device is available.
  static std::unique_ptr<VulkanEngineHost> create();

  const std::string& name() const override;
  const std::string& device_name() const override;

  std::unique_ptr<EngineContext> create_context(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package) override;

  std::unique_ptr<VulkanEngineContext> create_vulkan_context(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package);
};

// Per-program Vulkan state. Each executable owns its ComputeGraph and command
// resources while sharing this context's constant materialization tracker.
// Mutable state is not shared yet: compile throws while another live
// executable holds one of the method's mutable DataBinding keys.
class VulkanEngineContext final : public EngineContext {
 private:
  friend class VulkanEngineHost;

  struct Impl;
  std::unique_ptr<Impl> impl_;

  VulkanEngineContext(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package,
      std::unique_ptr<Impl> impl);

  std::unique_ptr<EngineExecutable> compile_method(
      const Method& method) override;

 public:
  ~VulkanEngineContext() override;

  // Source-byte accounting across every executable compiled so far.
  size_t unique_constant_bytes() const;
  size_t materialized_constant_bytes() const;
};

} // namespace ptn
