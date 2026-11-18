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

class CPUEngineHost final : public EngineHost {
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit CPUEngineHost(std::unique_ptr<Impl> impl);

 public:
  ~CPUEngineHost() override;

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

  size_t unique_constant_bytes() const;
  size_t materialized_constant_bytes() const;
};

} // namespace ptn
