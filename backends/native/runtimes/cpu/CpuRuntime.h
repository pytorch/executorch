/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Runtime.h>
#include <executorch/backends/native/runtimes/cpu/CpuRuntimeContext.h>

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>

namespace executorch {
namespace backends {
namespace native {

/**
 * CPU Runtime — fallback runtime that supports every op in the existing
 * portable kernel registry. Always available; expected at index 0 (host
 * slot invariant).
 *
 * Configurable for testing: pass a name override and/or an op allowlist
 * to spin up a SECOND CpuRuntime that pretends to be a different
 * runtime for multi-provider routing tests. (e.g. "fake_accel" with
 * supported_ops = {"aten::add", "aten::mul"}.)
 */
class CpuRuntime final : public Runtime {
 public:
  CpuRuntime() = default;

  // Test/dev constructor: override the registered name and restrict ops.
  // If supported_ops is empty, behaves like the default ("accept all
  // registered ops"). If non-empty, only ops in the set are accepted.
  CpuRuntime(
      std::string_view name,
      std::unordered_set<std::string> supported_ops)
      : name_(name), supported_ops_(std::move(supported_ops)) {}

  ~CpuRuntime() override = default;

  std::string_view name() const override {
    return name_;
  }

  bool is_available_on_device() const override {
    return true;
  }

  bool can_run(const OpDescriptor& op) const override;

  RuntimeContext& context() override {
    return ctx_;
  }

  std::unique_ptr<Engine> instantiate() override;

 private:
  CpuRuntimeContext ctx_;
  std::atomic<InstanceId> next_instance_id_{0};
  std::string_view name_ = "cpu";
  // Empty = accept any op the portable kernel registry can dispatch.
  // Non-empty = only the listed names.
  std::unordered_set<std::string> supported_ops_;
};

} // namespace native
} // namespace backends
} // namespace executorch
