/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/Provider.h>
#include <executorch/backends/portable/runtime_v2/cpu/CpuRuntimeContext.h>

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * CPU Provider — fallback runtime that supports every op in the existing
 * portable kernel registry. Always available; expected at index 0 (host
 * slot invariant).
 *
 * Configurable for testing: pass a name override and/or an op allowlist
 * to spin up a SECOND CpuProvider that pretends to be a different
 * runtime for multi-provider routing tests. (e.g. "fake_accel" with
 * supported_ops = {"aten::add", "aten::mul"}.)
 */
class CpuProvider final : public Provider {
 public:
  CpuProvider() = default;

  // Test/dev constructor: override the registered name and restrict ops.
  // If supported_ops is empty, behaves like the default ("accept all
  // registered ops"). If non-empty, only ops in the set are accepted.
  CpuProvider(std::string_view name,
              std::unordered_set<std::string> supported_ops)
      : name_(name), supported_ops_(std::move(supported_ops)) {}

  ~CpuProvider() override = default;

  std::string_view name() const override { return name_; }

  bool is_available_on_device() const override { return true; }

  bool can_run(const OpDescriptor& op) const override;

  RuntimeContext& context() override { return ctx_; }

  std::unique_ptr<Instance> instantiate() override;

 private:
  CpuRuntimeContext ctx_;
  std::atomic<InstanceId> next_instance_id_{0};
  std::string_view name_ = "cpu";
  // Empty = accept any op the portable kernel registry can dispatch.
  // Non-empty = only the listed names.
  std::unordered_set<std::string> supported_ops_;
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
