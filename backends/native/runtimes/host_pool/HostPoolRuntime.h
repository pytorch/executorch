/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Runtime.h>
#include <executorch/backends/native/runtimes/host_pool/HostPoolEngine.h>
#include <executorch/backends/native/runtimes/host_pool/HostPoolRuntimeContext.h>

#include <atomic>
#include <memory>
#include <string_view>

namespace executorch {
namespace backends {
namespace native {

/**
 * HostPoolRuntime — owns the canonical host-buffer pool for boundary
 * values (graph IO, cross-runtime intermediates). Not a compute runtime:
 * can_run() always returns false, so the router never assigns ops here.
 *
 * Conventionally lives at slot 0 of Plan::providers / Plan::instances.
 * Compute providers (CpuRuntime, MetalRuntime, ...) get per-runtime
 * device-side mirrors of these host buffers via
 * Engine::AllocRequest::mirror_partner.
 */
class HostPoolRuntime final : public Runtime {
 public:
  HostPoolRuntime() = default;
  ~HostPoolRuntime() override = default;

  std::string_view name() const override { return "host"; }

  bool is_available_on_device() const override { return true; }

  // HostPool is an allocator only — it does not execute kernels.
  bool can_run(const OpDescriptor& /*op*/) const override { return false; }

  RuntimeContext& context() override { return ctx_; }

  std::unique_ptr<Engine> instantiate() override {
    InstanceId id = next_instance_id_.fetch_add(1, std::memory_order_relaxed);
    return std::make_unique<HostPoolEngine>(ctx_, id);
  }

 private:
  HostPoolRuntimeContext ctx_;
  std::atomic<InstanceId> next_instance_id_{0};
};

}  // namespace native
}  // namespace backends
}  // namespace executorch
