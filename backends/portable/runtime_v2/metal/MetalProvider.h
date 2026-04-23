/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/Provider.h>

#include <atomic>
#include <memory>
#include <string_view>

// Forward declaration: keep MetalProvider.h pure-C++; the .mm
// implementation includes MetalStream.h.
namespace executorch {
namespace backends {
namespace metal_v2 {
class MetalStream;
}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * Metal Provider — Apple-silicon accelerator using the existing
 * `metal_v2` infrastructure (MetalStream, MetalOpRegistry, MetalOp).
 *
 * Owns a private MetalStream (via MetalStream::create()). Constructable
 * iff MTLCreateSystemDefaultDevice() returns non-nil.
 *
 * Op support is delegated to MetalOpRegistry: can_run() returns Some
 * iff the op's name is registered there. Initial set: aten::add /
 * aten::mul / aten::sub / aten::relu / aten::mm / aten::bmm.
 *
 * Buffers are MetalBuffers wrapping host pointers from
 * MetalStream::alloc() (Apple Silicon unified memory means
 * MTLBuffer.contents() is a directly-addressable host pointer).
 *
 * v1 constraint: at most one non-CPU provider per process. MetalProvider
 * is mutually exclusive with FakeAccelProvider in
 * make_default_providers().
 */
class MetalProvider final : public Provider {
 public:
  MetalProvider();
  ~MetalProvider() override;

  // Returns true iff stream construction succeeded (i.e., MTLDevice
  // present). Use after construction to decide whether to register.
  bool stream_ready() const;

  std::string_view name() const override { return "metal"; }
  bool is_available_on_device() const override { return stream_ready(); }

  bool can_run(const OpDescriptor& op) const override;

  RuntimeContext& context() override { return ctx_; }

  std::unique_ptr<Instance> instantiate() override;

  // Used by MetalInstance to obtain the stream we own.
  ::executorch::backends::metal_v2::MetalStream* stream() { return stream_.get(); }

 private:
  // Tag-only RuntimeContext (we don't carry any per-Provider state in
  // it; the MetalStream IS the per-Provider state and is held directly
  // on this class).
  struct MetalRuntimeContext : public RuntimeContext {};

  std::unique_ptr<::executorch::backends::metal_v2::MetalStream> stream_;
  MetalRuntimeContext ctx_;
  std::atomic<InstanceId> next_instance_id_{0};
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
