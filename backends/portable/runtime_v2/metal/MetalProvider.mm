/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/portable/runtime_v2/metal/MetalProvider.h>

#include <executorch/backends/portable/runtime_v2/metal/MetalInstance.h>

#include <executorch/backends/portable/runtime/metal_v2/MetalOpRegistry.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>

#include <executorch/runtime/platform/log.h>

#include <string>

namespace executorch {
namespace backends {
namespace portable_v2 {

MetalProvider::MetalProvider() {
  // MetalStream::create() returns nullptr internally if MTLDevice
  // construction fails (no Metal-capable GPU). The wrapping unique_ptr
  // would still be valid even if the underlying state isn't fully
  // constructed; check stream_ready() before using.
  stream_ = ::executorch::backends::metal_v2::MetalStream::create();
  if (!stream_) {
    ET_LOG(Info, "MetalProvider: MetalStream::create() returned nullptr; "
                 "no Metal-capable device available");
  }
}

MetalProvider::~MetalProvider() = default;

bool MetalProvider::stream_ready() const {
  return stream_ != nullptr && stream_->device() != nil;
}

bool MetalProvider::can_run(const OpDescriptor& op) const {
  if (!stream_ready()) return false;
  std::string n(op.name);
  return ::executorch::backends::metal_v2::MetalOpRegistry::shared().hasOp(n);
  // dtype-level filtering would happen here once OpDescriptor carries
  // dtype info; for v1 we accept unconditionally (assume Float, which is
  // what the metal_v2 ops default to).
}

std::unique_ptr<Instance> MetalProvider::instantiate() {
  InstanceId id = next_instance_id_.fetch_add(1, std::memory_order_relaxed);
  return std::make_unique<MetalInstance>(this, id);
}

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch
