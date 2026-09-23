// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>

#include <executorch/runtime/core/error.h>

namespace ptn {
class EngineHost;
}

namespace executorch::extension::native_module {
namespace internal {

using EngineHostFactory = std::shared_ptr<ptn::EngineHost> (*)();

// Registers the linked engine provider used on the first PTN method load.
// Re-registering the same factory is idempotent.
runtime::Error register_engine_host_factory(EngineHostFactory factory);

} // namespace internal
} // namespace executorch::extension::native_module

// Referencing this symbol forces the PTN Module provider out of a static
// archive on linkers that otherwise discard registration-only objects.
extern "C" void executorch_native_module_ptn_link_anchor();
