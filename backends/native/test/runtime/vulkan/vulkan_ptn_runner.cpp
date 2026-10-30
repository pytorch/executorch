// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// The Vulkan engine host for PTN test drivers.

#include <stdexcept>

#include <executorch/backends/native/runtime/vulkan/VulkanEngine.h>
#include <executorch/backends/native/test/runtime/PtnRunner.h>

namespace ptn::testing {

// cppcheck-suppress unusedFunction
std::unique_ptr<EngineHost> make_engine_host(const HostOptions& options) {
  if (options.allow_activation_quant_for_weight_only) {
    throw std::invalid_argument(
        "vulkan: never quantizes activations, so it rejects "
        "allow_activation_quant_for_weight_only");
  }
  return VulkanEngineHost::create();
}

} // namespace ptn::testing
