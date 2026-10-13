// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// The Vulkan engine for ptn_runner_main.cpp.

#include <executorch/backends/native/runtime/vulkan/VulkanEngine.h>
#include <executorch/backends/native/test/runtime/PtnRunner.h>

namespace ptn::testing {

std::unique_ptr<EngineHost> make_engine_host() {
  return VulkanEngineHost::create();
}

} // namespace ptn::testing
