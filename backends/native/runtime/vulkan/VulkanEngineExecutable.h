// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>

namespace vkcompute {
struct GraphConfig;
}

namespace ptn {

class EngineExecutable;
struct Method;
class Package;
class VulkanConstantMaterializationTracker;

std::unique_ptr<EngineExecutable> create_vulkan_engine_executable(
    const Method& method,
    const Package& package,
    VulkanConstantMaterializationTracker& materializations,
    const vkcompute::GraphConfig& config);

} // namespace ptn
