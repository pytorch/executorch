// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace vkcompute {
struct GraphConfig;
}

namespace ptn {

class EngineExecutable;
struct Method;
class Package;
class VulkanConstantMaterializationTracker;

// Non-empty mutable DataBinding keys held by the live executables of one
// context, each mapped to the method whose executable holds it.
using VulkanMutableStateOwners = std::unordered_map<std::string, std::string>;

std::unique_ptr<EngineExecutable> create_vulkan_engine_executable(
    const Method& method,
    const Package& package,
    VulkanConstantMaterializationTracker& materializations,
    VulkanMutableStateOwners& mutable_state_owners,
    const vkcompute::GraphConfig& config);

} // namespace ptn
