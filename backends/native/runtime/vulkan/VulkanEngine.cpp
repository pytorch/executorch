// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/VulkanEngine.h>

#include <stdexcept>
#include <string>
#include <utility>

#include <executorch/backends/native/runtime/vulkan/VulkanConstantMaterializationTracker.h>
#include <executorch/backends/native/runtime/vulkan/VulkanEngineExecutable.h>
#include <executorch/backends/vulkan/runtime/graph/GraphConfig.h>
#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>

namespace ptn {

using vkcompute::GraphConfig;
namespace vkapi = vkcompute::vkapi;

struct VulkanEngineHost::Impl {
  vkapi::Adapter* adapter = nullptr;
  std::string name = "vulkan";
  std::string device_name;
};

struct VulkanEngineContext::Impl {
  GraphConfig config;
  VulkanConstantMaterializationTracker materializations;
};

VulkanEngineHost::VulkanEngineHost(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

VulkanEngineHost::~VulkanEngineHost() = default;

std::unique_ptr<VulkanEngineHost> VulkanEngineHost::create() {
  std::unique_ptr<Impl> impl = std::make_unique<Impl>();
  try {
    // Brings up the VkInstance and the default adapter on first call. Note this
    // deliberately avoids api::available() / api::context(), which would
    // construct a whole global api::Context -- command pool, descriptor pool
    // and fences -- that nothing uses, since every ComputeGraph builds its own.
    impl->adapter = vkapi::runtime()->get_adapter_p();
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("vulkan: no Vulkan device available (is a driver or "
                    "SwiftShader present?): ") +
        e.what());
  }
  if (impl->adapter == nullptr) {
    throw std::runtime_error("vulkan: no Vulkan device available");
  }
  impl->device_name = impl->adapter->device_name();
  return std::unique_ptr<VulkanEngineHost>(
      new VulkanEngineHost(std::move(impl)));
}

const std::string& VulkanEngineHost::name() const {
  return impl_->name;
}

const std::string& VulkanEngineHost::device_name() const {
  return impl_->device_name;
}

std::unique_ptr<EngineContext> VulkanEngineHost::create_context(
    std::shared_ptr<const Program> program,
    std::shared_ptr<const Package> package) {
  return create_vulkan_context(std::move(program), std::move(package));
}

std::unique_ptr<VulkanEngineContext> VulkanEngineHost::create_vulkan_context(
    std::shared_ptr<const Program> program,
    std::shared_ptr<const Package> package) {
  std::unique_ptr<VulkanEngineContext::Impl> impl =
      std::make_unique<VulkanEngineContext::Impl>();
  return std::unique_ptr<VulkanEngineContext>(new VulkanEngineContext(
      std::move(program), std::move(package), std::move(impl)));
}

VulkanEngineContext::VulkanEngineContext(
    std::shared_ptr<const Program> program,
    std::shared_ptr<const Package> package,
    std::unique_ptr<Impl> impl)
    : EngineContext(std::move(program), std::move(package)),
      impl_(std::move(impl)) {}

VulkanEngineContext::~VulkanEngineContext() = default;

size_t VulkanEngineContext::unique_constant_bytes() const {
  return impl_->materializations.unique_constant_bytes();
}

size_t VulkanEngineContext::materialized_constant_bytes() const {
  return impl_->materializations.materialized_constant_bytes();
}

std::unique_ptr<EngineExecutable> VulkanEngineContext::compile_method(
    const Method& method) {
  return create_vulkan_engine_executable(
      method, package(), impl_->materializations, impl_->config);
}

} // namespace ptn
