/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cm_utils.h"

#include <executorch/backends/vulkan/runtime/api/Context.h>
#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {

static std::string componentTypeToString(VkComponentTypeKHR type) {
  switch (type) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR:
      return "float16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR:
      return "float32";
    case VK_COMPONENT_TYPE_FLOAT64_KHR:
      return "float64";
    case VK_COMPONENT_TYPE_SINT8_KHR:
      return "int8";
    case VK_COMPONENT_TYPE_SINT16_KHR:
      return "int16";
    case VK_COMPONENT_TYPE_SINT32_KHR:
      return "int32";
    case VK_COMPONENT_TYPE_SINT64_KHR:
      return "int64";
    case VK_COMPONENT_TYPE_UINT8_KHR:
      return "uint8";
    case VK_COMPONENT_TYPE_UINT16_KHR:
      return "uint16";
    case VK_COMPONENT_TYPE_UINT32_KHR:
      return "uint32";
    case VK_COMPONENT_TYPE_UINT64_KHR:
      return "uint64";
    default:
      return "unknown(" + std::to_string(static_cast<int>(type)) + ")";
  }
}

static std::string scopeToString(VkScopeKHR scope) {
  switch (scope) {
    case VK_SCOPE_DEVICE_KHR:
      return "Device";
    case VK_SCOPE_WORKGROUP_KHR:
      return "Workgroup";
    case VK_SCOPE_SUBGROUP_KHR:
      return "Subgroup";
    case VK_SCOPE_QUEUE_FAMILY_KHR:
      return "QueueFamily";
    default:
      return "unknown(" + std::to_string(static_cast<int>(scope)) + ")";
  }
}

void queryCooperativeMatrixProperties() {
#ifdef VK_KHR_cooperative_matrix
  auto* adapter = vkcompute::api::context()->adapter_ptr();
  VkPhysicalDevice physicalDevice = adapter->physical_handle();

  if (!adapter->supports_cooperative_matrix()) {
    std::cout << "VK_KHR_cooperative_matrix is NOT supported on this device."
              << std::endl;
    return;
  }

  std::cout << "\n=== Cooperative Matrix Properties (KHR) ===" << std::endl;

  uint32_t count = 0;
  VkResult result = vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
      physicalDevice, &count, nullptr);

  if (result != VK_SUCCESS || count == 0) {
    std::cout << "No cooperative matrix configurations found." << std::endl;
    return;
  }

  std::vector<VkCooperativeMatrixPropertiesKHR> properties(count);
  for (auto& prop : properties) {
    prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    prop.pNext = nullptr;
  }

  result = vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
      physicalDevice, &count, properties.data());

  if (result != VK_SUCCESS) {
    std::cerr << "Failed to query cooperative matrix properties." << std::endl;
    return;
  }

  std::cout << "Found " << count << " cooperative matrix configurations:\n"
            << std::endl;

  std::cout << std::left << std::setw(5) << "#" << std::setw(10) << "M"
            << std::setw(10) << "N" << std::setw(10) << "K" << std::setw(12)
            << "AType" << std::setw(12) << "BType" << std::setw(12) << "CType"
            << std::setw(12) << "ResultType" << std::setw(12) << "Scope"
            << std::endl;

  std::cout << std::string(95, '-') << std::endl;

  for (uint32_t i = 0; i < count; ++i) {
    const auto& p = properties[i];
    std::cout << std::left << std::setw(5) << i << std::setw(10) << p.MSize
              << std::setw(10) << p.NSize << std::setw(10) << p.KSize
              << std::setw(12) << componentTypeToString(p.AType)
              << std::setw(12) << componentTypeToString(p.BType)
              << std::setw(12) << componentTypeToString(p.CType)
              << std::setw(12) << componentTypeToString(p.ResultType)
              << std::setw(12) << scopeToString(p.scope) << std::endl;
  }

  std::cout << std::endl;

#else
  std::cout << "VK_KHR_cooperative_matrix not available at compile time."
            << std::endl;
#endif
}

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
