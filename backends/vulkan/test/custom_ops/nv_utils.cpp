// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "nv_utils.h"

#include <executorch/backends/vulkan/runtime/vk_api/Runtime.h>

#include <cstdio>
#include <iostream>
#include <vector>

namespace executorch {
namespace vulkan {
namespace prototyping {

using namespace vkcompute;


// Helper function to convert VkComponentTypeKHR to string
const char* componentTypeToString(VkComponentTypeKHR type) {
  switch (type) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR: return "float16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR: return "float32";
    case VK_COMPONENT_TYPE_FLOAT64_KHR: return "float64";
    case VK_COMPONENT_TYPE_SINT8_KHR: return "int8";
    case VK_COMPONENT_TYPE_SINT16_KHR: return "int16";
    case VK_COMPONENT_TYPE_SINT32_KHR: return "int32";
    case VK_COMPONENT_TYPE_SINT64_KHR: return "int64";
    case VK_COMPONENT_TYPE_UINT8_KHR: return "uint8";
    case VK_COMPONENT_TYPE_UINT16_KHR: return "uint16";
    case VK_COMPONENT_TYPE_UINT32_KHR: return "uint32";
    case VK_COMPONENT_TYPE_UINT64_KHR: return "uint64";
    default: return "unknown";
  }
}

// Helper function to convert VkScopeKHR to string
const char* scopeToString(VkScopeKHR scope) {
  switch (scope) {
    case VK_SCOPE_DEVICE_KHR: return "Device";
    case VK_SCOPE_WORKGROUP_KHR: return "Workgroup";
    case VK_SCOPE_SUBGROUP_KHR: return "Subgroup";
    case VK_SCOPE_QUEUE_FAMILY_KHR: return "QueueFamily";
    default: return "unknown";
  }
}

// Query and print cooperative matrix properties
void queryCooperativeMatrixProperties() {
  std::cout << "\n=== Cooperative Matrix Properties ===" << std::endl;

  VkPhysicalDevice physicalDevice = vkcompute::api::context()->adapter_ptr()->physical_handle();
  VkInstance instance = vkcompute::vkapi::runtime()->instance();

  // Query KHR cooperative matrix properties
  PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR =
      reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
          vkGetInstanceProcAddr(
              instance,
              "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));

  if (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR == nullptr) {
    std::cout << "VK_KHR_cooperative_matrix extension not available." << std::endl;
    return;
  }

  // Get count of supported matrix configurations
  uint32_t propCount = 0;
  VkResult result = vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
      physicalDevice, &propCount, nullptr);

  if (result != VK_SUCCESS || propCount == 0) {
    std::cout << "No cooperative matrix configurations supported." << std::endl;
    return;
  }

  // Allocate and query properties
  std::vector<VkCooperativeMatrixPropertiesKHR> matrixProps(propCount);
  for (auto& prop : matrixProps) {
    prop.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    prop.pNext = nullptr;
  }

  result = vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(
      physicalDevice, &propCount, matrixProps.data());

  if (result != VK_SUCCESS) {
    std::cout << "Failed to query cooperative matrix properties." << std::endl;
    return;
  }

  std::cout << "Found " << propCount << " cooperative matrix configurations:" << std::endl;
  std::cout << "----------------------------------------------------------------------" << std::endl;
  std::cout << "  #  |   M  |   N  |   K  | A Type  | B Type  | C Type  | R Type  | Scope" << std::endl;
  std::cout << "----------------------------------------------------------------------" << std::endl;

  for (uint32_t i = 0; i < propCount; ++i) {
    const auto& prop = matrixProps[i];
    printf(" %3u | %4u | %4u | %4u | %-7s | %-7s | %-7s | %-7s | %s\n",
        i,
        prop.MSize, prop.NSize, prop.KSize,
        componentTypeToString(prop.AType),
        componentTypeToString(prop.BType),
        componentTypeToString(prop.CType),
        componentTypeToString(prop.ResultType),
        scopeToString(prop.scope));
  }

  std::cout << "----------------------------------------------------------------------" << std::endl;

  // Filter and print configurations useful for FP32 linear (float A, float B, float C)
  std::cout << "\nConfigurations with float32 A, B, C types:" << std::endl;
  for (uint32_t i = 0; i < propCount; ++i) {
    const auto& prop = matrixProps[i];
    if (prop.AType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
        prop.BType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
        prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
      printf("  M=%u, N=%u, K=%u, Scope=%s\n",
          prop.MSize, prop.NSize, prop.KSize,
          scopeToString(prop.scope));
    }
  }

  // Filter and print configurations useful for FP16 input with FP32 accumulator
  std::cout << "\nConfigurations with float16 A/B, float32 C (mixed precision):" << std::endl;
  for (uint32_t i = 0; i < propCount; ++i) {
    const auto& prop = matrixProps[i];
    if (prop.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
        prop.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
        prop.CType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
      printf("  M=%u, N=%u, K=%u, Scope=%s\n",
          prop.MSize, prop.NSize, prop.KSize,
          scopeToString(prop.scope));
    }
  }

  std::cout << std::endl;

}

} // namespace prototyping
} // namespace vulkan
} // namespace executorch
