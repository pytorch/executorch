/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/arm/runtime/VGFVulkanFeatures.h>

namespace executorch {
namespace backends {
namespace vgf {
namespace {

TEST(VgfVulkanFeaturesTest, EnablesDataGraphShaderModule) {
  VkPhysicalDeviceTensorFeaturesARM next{};
  next.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TENSOR_FEATURES_ARM;

  const auto features = make_vgf_data_graph_features(&next);

  EXPECT_EQ(
      features.sType,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM);
  EXPECT_EQ(features.pNext, &next);
  EXPECT_EQ(features.dataGraph, VK_TRUE);
  EXPECT_EQ(features.dataGraphShaderModule, VK_TRUE);
}

// cppcheck-suppress syntaxError
TEST(VgfVulkanFeaturesTest, RequiresDataGraphShaderModuleSupport) {
  VkPhysicalDeviceDataGraphFeaturesARM available{};
  available.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DATA_GRAPH_FEATURES_ARM;

  EXPECT_FALSE(vgf_data_graph_features_supported(available));

  available.dataGraph = VK_TRUE;
  EXPECT_FALSE(vgf_data_graph_features_supported(available));

  available.dataGraphShaderModule = VK_TRUE;
  EXPECT_TRUE(vgf_data_graph_features_supported(available));
}

} // namespace
} // namespace vgf
} // namespace backends
} // namespace executorch
