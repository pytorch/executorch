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

TEST(VgfVulkanFeaturesTest, EnablesHostMemoryImportWhenRequestedAndAdvertised) {
  EXPECT_TRUE(vgf_host_memory_import_should_be_enabled(
      /*requested=*/true, /*physical_device_advertised=*/true));
}

TEST(
    VgfVulkanFeaturesTest,
    DoesNotEnableHostMemoryImportWhenAdvertisedButNotRequested) {
  EXPECT_FALSE(vgf_host_memory_import_should_be_enabled(
      /*requested=*/false, /*physical_device_advertised=*/true));
}

TEST(
    VgfVulkanFeaturesTest,
    DoesNotEnableHostMemoryImportWhenRequestedButNotAdvertised) {
  EXPECT_FALSE(vgf_host_memory_import_should_be_enabled(
      /*requested=*/true, /*physical_device_advertised=*/false));
}

TEST(VgfVulkanFeaturesTest, LegacyHostMemoryImportPathRemainsDisabled) {
  EXPECT_FALSE(vgf_host_memory_import_should_be_enabled(
      /*requested=*/false, /*physical_device_advertised=*/false));
}

TEST(VgfVulkanFeaturesTest, HostImportCommandPoolSupportsIndividualReset) {
  EXPECT_EQ(
      vgf_command_pool_flags(/*host_memory_import_enabled=*/true),
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
}

TEST(VgfVulkanFeaturesTest, LegacyCommandPoolFlagsRemainUnchanged) {
  EXPECT_EQ(vgf_command_pool_flags(/*host_memory_import_enabled=*/false), 0u);
}

TEST(VgfVulkanFeaturesTest, HostMemoryImportCapabilitiesDefaultToUnavailable) {
  const VgfHostMemoryImportCapabilities capabilities{};
  EXPECT_FALSE(capabilities.physical_device_advertised);
  EXPECT_FALSE(capabilities.logical_device_enabled);
  EXPECT_EQ(capabilities.min_imported_host_pointer_alignment, 0u);
}

TEST(
    VgfVulkanFeaturesTest,
    HostMemoryImportCapabilitiesDistinguishAdvertisedFromEnabled) {
  VgfHostMemoryImportCapabilities capabilities{};
  capabilities.physical_device_advertised = true;
  capabilities.logical_device_enabled = false;
  capabilities.min_imported_host_pointer_alignment = 4096;

  EXPECT_TRUE(capabilities.physical_device_advertised);
  EXPECT_FALSE(capabilities.logical_device_enabled);
  EXPECT_EQ(capabilities.min_imported_host_pointer_alignment, 4096u);
}

} // namespace
} // namespace vgf
} // namespace backends
} // namespace executorch
