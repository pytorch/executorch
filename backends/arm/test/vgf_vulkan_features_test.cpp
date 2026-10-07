/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/arm/runtime/VGFVulkanFeatures.h>
#include <executorch/backends/arm/runtime/VGFZeroCopy.h>

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

VgfZeroCopyIoMetadata make_eligible_linear_tensor_metadata_for_test() {
  const int64_t dimensions[] = {1, 8, 8, 4};
  const int64_t strides[] = {256, 32, 4, 1};
  const VkTensorDescriptionARM description{
      .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
      .pNext = nullptr,
      .tiling = VK_TENSOR_TILING_LINEAR_ARM,
      .format = VK_FORMAT_R32_SFLOAT,
      .dimensionCount = 4,
      .pDimensions = dimensions,
      .pStrides = strides,
      .usage = vgf_tensor_usage_flags(/*image_aliasing=*/false),
  };
  VgfHostMemoryImportCapabilities host_capabilities{};
  host_capabilities.physical_device_advertised = true;
  host_capabilities.logical_device_enabled = true;

  auto metadata = make_vgf_zero_copy_io_metadata(
      /*is_model_boundary_resource=*/true,
      /*is_input=*/true,
      VK_DESCRIPTOR_TYPE_TENSOR_ARM,
      VK_FORMAT_R32_SFLOAT,
      /*tensor_backed=*/true,
      /*has_alias_group=*/false,
      /*has_incompatible_image_aliasing=*/false,
      &description,
      host_capabilities);
  metadata.mapped_to_model_boundary = true;
  metadata.executorch_argument_index = 0;
  return metadata;
}

TEST(VgfZeroCopyMetadataTest, EligibleLinearTensorBoundary) {
  auto metadata = make_eligible_linear_tensor_metadata_for_test();

  EXPECT_EQ(
      metadata.usage,
      VK_TENSOR_USAGE_SHADER_BIT_ARM | VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM |
          VK_TENSOR_USAGE_TRANSFER_DST_BIT_ARM |
          VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM);
  EXPECT_EQ(metadata.dimensions, (std::vector<int64_t>{1, 8, 8, 4}));
  EXPECT_TRUE(metadata.has_explicit_strides);
  EXPECT_EQ(metadata.strides, (std::vector<int64_t>{256, 32, 4, 1}));

#if defined(VK_EXT_external_memory_host)
  VkExternalMemoryProperties properties{};
  properties.externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
  properties.compatibleHandleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
  vgf_cache_external_tensor_properties(&metadata, properties);
#else
  metadata.external_tensor_query_performed = true;
  metadata.external_tensor_host_allocation_supported = true;
#endif

  std::vector<VgfZeroCopyIoMetadata> all_metadata{metadata};
  EXPECT_TRUE(vgf_record_boundary_binding(
      &all_metadata,
      0,
      VgfBoundaryBindingRef{.segment_index = 0, .set_index = 0, .binding = 3}));
  vgf_finalize_zero_copy_io_metadata(&all_metadata[0]);

  EXPECT_TRUE(all_metadata[0].eligible);
}

TEST(VgfZeroCopyMetadataTest, ImageAndAliasedResourcesAreRejected) {
  VgfHostMemoryImportCapabilities host_capabilities{};
  host_capabilities.physical_device_advertised = true;
  host_capabilities.logical_device_enabled = true;

  auto image_metadata = make_vgf_zero_copy_io_metadata(
      /*is_model_boundary_resource=*/true,
      /*is_input=*/true,
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_FORMAT_R32_SFLOAT,
      /*tensor_backed=*/false,
      /*has_alias_group=*/false,
      /*has_incompatible_image_aliasing=*/false,
      nullptr,
      host_capabilities);
  EXPECT_FALSE(vgf_zero_copy_io_structurally_eligible(image_metadata));

  auto aliased_tensor_metadata =
      make_eligible_linear_tensor_metadata_for_test();
  aliased_tensor_metadata.has_alias_group = true;
  EXPECT_FALSE(vgf_zero_copy_io_structurally_eligible(aliased_tensor_metadata));
}

TEST(VgfZeroCopyMetadataTest, MutableEndpointsDoNotShiftExternalArguments) {
  std::vector<VgfZeroCopyIoMetadata> metadata(
      5, make_eligible_linear_tensor_metadata_for_test());
  for (size_t i = 0; i < metadata.size(); ++i) {
    auto& entry = metadata[i];
    entry.mapped_to_model_boundary = false;
    entry.executorch_argument_index = -1;
    entry.external_tensor_query_performed = true;
    entry.external_tensor_host_allocation_supported = true;
    entry.bindings.push_back(VgfBoundaryBindingRef{
        .segment_index = 0,
        .set_index = 0,
        .binding = static_cast<uint32_t>(i),
    });
  }
  metadata[0].has_alias_group = true;
  metadata[3].has_alias_group = true;
  metadata[3].is_input = false;
  metadata[4].is_input = false;

  const auto mapping = vgf_resolve_external_io_mapping(
      /*serialized_inputs=*/{1, 0, -1, 2},
      /*serialized_outputs=*/{3, 4},
      /*mutable_inputs=*/{false, true, false, false},
      /*mutable_outputs=*/{true, false},
      metadata);

  EXPECT_EQ(mapping.inputs, (std::vector<int>{1, -1, 2}));
  EXPECT_EQ(mapping.outputs, (std::vector<int>{4}));
  EXPECT_EQ(metadata[1].executorch_argument_index, 0);
  EXPECT_EQ(metadata[2].executorch_argument_index, 2);
  EXPECT_EQ(metadata[4].executorch_argument_index, 3);
  for (size_t i : {0u, 3u}) {
    EXPECT_FALSE(metadata[i].mapped_to_model_boundary);
    EXPECT_EQ(metadata[i].executorch_argument_index, -1);
  }
  for (auto& entry : metadata) {
    vgf_finalize_zero_copy_io_metadata(&entry);
  }
  EXPECT_FALSE(metadata[0].eligible);
  EXPECT_FALSE(metadata[3].eligible);
  EXPECT_TRUE(metadata[1].eligible);
  EXPECT_TRUE(metadata[2].eligible);
  EXPECT_TRUE(metadata[4].eligible);
}

TEST(VgfZeroCopyMetadataTest, NonTensorDescriptorIsRejected) {
  VgfHostMemoryImportCapabilities host_capabilities{};
  host_capabilities.logical_device_enabled = true;
  auto metadata = make_vgf_zero_copy_io_metadata(
      /*is_model_boundary_resource=*/true,
      /*is_input=*/true,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_FORMAT_R32_SFLOAT,
      /*tensor_backed=*/false,
      /*has_alias_group=*/false,
      /*has_incompatible_image_aliasing=*/false,
      nullptr,
      host_capabilities);

  EXPECT_FALSE(vgf_zero_copy_io_structurally_eligible(metadata));
}

TEST(VgfZeroCopyMetadataTest, RecordsMultipleSegmentBindingReferences) {
  auto metadata = make_eligible_linear_tensor_metadata_for_test();
  std::vector<VgfZeroCopyIoMetadata> all_metadata{metadata};

  EXPECT_TRUE(vgf_record_boundary_binding(
      &all_metadata,
      0,
      VgfBoundaryBindingRef{.segment_index = 0, .set_index = 0, .binding = 1}));
  EXPECT_TRUE(vgf_record_boundary_binding(
      &all_metadata,
      0,
      VgfBoundaryBindingRef{.segment_index = 2, .set_index = 0, .binding = 7}));

  ASSERT_EQ(all_metadata[0].bindings.size(), 2u);
  EXPECT_EQ(
      all_metadata[0].bindings[0],
      (VgfBoundaryBindingRef{
          .segment_index = 0, .set_index = 0, .binding = 1}));
  EXPECT_EQ(
      all_metadata[0].bindings[1],
      (VgfBoundaryBindingRef{
          .segment_index = 2, .set_index = 0, .binding = 7}));
  EXPECT_TRUE(all_metadata[0].descriptor_bindings_unambiguous);
}

TEST(VgfZeroCopyMetadataTest, UnsupportedExternalTensorCapabilityIsIneligible) {
  auto metadata = make_eligible_linear_tensor_metadata_for_test();
  VkExternalMemoryProperties properties{};
  properties.externalMemoryFeatures = 0;
  properties.compatibleHandleTypes = 0;
  vgf_cache_external_tensor_properties(&metadata, properties);
  metadata.bindings.push_back(
      VgfBoundaryBindingRef{.segment_index = 0, .set_index = 0, .binding = 0});
  vgf_finalize_zero_copy_io_metadata(&metadata);

  EXPECT_TRUE(metadata.external_tensor_query_performed);
  EXPECT_FALSE(metadata.external_tensor_host_allocation_supported);
  EXPECT_FALSE(metadata.eligible);
}

TEST(VgfZeroCopyMetadataTest, DedicatedOnlyExternalTensorIsIneligible) {
  auto metadata = make_eligible_linear_tensor_metadata_for_test();
#if defined(VK_EXT_external_memory_host)
  VkExternalMemoryProperties properties{};
  properties.externalMemoryFeatures =
      VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT |
      VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT;
  properties.compatibleHandleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
  vgf_cache_external_tensor_properties(&metadata, properties);
#else
  metadata.external_tensor_query_performed = true;
  metadata.external_tensor_host_allocation_supported = true;
  metadata.external_tensor_requires_dedicated_allocation = true;
#endif
  metadata.bindings.push_back(
      VgfBoundaryBindingRef{.segment_index = 0, .set_index = 0, .binding = 0});
  vgf_finalize_zero_copy_io_metadata(&metadata);

  EXPECT_TRUE(metadata.external_tensor_requires_dedicated_allocation);
  EXPECT_FALSE(metadata.eligible);
}

TEST(VgfZeroCopyMetadataTest, AmbiguousDescriptorLocationIsIneligible) {
  auto first = make_eligible_linear_tensor_metadata_for_test();
  auto second = make_eligible_linear_tensor_metadata_for_test();
  std::vector<VgfZeroCopyIoMetadata> all_metadata{first, second};
  const VgfBoundaryBindingRef ref{
      .segment_index = 1, .set_index = 0, .binding = 5};

  EXPECT_TRUE(vgf_record_boundary_binding(&all_metadata, 0, ref));
  EXPECT_FALSE(vgf_record_boundary_binding(&all_metadata, 1, ref));
  EXPECT_FALSE(all_metadata[0].descriptor_bindings_unambiguous);
  EXPECT_FALSE(all_metadata[1].descriptor_bindings_unambiguous);
}

} // namespace
} // namespace vgf
} // namespace backends
} // namespace executorch
