/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <executorch/backends/arm/runtime/VGFVulkanFeatures.h>
#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

namespace executorch {
namespace backends {
namespace vgf {

// Identifies one descriptor binding that refers to a model-boundary resource.
// An IO can appear in multiple segments and bindings, so this is deliberately
// a many-to-one relation with VgfZeroCopyIoMetadata.
struct VgfBoundaryBindingRef {
  uint32_t segment_index = 0;
  uint32_t set_index = 0;
  uint32_t binding = 0;

  bool operator==(const VgfBoundaryBindingRef& other) const {
    return segment_index == other.segment_index &&
        set_index == other.set_index && binding == other.binding;
  }
};

// Zero-copy eligibility information for one model input or output.
// This metadata does not change resource allocation or execution.
struct VgfZeroCopyIoMetadata {
  bool eligible = false;

  // Identifies the corresponding ExecuTorch input or output.
  // IO vector order does not necessarily match ExecuTorch argument order.
  bool is_model_boundary_resource = false;
  bool mapped_to_model_boundary = false;
  bool is_input = false;
  int64_t executorch_argument_index = -1;

  // Resource/alias state.
  VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
  bool tensor_backed = false;
  bool has_alias_group = false;
  bool has_incompatible_image_aliasing = false;
  bool descriptor_bindings_unambiguous = true;

  // Tensor properties required to evaluate and later implement zero-copy.
  // Dimension and stride storage is copied because the VGF decoder is
  // temporary.
  VkTensorCreateFlagsARM tensor_create_flags = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkTensorTilingARM tiling = VK_TENSOR_TILING_LINEAR_ARM;
  VkTensorUsageFlagsARM usage = 0;
  std::vector<int64_t> dimensions;
  bool has_explicit_strides = false;
  std::vector<int64_t> strides;

  // VK_EXT_external_memory_host state from the actual VGF device plus the
  // exact-tensor external-memory capability query.
  bool host_memory_import_advertised = false;
  bool host_memory_import_enabled = false;
  bool external_tensor_query_performed = false;
  bool external_tensor_host_allocation_supported = false;
  bool external_tensor_requires_dedicated_allocation = false;
  VkExternalMemoryFeatureFlags external_memory_features = 0;
  VkExternalMemoryHandleTypeFlags compatible_handle_types = 0;

  std::vector<VgfBoundaryBindingRef> bindings;
};

// We keep this as the single source of truth for VGF tensor usage.
inline VkTensorUsageFlagsARM vgf_tensor_usage_flags(bool image_aliasing) {
  VkTensorUsageFlagsARM usage = VK_TENSOR_USAGE_SHADER_BIT_ARM |
      VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM |
      VK_TENSOR_USAGE_TRANSFER_DST_BIT_ARM | VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
  if (image_aliasing) {
    usage |= VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_ARM;
  }
  return usage;
}

inline VgfZeroCopyIoMetadata make_vgf_zero_copy_io_metadata(
    bool is_model_boundary_resource,
    bool is_input,
    VkDescriptorType descriptor_type,
    VkFormat resource_format,
    bool tensor_backed,
    bool has_alias_group,
    bool has_incompatible_image_aliasing,
    const VkTensorDescriptionARM* tensor_description,
    const VgfHostMemoryImportCapabilities& host_memory_capabilities) {
  VgfZeroCopyIoMetadata metadata{};
  metadata.is_model_boundary_resource = is_model_boundary_resource;
  metadata.is_input = is_input;
  metadata.descriptor_type = descriptor_type;
  metadata.tensor_backed = tensor_backed;
  metadata.has_alias_group = has_alias_group;
  metadata.has_incompatible_image_aliasing = has_incompatible_image_aliasing;
  metadata.format = resource_format;
  metadata.host_memory_import_advertised =
      host_memory_capabilities.physical_device_advertised;
  metadata.host_memory_import_enabled =
      host_memory_capabilities.logical_device_enabled;

  if (tensor_description != nullptr) {
    metadata.format = tensor_description->format;
    metadata.tiling = tensor_description->tiling;
    metadata.usage = tensor_description->usage;

    if (tensor_description->dimensionCount != 0 &&
        tensor_description->pDimensions != nullptr) {
      metadata.dimensions.assign(
          tensor_description->pDimensions,
          tensor_description->pDimensions + tensor_description->dimensionCount);
    }

    metadata.has_explicit_strides = tensor_description->pStrides != nullptr;
    if (metadata.has_explicit_strides &&
        tensor_description->dimensionCount != 0) {
      metadata.strides.assign(
          tensor_description->pStrides,
          tensor_description->pStrides + tensor_description->dimensionCount);
    }
  }

  return metadata;
}

// Check tensor structure only. Boundary mapping, descriptor bindings, and
// device import capabilities are checked during finalization.
inline bool vgf_zero_copy_io_structurally_eligible(
    const VgfZeroCopyIoMetadata& metadata) {
  return metadata.is_model_boundary_resource &&
      metadata.descriptor_type == VK_DESCRIPTOR_TYPE_TENSOR_ARM &&
      metadata.tensor_backed &&
      metadata.tiling == VK_TENSOR_TILING_LINEAR_ARM &&
      !metadata.has_alias_group && !metadata.has_incompatible_image_aliasing;
}

inline void vgf_cache_external_tensor_properties(
    VgfZeroCopyIoMetadata* metadata,
    const VkExternalMemoryProperties& properties) {
  if (metadata == nullptr) {
    return;
  }

  metadata->external_tensor_query_performed = true;
  metadata->external_memory_features = properties.externalMemoryFeatures;
  metadata->compatible_handle_types = properties.compatibleHandleTypes;
  metadata->external_tensor_requires_dedicated_allocation =
      (properties.externalMemoryFeatures &
       VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0;

#if defined(VK_EXT_external_memory_host)
  metadata->external_tensor_host_allocation_supported =
      (properties.externalMemoryFeatures &
       VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0 &&
      (properties.compatibleHandleTypes &
       VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT) != 0;
#else
  metadata->external_tensor_host_allocation_supported = false;
#endif
}

inline bool vgf_zero_copy_io_is_eligible(
    const VgfZeroCopyIoMetadata& metadata) {
  return vgf_zero_copy_io_structurally_eligible(metadata) &&
      metadata.mapped_to_model_boundary &&
      metadata.executorch_argument_index >= 0 &&
      metadata.descriptor_bindings_unambiguous && !metadata.bindings.empty() &&
      metadata.host_memory_import_enabled &&
      metadata.external_tensor_query_performed &&
      metadata.external_tensor_host_allocation_supported &&
      !metadata.external_tensor_requires_dedicated_allocation;
}

inline void vgf_finalize_zero_copy_io_metadata(
    VgfZeroCopyIoMetadata* metadata) {
  if (metadata != nullptr) {
    metadata->eligible = vgf_zero_copy_io_is_eligible(*metadata);
  }
}

// Record an exact segment/set/binding reference. Reusing the same descriptor
// location is treated as ambiguous rather than guessed; all involved boundary
// IO metadata is marked ineligible, while VGF initialization itself continues.
inline bool vgf_record_boundary_binding(
    std::vector<VgfZeroCopyIoMetadata>* all_metadata,
    size_t io_index,
    const VgfBoundaryBindingRef& binding_ref) {
  if (all_metadata == nullptr || io_index >= all_metadata->size()) {
    return false;
  }

  for (size_t other_io_index = 0; other_io_index < all_metadata->size();
       ++other_io_index) {
    const auto& existing_bindings = (*all_metadata)[other_io_index].bindings;
    const bool binding_already_used = std::any_of(
        existing_bindings.begin(),
        existing_bindings.end(),
        [&binding_ref](const VgfBoundaryBindingRef& existing) {
          return existing == binding_ref;
        });
    if (binding_already_used) {
      (*all_metadata)[io_index].descriptor_bindings_unambiguous = false;
      (*all_metadata)[io_index].eligible = false;
      (*all_metadata)[other_io_index].descriptor_bindings_unambiguous = false;
      (*all_metadata)[other_io_index].eligible = false;
      return false;
    }
  }

  (*all_metadata)[io_index].bindings.push_back(binding_ref);
  return true;
}

} // namespace vgf
} // namespace backends
} // namespace executorch
