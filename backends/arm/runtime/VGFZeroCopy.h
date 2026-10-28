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
// Runtime selection state for one eligible boundary tensor. Descriptor
// bindings stay in VgfZeroCopyIoMetadata because one IO can be referenced by
// multiple segment-owned descriptor sets.
struct VgfBoundaryBindingState {
  VkTensorViewARM owned_view = VK_NULL_HANDLE;
  VkTensorViewARM selected_view = VK_NULL_HANDLE;
  VkTensorViewARM bound_view = VK_NULL_HANDLE;
  bool import_active = false;
  bool dirty = false;

  void initialize(VkTensorViewARM view) {
    owned_view = view;
    selected_view = view;
    bound_view = view;
    import_active = false;
    dirty = false;
  }

  bool select(VkTensorViewARM view, bool imported) {
    import_active = imported;
    if (selected_view == view) {
      return false;
    }
    selected_view = view;
    dirty = selected_view != bound_view;
    return true;
  }

  void mark_clean() {
    bound_view = selected_view;
    dirty = false;
  }
};

struct VgfBoundaryDescriptorUpdate {
  size_t io_index = 0;
  VgfBoundaryBindingRef binding{};
  VkTensorViewARM tensor_view = VK_NULL_HANDLE;
};

// Apply the binding-selection policy after a host-import attempt.
//
// A successful import is immediately selected, including a cache hit. This is
// important when the owned view was selected by an earlier fallback while the
// imported slot remained cached. A failed import restores the owned view so the
// next dispatch cannot accidentally use an imported tensor selected by a prior
// attempt. Descriptor writes remain deferred until flush_boundary_bindings().
inline bool vgf_apply_import_binding_selection(
    VgfBoundaryBindingState* state,
    bool import_accepted,
    VkTensorViewARM imported_view) {
  if (state == nullptr || state->owned_view == VK_NULL_HANDLE) {
    return false;
  }

  if (!import_accepted) {
    state->select(state->owned_view, false);
    return true;
  }

  if (imported_view == VK_NULL_HANDLE) {
    return false;
  }

  state->select(imported_view, true);
  return true;
}

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

inline std::vector<VgfBoundaryDescriptorUpdate>
vgf_collect_boundary_descriptor_updates(
    const std::vector<VgfZeroCopyIoMetadata>& metadata,
    const std::vector<VgfBoundaryBindingState>& states) {
  std::vector<VgfBoundaryDescriptorUpdate> updates;
  const size_t count = std::min(metadata.size(), states.size());
  for (size_t io_index = 0; io_index < count; ++io_index) {
    const auto& state = states[io_index];
    const auto& io_metadata = metadata[io_index];
    if (!state.dirty || !io_metadata.eligible) {
      continue;
    }
    updates.reserve(updates.size() + io_metadata.bindings.size());
    std::transform(
        io_metadata.bindings.begin(),
        io_metadata.bindings.end(),
        std::back_inserter(updates),
        [&](const VgfBoundaryBindingRef& binding) {
          return VgfBoundaryDescriptorUpdate{
              .io_index = io_index,
              .binding = binding,
              .tensor_view = state.selected_view,
          };
        });
  }
  return updates;
}

inline bool vgf_boundary_bindings_dirty(
    const std::vector<VgfBoundaryBindingState>& states) {
  return std::any_of(states.begin(), states.end(), [](const auto& state) {
    return state.dirty;
  });
}

struct VgfExternalIoMapping {
  std::vector<int> inputs;
  std::vector<int> outputs;
};

inline VgfExternalIoMapping vgf_resolve_external_io_mapping(
    const std::vector<int>& serialized_inputs,
    const std::vector<int>& serialized_outputs,
    const std::vector<bool>& mutable_inputs,
    const std::vector<bool>& mutable_outputs,
    std::vector<VgfZeroCopyIoMetadata>& metadata) {
  VgfExternalIoMapping mapping;
  for (size_t i = 0; i < serialized_inputs.size(); ++i) {
    if (!mutable_inputs[i]) {
      mapping.inputs.push_back(serialized_inputs[i]);
    }
  }
  for (size_t i = 0; i < serialized_outputs.size(); ++i) {
    if (!mutable_outputs[i]) {
      mapping.outputs.push_back(serialized_outputs[i]);
    }
  }

  for (size_t i = 0; i < mapping.inputs.size(); ++i) {
    const int io_index = mapping.inputs[i];
    if (io_index >= 0 && static_cast<size_t>(io_index) < metadata.size()) {
      metadata[io_index].mapped_to_model_boundary = true;
      metadata[io_index].executorch_argument_index = static_cast<int64_t>(i);
    }
  }
  for (size_t i = 0; i < mapping.outputs.size(); ++i) {
    const int io_index = mapping.outputs[i];
    if (io_index >= 0 && static_cast<size_t>(io_index) < metadata.size()) {
      metadata[io_index].mapped_to_model_boundary = true;
      metadata[io_index].executorch_argument_index =
          static_cast<int64_t>(mapping.inputs.size() + i);
    }
  }
  return mapping;
}

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

// Reasons are stable indices for optional EXECUTORCH_VGF_IO_STATS counters.
// Keep kCount last and append new reasons before it.
enum class VgfZeroCopyImportRejectionReason : uint32_t {
  kNone = 0,
  kInvalidIoIndex,
  kFeatureDisabled,
  kMetadataIneligible,
  kExternalTensorUnsupported,
  kDedicatedAllocationUnsupported,
  kNullHostPointer,
  kInvalidHostRange,
  kPointerAlignment,
  kLogicalSizeAlignment,
  kVulkanFunctionUnavailable,
  kHostPointerPropertiesQueryFailed,
  kNoCompatibleCoherentMemoryType,
  kMemoryRequirementsTooLarge,
  kTensorCreateFailed,
  kMemoryAllocateFailed,
  kTensorBindFailed,
  kTensorViewCreateFailed,
  kDuplicateActiveRange,
  kOverlappingActiveRange,
  kCount,
};

// cppcheck-suppress unusedFunction
const char* vgf_zero_copy_import_rejection_reason_string(
    VgfZeroCopyImportRejectionReason reason);

// One application-owned host buffer imported for one VGF boundary IO.
// The host allocation is never owned or freed by the delegate.
struct VgfZeroCopySlot {
  void* host_pointer = nullptr;
  size_t logical_size = 0;
  size_t imported_size = 0;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkTensorARM tensor = VK_NULL_HANDLE;
  VkTensorViewARM tensor_view = VK_NULL_HANDLE;
  uint32_t memory_type_index = UINT32_MAX;

  bool valid() const {
    return host_pointer != nullptr && imported_size != 0 &&
        memory != VK_NULL_HANDLE && tensor != VK_NULL_HANDLE &&
        tensor_view != VK_NULL_HANDLE;
  }
};

struct VgfZeroCopyImportResult {
  bool accepted = false;
  bool cache_hit = false;
  VgfZeroCopyImportRejectionReason reason =
      VgfZeroCopyImportRejectionReason::kNone;
  VkResult vk_result = VK_SUCCESS;
};

// Vulkan call table for host-memory import operations.
// Production uses the loaded Vulkan entry points; unit tests can
// inject mock functions without requiring a Vulkan device.
//
// The guard checks Vulkan header support for VK_EXT_external_memory_host.
// It is not a test-only flag or a runtime extension availability check.
#if defined(VK_EXT_external_memory_host)
struct VgfZeroCopyVulkanApi {
  void* context = nullptr;

  VkResult (*get_memory_host_pointer_properties)(
      void* context,
      VkDevice device,
      VkExternalMemoryHandleTypeFlagBits handle_type,
      const void* host_pointer,
      VkMemoryHostPointerPropertiesEXT* properties) = nullptr;
  void (*get_physical_device_memory_properties)(
      void* context,
      VkPhysicalDevice physical_device,
      VkPhysicalDeviceMemoryProperties* properties) = nullptr;
  VkResult (*create_tensor)(
      void* context,
      VkDevice device,
      const VkTensorCreateInfoARM* create_info,
      const VkAllocationCallbacks* allocator,
      VkTensorARM* tensor) = nullptr;
  void (*get_tensor_memory_requirements)(
      void* context,
      VkDevice device,
      const VkTensorMemoryRequirementsInfoARM* info,
      VkMemoryRequirements2* requirements) = nullptr;
  VkResult (*allocate_memory)(
      void* context,
      VkDevice device,
      const VkMemoryAllocateInfo* allocate_info,
      const VkAllocationCallbacks* allocator,
      VkDeviceMemory* memory) = nullptr;
  VkResult (*bind_tensor_memory)(
      void* context,
      VkDevice device,
      uint32_t bind_info_count,
      const VkBindTensorMemoryInfoARM* bind_infos) = nullptr;
  VkResult (*create_tensor_view)(
      void* context,
      VkDevice device,
      const VkTensorViewCreateInfoARM* create_info,
      const VkAllocationCallbacks* allocator,
      VkTensorViewARM* tensor_view) = nullptr;
  void (*destroy_tensor_view)(
      void* context,
      VkDevice device,
      VkTensorViewARM tensor_view,
      const VkAllocationCallbacks* allocator) = nullptr;
  void (*destroy_tensor)(
      void* context,
      VkDevice device,
      VkTensorARM tensor,
      const VkAllocationCallbacks* allocator) = nullptr;
  void (*free_memory)(
      void* context,
      VkDevice device,
      VkDeviceMemory memory,
      const VkAllocationCallbacks* allocator) = nullptr;
};
#else
struct VgfZeroCopyVulkanApi {
  void* context = nullptr;
};
#endif

// Per-method resource manager. Slots and active host ranges live as long as
// the VgfRepr that owns this manager, not as process-global state.
class VgfZeroCopyImportManager final {
 public:
  VgfZeroCopyImportManager() = default;
  ~VgfZeroCopyImportManager();

  VgfZeroCopyImportManager(const VgfZeroCopyImportManager&) = delete;
  VgfZeroCopyImportManager& operator=(const VgfZeroCopyImportManager&) = delete;

  void reset(
      VkPhysicalDevice physical_device,
      VkDevice device,
      const VgfHostMemoryImportCapabilities& capabilities,
      size_t io_count,
      const VgfZeroCopyVulkanApi* api = nullptr);

  VgfZeroCopyImportResult try_import_candidate(
      size_t io_index,
      void* host_pointer,
      size_t logical_nbytes,
      const VgfZeroCopyIoMetadata& metadata);

  void release(size_t io_index);
  void release_all();

  const VgfZeroCopySlot* slot(size_t io_index) const;
  size_t slot_count() const {
    return slots_.size();
  }

 private:
  bool api_ready() const;
  bool find_coherent_memory_type(
      uint32_t compatible_memory_type_bits,
      uint32_t* memory_type_index) const;
  void destroy_slot(VgfZeroCopySlot* slot) noexcept;
  VgfZeroCopyImportResult reject(
      VgfZeroCopyImportRejectionReason reason,
      bool import_failure,
      VkResult vk_result = VK_SUCCESS) const;

  VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VgfHostMemoryImportCapabilities capabilities_{};
  VgfZeroCopyVulkanApi api_{};
  std::vector<VgfZeroCopySlot> slots_;
};

} // namespace vgf
} // namespace backends
} // namespace executorch
