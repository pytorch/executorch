/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/arm/runtime/VGFZeroCopy.h>

#include <executorch/backends/arm/runtime/VGFExecutionStats.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <limits>
#include <utility>

namespace executorch {
namespace backends {
namespace vgf {
namespace {

#if defined(VK_EXT_external_memory_host)
VkResult default_get_memory_host_pointer_properties(
    void*,
    VkDevice device,
    VkExternalMemoryHandleTypeFlagBits handle_type,
    const void* host_pointer,
    VkMemoryHostPointerPropertiesEXT* properties) {
  return vkGetMemoryHostPointerPropertiesEXT(
      device, handle_type, host_pointer, properties);
}

void default_get_physical_device_memory_properties(
    void*,
    VkPhysicalDevice physical_device,
    VkPhysicalDeviceMemoryProperties* properties) {
  vkGetPhysicalDeviceMemoryProperties(physical_device, properties);
}

VkResult default_create_tensor(
    void*,
    VkDevice device,
    const VkTensorCreateInfoARM* create_info,
    const VkAllocationCallbacks* allocator,
    VkTensorARM* tensor) {
  return vkCreateTensorARM(device, create_info, allocator, tensor);
}

void default_get_tensor_memory_requirements(
    void*,
    VkDevice device,
    const VkTensorMemoryRequirementsInfoARM* info,
    VkMemoryRequirements2* requirements) {
  vkGetTensorMemoryRequirementsARM(device, info, requirements);
}

VkResult default_allocate_memory(
    void*,
    VkDevice device,
    const VkMemoryAllocateInfo* allocate_info,
    const VkAllocationCallbacks* allocator,
    VkDeviceMemory* memory) {
  return vkAllocateMemory(device, allocate_info, allocator, memory);
}

VkResult default_bind_tensor_memory(
    void*,
    VkDevice device,
    uint32_t bind_info_count,
    const VkBindTensorMemoryInfoARM* bind_infos) {
  return vkBindTensorMemoryARM(device, bind_info_count, bind_infos);
}

VkResult default_create_tensor_view(
    void*,
    VkDevice device,
    const VkTensorViewCreateInfoARM* create_info,
    const VkAllocationCallbacks* allocator,
    VkTensorViewARM* tensor_view) {
  return vkCreateTensorViewARM(device, create_info, allocator, tensor_view);
}

void default_destroy_tensor_view(
    void*,
    VkDevice device,
    VkTensorViewARM tensor_view,
    const VkAllocationCallbacks* allocator) {
  vkDestroyTensorViewARM(device, tensor_view, allocator);
}

void default_destroy_tensor(
    void*,
    VkDevice device,
    VkTensorARM tensor,
    const VkAllocationCallbacks* allocator) {
  vkDestroyTensorARM(device, tensor, allocator);
}

void default_free_memory(
    void*,
    VkDevice device,
    VkDeviceMemory memory,
    const VkAllocationCallbacks* allocator) {
  vkFreeMemory(device, memory, allocator);
}

VgfZeroCopyVulkanApi make_default_api() {
  VgfZeroCopyVulkanApi api{};
  api.get_memory_host_pointer_properties =
      vkGetMemoryHostPointerPropertiesEXT != nullptr
      ? default_get_memory_host_pointer_properties
      : nullptr;
  api.get_physical_device_memory_properties =
      vkGetPhysicalDeviceMemoryProperties != nullptr
      ? default_get_physical_device_memory_properties
      : nullptr;
  api.create_tensor =
      vkCreateTensorARM != nullptr ? default_create_tensor : nullptr;
  api.get_tensor_memory_requirements =
      vkGetTensorMemoryRequirementsARM != nullptr
      ? default_get_tensor_memory_requirements
      : nullptr;
  api.allocate_memory =
      vkAllocateMemory != nullptr ? default_allocate_memory : nullptr;
  api.bind_tensor_memory =
      vkBindTensorMemoryARM != nullptr ? default_bind_tensor_memory : nullptr;
  api.create_tensor_view =
      vkCreateTensorViewARM != nullptr ? default_create_tensor_view : nullptr;
  api.destroy_tensor_view =
      vkDestroyTensorViewARM != nullptr ? default_destroy_tensor_view : nullptr;
  api.destroy_tensor =
      vkDestroyTensorARM != nullptr ? default_destroy_tensor : nullptr;
  api.free_memory = vkFreeMemory != nullptr ? default_free_memory : nullptr;
  return api;
}
#endif

bool ranges_overlap(
    uintptr_t first_begin,
    uintptr_t first_end,
    uintptr_t second_begin,
    uintptr_t second_end) {
  return first_begin < second_end && second_begin < first_end;
}

} // namespace

// cppcheck-suppress unusedFunction
const char* vgf_zero_copy_import_rejection_reason_string(
    VgfZeroCopyImportRejectionReason reason) {
  switch (reason) {
    case VgfZeroCopyImportRejectionReason::kNone:
      return "none";
    case VgfZeroCopyImportRejectionReason::kInvalidIoIndex:
      return "invalid_io_index";
    case VgfZeroCopyImportRejectionReason::kFeatureDisabled:
      return "feature_disabled";
    case VgfZeroCopyImportRejectionReason::kMetadataIneligible:
      return "metadata_ineligible";
    case VgfZeroCopyImportRejectionReason::kExternalTensorUnsupported:
      return "external_tensor_unsupported";
    case VgfZeroCopyImportRejectionReason::kDedicatedAllocationUnsupported:
      return "dedicated_allocation_unsupported";
    case VgfZeroCopyImportRejectionReason::kNullHostPointer:
      return "null_host_pointer";
    case VgfZeroCopyImportRejectionReason::kInvalidHostRange:
      return "invalid_host_range";
    case VgfZeroCopyImportRejectionReason::kPointerAlignment:
      return "pointer_alignment";
    case VgfZeroCopyImportRejectionReason::kLogicalSizeAlignment:
      return "logical_size_alignment";
    case VgfZeroCopyImportRejectionReason::kVulkanFunctionUnavailable:
      return "vulkan_function_unavailable";
    case VgfZeroCopyImportRejectionReason::kHostPointerPropertiesQueryFailed:
      return "host_pointer_properties_query_failed";
    case VgfZeroCopyImportRejectionReason::kNoCompatibleCoherentMemoryType:
      return "no_compatible_coherent_memory_type";
    case VgfZeroCopyImportRejectionReason::kMemoryRequirementsTooLarge:
      return "memory_requirements_too_large";
    case VgfZeroCopyImportRejectionReason::kTensorCreateFailed:
      return "tensor_create_failed";
    case VgfZeroCopyImportRejectionReason::kMemoryAllocateFailed:
      return "memory_allocate_failed";
    case VgfZeroCopyImportRejectionReason::kTensorBindFailed:
      return "tensor_bind_failed";
    case VgfZeroCopyImportRejectionReason::kTensorViewCreateFailed:
      return "tensor_view_create_failed";
    case VgfZeroCopyImportRejectionReason::kDuplicateActiveRange:
      return "duplicate_active_range";
    case VgfZeroCopyImportRejectionReason::kOverlappingActiveRange:
      return "overlapping_active_range";
    case VgfZeroCopyImportRejectionReason::kCount:
      return "count";
  }
  return "unknown";
}

VgfZeroCopyImportManager::~VgfZeroCopyImportManager() {
  release_all();
}

void VgfZeroCopyImportManager::reset(
    VkPhysicalDevice physical_device,
    VkDevice device,
    const VgfHostMemoryImportCapabilities& capabilities,
    size_t io_count,
    const VgfZeroCopyVulkanApi* api) {
  // Release with the old device/API before replacing the configuration.
  release_all();
  physical_device_ = physical_device;
  device_ = device;
  capabilities_ = capabilities;
#if defined(VK_EXT_external_memory_host)
  api_ = api != nullptr ? *api : make_default_api();
#else
  (void)api;
  api_ = {};
#endif
  slots_.assign(io_count, VgfZeroCopySlot{});
}

const VgfZeroCopySlot* VgfZeroCopyImportManager::slot(size_t io_index) const {
  return io_index < slots_.size() ? &slots_[io_index] : nullptr;
}

bool VgfZeroCopyImportManager::api_ready() const {
#if defined(VK_EXT_external_memory_host)
  return physical_device_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE &&
      api_.get_memory_host_pointer_properties != nullptr &&
      api_.get_physical_device_memory_properties != nullptr &&
      api_.create_tensor != nullptr &&
      api_.get_tensor_memory_requirements != nullptr &&
      api_.allocate_memory != nullptr && api_.bind_tensor_memory != nullptr &&
      api_.create_tensor_view != nullptr &&
      api_.destroy_tensor_view != nullptr && api_.destroy_tensor != nullptr &&
      api_.free_memory != nullptr;
#else
  return false;
#endif
}

bool VgfZeroCopyImportManager::find_coherent_memory_type(
    uint32_t compatible_memory_type_bits,
    uint32_t* memory_type_index) const {
#if defined(VK_EXT_external_memory_host)
  if (memory_type_index == nullptr || compatible_memory_type_bits == 0 ||
      api_.get_physical_device_memory_properties == nullptr) {
    return false;
  }

  VkPhysicalDeviceMemoryProperties memory_properties{};
  api_.get_physical_device_memory_properties(
      api_.context, physical_device_, &memory_properties);

  const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount && i < 32; ++i) {
    if ((compatible_memory_type_bits & (1u << i)) == 0) {
      continue;
    }
    const VkMemoryPropertyFlags flags =
        memory_properties.memoryTypes[i].propertyFlags;
    if ((flags & required) == required) {
      *memory_type_index = i;
      return true;
    }
  }
#else
  (void)compatible_memory_type_bits;
  (void)memory_type_index;
#endif
  return false;
}

void VgfZeroCopyImportManager::destroy_slot(VgfZeroCopySlot* slot) noexcept {
  if (slot == nullptr) {
    return;
  }
#if defined(VK_EXT_external_memory_host)
  // Vulkan-valid destruction order: view -> tensor -> bound memory.  The host
  // allocation behind pHostPointer is application owned and is never freed.
  if (slot->tensor_view != VK_NULL_HANDLE &&
      api_.destroy_tensor_view != nullptr) {
    api_.destroy_tensor_view(api_.context, device_, slot->tensor_view, nullptr);
  }
  if (slot->tensor != VK_NULL_HANDLE && api_.destroy_tensor != nullptr) {
    api_.destroy_tensor(api_.context, device_, slot->tensor, nullptr);
  }
  if (slot->memory != VK_NULL_HANDLE && api_.free_memory != nullptr) {
    api_.free_memory(api_.context, device_, slot->memory, nullptr);
  }
#endif
  *slot = {};
}

void VgfZeroCopyImportManager::release(size_t io_index) {
  if (io_index < slots_.size()) {
    destroy_slot(&slots_[io_index]);
  }
}

void VgfZeroCopyImportManager::release_all() {
  for (auto& slot : slots_) {
    destroy_slot(&slot);
  }
}

VgfZeroCopyImportResult VgfZeroCopyImportManager::reject(
    VgfZeroCopyImportRejectionReason reason,
    bool import_failure,
    VkResult vk_result) const {
  const size_t reason_index = static_cast<size_t>(reason);
  if (import_failure) {
    VGF_STATS_IMPORT_FAILURE(reason_index);
  } else {
    VGF_STATS_IMPORT_REJECTION(reason_index);
  }
  return VgfZeroCopyImportResult{
      false,
      false,
      reason,
      vk_result,
  };
}

VgfZeroCopyImportResult VgfZeroCopyImportManager::try_import_candidate(
    size_t io_index,
    void* host_pointer,
    size_t logical_nbytes,
    const VgfZeroCopyIoMetadata& metadata) {
  if (io_index >= slots_.size()) {
    return reject(VgfZeroCopyImportRejectionReason::kInvalidIoIndex, false);
  }

  if (!capabilities_.physical_device_advertised ||
      !capabilities_.logical_device_enabled ||
      capabilities_.min_imported_host_pointer_alignment == 0) {
    return reject(VgfZeroCopyImportRejectionReason::kFeatureDisabled, false);
  }

  if (!metadata.eligible) {
    if (!metadata.external_tensor_query_performed ||
        !metadata.external_tensor_host_allocation_supported) {
      return reject(
          VgfZeroCopyImportRejectionReason::kExternalTensorUnsupported, false);
    }
    if (metadata.external_tensor_requires_dedicated_allocation) {
      return reject(
          VgfZeroCopyImportRejectionReason::kDedicatedAllocationUnsupported,
          false);
    }
    return reject(VgfZeroCopyImportRejectionReason::kMetadataIneligible, false);
  }

  if (metadata.external_tensor_requires_dedicated_allocation) {
    return reject(
        VgfZeroCopyImportRejectionReason::kDedicatedAllocationUnsupported,
        false);
  }

  if (host_pointer == nullptr) {
    return reject(VgfZeroCopyImportRejectionReason::kNullHostPointer, false);
  }
  if (logical_nbytes == 0 ||
      logical_nbytes > std::numeric_limits<VkDeviceSize>::max()) {
    return reject(VgfZeroCopyImportRejectionReason::kInvalidHostRange, false);
  }

  const VkDeviceSize min_alignment =
      capabilities_.min_imported_host_pointer_alignment;
  const uintptr_t range_begin = reinterpret_cast<uintptr_t>(host_pointer);
  if (static_cast<VkDeviceSize>(range_begin) % min_alignment != 0) {
    return reject(VgfZeroCopyImportRejectionReason::kPointerAlignment, false);
  }

  const VkDeviceSize logical_size = static_cast<VkDeviceSize>(logical_nbytes);
  // Vulkan requires the imported allocationSize to satisfy the host-import
  // alignment requirement. ExecuTorch currently gives us tensor.nbytes(), not
  // guaranteed backing capacity, so we must not round allocationSize beyond
  // the only host range we know is valid. Requiring nbytes itself to be aligned
  // lets allocationSize == nbytes remain both Vulkan-valid and in-bounds.
  if (logical_size % min_alignment != 0) {
    return reject(
        VgfZeroCopyImportRejectionReason::kLogicalSizeAlignment, false);
  }

  if (logical_nbytes > std::numeric_limits<uintptr_t>::max() - range_begin) {
    return reject(VgfZeroCopyImportRejectionReason::kInvalidHostRange, false);
  }
  const uintptr_t range_end = range_begin + logical_nbytes;

  const auto& current = slots_[io_index];
  if (current.valid() && current.host_pointer == host_pointer &&
      current.logical_size == logical_nbytes &&
      current.imported_size == logical_nbytes) {
    return VgfZeroCopyImportResult{
        true,
        true,
        VgfZeroCopyImportRejectionReason::kNone,
        VK_SUCCESS,
    };
  }

  // Include the current slot in overlap checking. The exact same range was
  // handled by the cache-hit path above; any other overlapping replacement
  // must fall back because the current import remains alive until commit.
  for (size_t other_io = 0; other_io < slots_.size(); ++other_io) {
    if (!slots_[other_io].valid()) {
      continue;
    }
    const auto& other = slots_[other_io];
    const uintptr_t other_begin =
        reinterpret_cast<uintptr_t>(other.host_pointer);
    if (other.imported_size >
        std::numeric_limits<uintptr_t>::max() - other_begin) {
      // An impossible/corrupt tracked range is safest treated as conflicting.
      return reject(
          VgfZeroCopyImportRejectionReason::kOverlappingActiveRange, false);
    }
    const uintptr_t other_end = other_begin + other.imported_size;
    if (range_begin == other_begin && range_end == other_end) {
      return reject(
          VgfZeroCopyImportRejectionReason::kDuplicateActiveRange, false);
    }
    if (ranges_overlap(range_begin, range_end, other_begin, other_end)) {
      return reject(
          VgfZeroCopyImportRejectionReason::kOverlappingActiveRange, false);
    }
  }

#if !defined(VK_EXT_external_memory_host)
  return reject(VgfZeroCopyImportRejectionReason::kFeatureDisabled, false);
#else
  if (!api_ready()) {
    return reject(
        VgfZeroCopyImportRejectionReason::kVulkanFunctionUnavailable, false);
  }

  if (metadata.dimensions.size() > std::numeric_limits<uint32_t>::max() ||
      (metadata.has_explicit_strides &&
       metadata.strides.size() != metadata.dimensions.size())) {
    return reject(VgfZeroCopyImportRejectionReason::kMetadataIneligible, false);
  }

  VgfZeroCopySlot candidate{};
  candidate.host_pointer = host_pointer;
  candidate.logical_size = logical_nbytes;
  candidate.imported_size = logical_nbytes;

  auto fail_candidate = [&](VgfZeroCopyImportRejectionReason reason,
                            bool import_failure,
                            VkResult vk_result = VK_SUCCESS) {
    destroy_slot(&candidate);
    return reject(reason, import_failure, vk_result);
  };

  {
    VGF_STATS_TIME(import_ns);

    VkMemoryHostPointerPropertiesEXT host_properties{
        .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
        .pNext = nullptr,
        .memoryTypeBits = 0,
    };
    VkResult result = api_.get_memory_host_pointer_properties(
        api_.context,
        device_,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        host_pointer,
        &host_properties);
    if (result != VK_SUCCESS) {
      return fail_candidate(
          VgfZeroCopyImportRejectionReason::kHostPointerPropertiesQueryFailed,
          true,
          result);
    }

    const VkTensorDescriptionARM description{
        .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
        .pNext = nullptr,
        .tiling = metadata.tiling,
        .format = metadata.format,
        .dimensionCount = static_cast<uint32_t>(metadata.dimensions.size()),
        .pDimensions =
            metadata.dimensions.empty() ? nullptr : metadata.dimensions.data(),
        .pStrides =
            metadata.has_explicit_strides ? metadata.strides.data() : nullptr,
        .usage = metadata.usage,
    };

    const VkExternalMemoryTensorCreateInfoARM external_tensor_info{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_TENSOR_CREATE_INFO_ARM,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
    };
    const VkTensorCreateInfoARM create_info{
        .sType = VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM,
        .pNext = &external_tensor_info,
        .flags = metadata.tensor_create_flags,
        .pDescription = &description,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    result = api_.create_tensor(
        api_.context, device_, &create_info, nullptr, &candidate.tensor);
    if (result != VK_SUCCESS) {
      return fail_candidate(
          VgfZeroCopyImportRejectionReason::kTensorCreateFailed, true, result);
    }

    VkMemoryDedicatedRequirements dedicated_requirements{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
        .pNext = nullptr,
    };
    VkMemoryRequirements2 memory_requirements{
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = &dedicated_requirements,
    };
    const VkTensorMemoryRequirementsInfoARM requirements_info{
        .sType = VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM,
        .pNext = nullptr,
        .tensor = candidate.tensor,
    };
    api_.get_tensor_memory_requirements(
        api_.context, device_, &requirements_info, &memory_requirements);

    if (dedicated_requirements.requiresDedicatedAllocation == VK_TRUE) {
      return fail_candidate(
          VgfZeroCopyImportRejectionReason::kDedicatedAllocationUnsupported,
          false);
    }

    // We know only tensor.nbytes() of backing capacity. Do not round the host
    // range up. The exact imported allocation is therefore logical_size.
    const VkDeviceSize allocation_size = logical_size;
    if (memory_requirements.memoryRequirements.size > allocation_size) {
      return fail_candidate(
          VgfZeroCopyImportRejectionReason::kMemoryRequirementsTooLarge, false);
    }

    const uint32_t compatible_bits =
        memory_requirements.memoryRequirements.memoryTypeBits &
        host_properties.memoryTypeBits;
    if (!find_coherent_memory_type(
            compatible_bits, &candidate.memory_type_index)) {
      return fail_candidate(
          VgfZeroCopyImportRejectionReason::kNoCompatibleCoherentMemoryType,
          false);
    }

    const VkImportMemoryHostPointerInfoEXT import_info{
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .pNext = nullptr,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        .pHostPointer = host_pointer,
    };
    const VkMemoryAllocateInfo allocate_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
        .allocationSize = allocation_size,
        .memoryTypeIndex = candidate.memory_type_index,
    };
    result = api_.allocate_memory(
        api_.context, device_, &allocate_info, nullptr, &candidate.memory);
    if (result != VK_SUCCESS) {
      return fail_candidate(
          VgfZeroCopyImportRejectionReason::kMemoryAllocateFailed,
          true,
          result);
    }

    const VkBindTensorMemoryInfoARM bind_info{
        .sType = VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM,
        .pNext = nullptr,
        .tensor = candidate.tensor,
        .memory = candidate.memory,
        .memoryOffset = 0,
    };
    result = api_.bind_tensor_memory(api_.context, device_, 1, &bind_info);
    if (result != VK_SUCCESS) {
      return fail_candidate(
          VgfZeroCopyImportRejectionReason::kTensorBindFailed, true, result);
    }

    const VkTensorViewCreateInfoARM tensor_view_info{
        .sType = VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM,
        .pNext = nullptr,
        .flags = 0,
        .tensor = candidate.tensor,
        .format = metadata.format,
    };
    result = api_.create_tensor_view(
        api_.context,
        device_,
        &tensor_view_info,
        nullptr,
        &candidate.tensor_view);
    if (result != VK_SUCCESS) {
      return fail_candidate(
          VgfZeroCopyImportRejectionReason::kTensorViewCreateFailed,
          true,
          result);
    }

    // Transactional commit. The previous working slot remains alive until the
    // replacement has been fully created, bound, and has a valid view.
    std::swap(slots_[io_index], candidate);
    destroy_slot(&candidate); // candidate now contains the previous slot.
  }

  VGF_STATS_IMPORT_SUCCESS();
  return VgfZeroCopyImportResult{
      true,
      false,
      VgfZeroCopyImportRejectionReason::kNone,
      VK_SUCCESS,
  };
#endif
}

} // namespace vgf
} // namespace backends
} // namespace executorch
