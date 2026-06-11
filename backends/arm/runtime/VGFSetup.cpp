/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * VGF functions which prepare a graph for execution by allocating the
 * appropriate vulkan structures.
 */

#include <executorch/backends/arm/runtime/VGFSetup.h>

#include <cstdlib>
#include <limits>

#ifdef ET_EVENT_TRACER_ENABLED
#include <executorch/runtime/core/event_tracer_hooks_delegate.h>
#endif

#include <vgf/decoder.hpp>
#if __has_include(<vgf/version.h>)
#include <vgf/version.h>
#endif
#include <vgf/vulkan_helpers.generated.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <type_traits>
#include <unordered_map>

using namespace mlsdk;

#if defined(MLSDK_VGF_LIBRARY_API_VERSION_MAJOR) && \
    defined(MLSDK_VGF_LIBRARY_API_VERSION_MINOR)
#define EXECUTORCH_ARM_VGF_HAS_DECODER_V10_APIS \
  ((MLSDK_VGF_LIBRARY_API_VERSION_MAJOR > 0) || \
   (MLSDK_VGF_LIBRARY_API_VERSION_MAJOR == 0 && \
    MLSDK_VGF_LIBRARY_API_VERSION_MINOR >= 10))
#else
#define EXECUTORCH_ARM_VGF_HAS_DECODER_V10_APIS 0
#endif

namespace executorch {
namespace backends {
namespace vgf {

/* static function to map format to byte count */
static uint32_t get_format_size(VkFormat format);

// SPV_ARM_tensor does not support rank-0 representations according to the spec.
// Use an unsqueezed dimension when the resource table contains an empty
// shape. Tensors are output as rank 0 when copied back from the vgf backend.
namespace {
constexpr int64_t kScalarSentinelDimension = 1;
static bool is_image_descriptor_type(VkDescriptorType descriptor_type);
static bool is_tensor_like_descriptor_type(VkDescriptorType descriptor_type);

enum class FormatScalarKind {
  Bool,
  Uint,
  Sint,
  Float,
};

struct FormatInfo {
  uint32_t component_count = 0;
  uint32_t bytes_per_component = 0;
  FormatScalarKind scalar_kind = FormatScalarKind::Uint;
};

struct AliasLogicalContract {
  bool initialized = false;
  vector<int64_t> shape;
  vector<int64_t> stride;
  size_t logical_byte_size = 0;
  uint32_t scalar_bytes = 0;
  FormatScalarKind scalar_kind = FormatScalarKind::Uint;
  bool image_initialized = false;
  uint32_t image_component_count = 0;
};

static size_t element_count_from_shape(const vector<int64_t>& shape) {
  if (shape.empty()) {
    return 1;
  }
  size_t count = 1;
  for (auto dim : shape) {
    if (dim <= 0) {
      return 0;
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}

#ifdef ET_EVENT_TRACER_ENABLED
class ScopedVgfProfileEvent {
 public:
  ScopedVgfProfileEvent(
      executorch::runtime::EventTracer* event_tracer,
      const char* name)
      : event_tracer_(event_tracer),
        entry_(executorch::runtime::event_tracer_start_profiling_delegate(
            event_tracer_,
            name,
            /*delegate_debug_id=*/-1)) {}

  ~ScopedVgfProfileEvent() {
    executorch::runtime::event_tracer_end_profiling_delegate(
        event_tracer_, entry_);
  }

 private:
  executorch::runtime::EventTracer* event_tracer_;
  executorch::runtime::EventTracerEntry entry_;
};
#endif

#define VGF_CONCAT_INNER(a, b) a##b
#define VGF_CONCAT(a, b) VGF_CONCAT_INNER(a, b)

#ifdef ET_EVENT_TRACER_ENABLED
#define VGF_PROFILE_SCOPE(event_tracer, name)                      \
  ScopedVgfProfileEvent VGF_CONCAT(_vgf_profile_scope_, __LINE__)( \
      event_tracer, name)
#else
#define VGF_PROFILE_SCOPE(event_tracer, name) (void)(event_tracer)
#endif

static vector<int64_t> normalize_stride(
    const vector<int64_t>& shape,
    const vector<int64_t>& stride) {
  if (!stride.empty()) {
    return stride;
  }

  vector<int64_t> contiguous_stride(shape.size(), 1);
  int64_t running = 1;
  for (size_t idx = shape.size(); idx > 0; --idx) {
    contiguous_stride[idx - 1] = running;
    running *= shape[idx - 1];
  }
  return contiguous_stride;
}

static uint32_t get_format_component_count(VkFormat format) {
  switch (format) {
    case VK_FORMAT_R8_BOOL_ARM:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R64_SINT:
      return 1;
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32_SFLOAT:
      return 2;
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return 4;
    default:
      ET_LOG(
          Error,
          "Unsupported image VkFormat %u for component count",
          static_cast<uint32_t>(format));
      return 0;
  }
}

static bool get_format_info(VkFormat format, FormatInfo* info) {
  switch (format) {
    case VK_FORMAT_R8_BOOL_ARM:
      *info = FormatInfo{1, 1, FormatScalarKind::Bool};
      return true;
    case VK_FORMAT_R8_UINT:
      *info = FormatInfo{1, 1, FormatScalarKind::Uint};
      return true;
    case VK_FORMAT_R8_SINT:
      *info = FormatInfo{1, 1, FormatScalarKind::Sint};
      return true;
    case VK_FORMAT_R16_UINT:
      *info = FormatInfo{1, 2, FormatScalarKind::Uint};
      return true;
    case VK_FORMAT_R16_SINT:
      *info = FormatInfo{1, 2, FormatScalarKind::Sint};
      return true;
    case VK_FORMAT_R16_SFLOAT:
      *info = FormatInfo{1, 2, FormatScalarKind::Float};
      return true;
    case VK_FORMAT_R32_UINT:
      *info = FormatInfo{1, 4, FormatScalarKind::Uint};
      return true;
    case VK_FORMAT_R32_SINT:
      *info = FormatInfo{1, 4, FormatScalarKind::Sint};
      return true;
    case VK_FORMAT_R32_SFLOAT:
      *info = FormatInfo{1, 4, FormatScalarKind::Float};
      return true;
    case VK_FORMAT_R64_SINT:
      *info = FormatInfo{1, 8, FormatScalarKind::Sint};
      return true;
    case VK_FORMAT_R8G8_UINT:
      *info = FormatInfo{2, 1, FormatScalarKind::Uint};
      return true;
    case VK_FORMAT_R8G8_SINT:
      *info = FormatInfo{2, 1, FormatScalarKind::Sint};
      return true;
    case VK_FORMAT_R16G16_UINT:
      *info = FormatInfo{2, 2, FormatScalarKind::Uint};
      return true;
    case VK_FORMAT_R16G16_SINT:
      *info = FormatInfo{2, 2, FormatScalarKind::Sint};
      return true;
    case VK_FORMAT_R16G16_SFLOAT:
      *info = FormatInfo{2, 2, FormatScalarKind::Float};
      return true;
    case VK_FORMAT_R32G32_UINT:
      *info = FormatInfo{2, 4, FormatScalarKind::Uint};
      return true;
    case VK_FORMAT_R32G32_SINT:
      *info = FormatInfo{2, 4, FormatScalarKind::Sint};
      return true;
    case VK_FORMAT_R32G32_SFLOAT:
      *info = FormatInfo{2, 4, FormatScalarKind::Float};
      return true;
    case VK_FORMAT_R8G8B8A8_UINT:
      *info = FormatInfo{4, 1, FormatScalarKind::Uint};
      return true;
    case VK_FORMAT_R8G8B8A8_SINT:
      *info = FormatInfo{4, 1, FormatScalarKind::Sint};
      return true;
    case VK_FORMAT_R16G16B16A16_UINT:
      *info = FormatInfo{4, 2, FormatScalarKind::Uint};
      return true;
    case VK_FORMAT_R16G16B16A16_SINT:
      *info = FormatInfo{4, 2, FormatScalarKind::Sint};
      return true;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      *info = FormatInfo{4, 2, FormatScalarKind::Float};
      return true;
    case VK_FORMAT_R32G32B32A32_UINT:
      *info = FormatInfo{4, 4, FormatScalarKind::Uint};
      return true;
    case VK_FORMAT_R32G32B32A32_SINT:
      *info = FormatInfo{4, 4, FormatScalarKind::Sint};
      return true;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      *info = FormatInfo{4, 4, FormatScalarKind::Float};
      return true;
    default:
      ET_LOG(Error, "Unsupported VkFormat %u", static_cast<uint32_t>(format));
      return false;
  }
}

static bool validate_image_shape_and_format(
    const vector<int64_t>& shape,
    VkFormat format,
    VkExtent3D* image_extent,
    size_t* staging_size = nullptr) {
  const uint32_t format_component_count = get_format_component_count(format);
  const size_t bytes_per_pixel = get_format_size(format);
  if (format_component_count == 0 || bytes_per_pixel == 0) {
    return false;
  }

  int64_t height = 0;
  int64_t width = 0;
  int64_t channels = 0;
  if (shape.size() == 4) {
    if (shape[0] != 1) {
      ET_LOG(Error, "Only batch size 1 images are currently supported");
      return false;
    }
    height = shape[1];
    width = shape[2];
    channels = shape[3];
  } else if (shape.size() == 3) {
    height = shape[0];
    width = shape[1];
    channels = shape[2];
  } else {
    ET_LOG(Error, "Unsupported image shape rank %zu", shape.size());
    return false;
  }

  if (height <= 0 || width <= 0 || channels <= 0) {
    ET_LOG(
        Error,
        "Image shape dimensions must be positive, got [%lld, %lld, %lld]",
        static_cast<long long>(height),
        static_cast<long long>(width),
        static_cast<long long>(channels));
    return false;
  }

  if (static_cast<uint32_t>(channels) != format_component_count) {
    ET_LOG(
        Error,
        "Image channel count %lld does not match VkFormat %u component count %u",
        static_cast<long long>(channels),
        static_cast<uint32_t>(format),
        format_component_count);
    return false;
  }

  image_extent->width = static_cast<uint32_t>(width);
  image_extent->height = static_cast<uint32_t>(height);
  image_extent->depth = 1;

  if (staging_size != nullptr) {
    const size_t pixel_count = static_cast<size_t>(image_extent->width) *
        static_cast<size_t>(image_extent->height) *
        static_cast<size_t>(image_extent->depth);
    if (pixel_count > std::numeric_limits<size_t>::max() / bytes_per_pixel) {
      ET_LOG(Error, "Image staging allocation size overflow");
      return false;
    }
    *staging_size = pixel_count * bytes_per_pixel;
  }

  return true;
}

static bool validate_alias_group_logical_contract(
    uint32_t alias_group_id,
    uint32_t resource_index,
    VkDescriptorType descriptor_type,
    VkFormat format,
    const vector<int64_t>& shape,
    const vector<int64_t>& stride,
    AliasLogicalContract* contract) {
  FormatInfo format_info;
  if (!get_format_info(format, &format_info)) {
    return false;
  }

  size_t logical_byte_size = 0;
  if (is_image_descriptor_type(descriptor_type)) {
    VkExtent3D image_extent = {};
    if (!validate_image_shape_and_format(
            shape, format, &image_extent, &logical_byte_size)) {
      return false;
    }
  } else if (is_tensor_like_descriptor_type(descriptor_type)) {
    if (format_info.component_count != 1) {
      ET_LOG(
          Error,
          "Alias group %u tensor-like resource %u must use a scalar VkFormat",
          alias_group_id,
          resource_index);
      return false;
    }
    logical_byte_size =
        element_count_from_shape(shape) * get_format_size(format);
  } else {
    ET_LOG(
        Error,
        "Alias group %u contains unsupported descriptor type %u for resource %u",
        alias_group_id,
        static_cast<uint32_t>(descriptor_type),
        resource_index);
    return false;
  }

  const vector<int64_t> normalized_stride = normalize_stride(shape, stride);
  if (!contract->initialized) {
    contract->initialized = true;
    contract->shape = shape;
    contract->stride = normalized_stride;
    contract->logical_byte_size = logical_byte_size;
    contract->scalar_bytes = format_info.bytes_per_component;
    contract->scalar_kind = format_info.scalar_kind;
  } else {
    if (contract->shape != shape || contract->stride != normalized_stride) {
      ET_LOG(
          Error,
          "Alias group %u has mismatched logical layout at resource %u",
          alias_group_id,
          resource_index);
      return false;
    }
    if (contract->logical_byte_size != logical_byte_size) {
      ET_LOG(
          Error,
          "Alias group %u has mismatched logical byte size at resource %u",
          alias_group_id,
          resource_index);
      return false;
    }
    if (contract->scalar_bytes != format_info.bytes_per_component ||
        contract->scalar_kind != format_info.scalar_kind) {
      ET_LOG(
          Error,
          "Alias group %u has mismatched scalar format at resource %u",
          alias_group_id,
          resource_index);
      return false;
    }
  }

  if (is_image_descriptor_type(descriptor_type)) {
    if (!contract->image_initialized) {
      contract->image_initialized = true;
      contract->image_component_count = format_info.component_count;
    } else if (contract->image_component_count != format_info.component_count) {
      ET_LOG(
          Error,
          "Alias group %u has mismatched image channel packing at resource %u",
          alias_group_id,
          resource_index);
      return false;
    }
  }

  if (contract->image_initialized && !shape.empty() &&
      static_cast<uint32_t>(shape.back()) != contract->image_component_count) {
    ET_LOG(
        Error,
        "Alias group %u shape channel dimension does not match image packing at resource %u",
        alias_group_id,
        resource_index);
    return false;
  }

  return true;
}

static VkDescriptorType resolve_descriptor_type(
    unique_ptr<vgflib::ModelResourceTableDecoder>& resource_decoder,
    uint32_t index) {
  auto descriptor_type = resource_decoder->getDescriptorType(index);
  if (descriptor_type.has_value()) {
    return vgflib::ToVkDescriptorType(descriptor_type.value());
  }
  ET_LOG(
      Info,
      "Resource %u has no explicit descriptor type; assuming VK_DESCRIPTOR_TYPE_TENSOR_ARM",
      index);
  return VK_DESCRIPTOR_TYPE_TENSOR_ARM;
}

static VkPipelineStageFlags2 vgf_execution_stage_mask() {
  return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
      VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM;
}

static VkAccessFlags2 vgf_execution_read_access_mask() {
  return VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM;
}

static VkAccessFlags2 vgf_execution_write_access_mask() {
  return VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM;
}

static bool is_image_descriptor_type(VkDescriptorType descriptor_type) {
  return descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
      descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
      descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
}

static bool is_tensor_like_descriptor_type(VkDescriptorType descriptor_type) {
  return descriptor_type == VK_DESCRIPTOR_TYPE_TENSOR_ARM ||
      descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
}

static VkResult submit_and_wait_with_fence(
    VkDevice device,
    VkQueue queue,
    const VkSubmitInfo* submit_info) {
  VkFence fence = VK_NULL_HANDLE;

  const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
  };

  VkResult result = vkCreateFence(device, &fence_info, nullptr, &fence);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to create Vulkan fence, error %d", result);
    return result;
  }

  result = vkQueueSubmit(queue, 1, submit_info, fence);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Vulkan queue submit failed, error %d", result);
    vkDestroyFence(device, fence, nullptr);
    return result;
  }

  result = vkWaitForFences(
      device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max());

  vkDestroyFence(device, fence, nullptr);
  return result;
}

static void record_image_layout_transition(
    VkCommandBuffer command_buffer,
    VkImage image,
    VkImageLayout old_layout,
    VkImageLayout new_layout) {
  const VkImageMemoryBarrier2 image_barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .pNext = nullptr,
      .srcStageMask = old_layout == VK_IMAGE_LAYOUT_UNDEFINED
          ? VK_PIPELINE_STAGE_2_NONE
          : (VK_PIPELINE_STAGE_2_TRANSFER_BIT | vgf_execution_stage_mask()),
      .srcAccessMask = old_layout == VK_IMAGE_LAYOUT_UNDEFINED
          ? VK_ACCESS_2_NONE
          : (VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
             vgf_execution_read_access_mask() |
             vgf_execution_write_access_mask()),
      .dstStageMask =
          VK_PIPELINE_STAGE_2_TRANSFER_BIT | vgf_execution_stage_mask(),
      .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT |
          VK_ACCESS_2_TRANSFER_WRITE_BIT | vgf_execution_read_access_mask() |
          vgf_execution_write_access_mask(),
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange =
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .baseMipLevel = 0,
              .levelCount = 1,
              .baseArrayLayer = 0,
              .layerCount = 1,
          },
  };
  const VkDependencyInfo dependency_info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .pNext = nullptr,
      .memoryBarrierCount = 0,
      .pMemoryBarriers = nullptr,
      .bufferMemoryBarrierCount = 0,
      .pBufferMemoryBarriers = nullptr,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &image_barrier,
  };
  vkCmdPipelineBarrier2(command_buffer, &dependency_info);
}

} // namespace

void destroy_tensor(
    VkDevice device,
    VkTensorViewARM tensor_view,
    VkTensorARM tensor) {
  vkDestroyTensorViewARM(device, tensor_view, nullptr);
  vkDestroyTensorARM(device, tensor, nullptr);
}

void destroy_buffer(VkDevice device, VkBuffer buffer) {
  vkDestroyBuffer(device, buffer, nullptr);
}

void free_image(
    VkDevice device,
    VkImageView image_view,
    VkImage image,
    VkSampler sampler,
    VkDeviceMemory memory) {
  if (sampler != VK_NULL_HANDLE) {
    vkDestroySampler(device, sampler, nullptr);
  }
  if (image_view != VK_NULL_HANDLE) {
    vkDestroyImageView(device, image_view, nullptr);
  }
  if (image != VK_NULL_HANDLE) {
    vkDestroyImage(device, image, nullptr);
  }
  if (memory != VK_NULL_HANDLE) {
    vkFreeMemory(device, memory, nullptr);
  }
}

static bool find_memory_index_from_bits(
    VkPhysicalDevice vk_physical,
    uint32_t memory_type_bits,
    VkMemoryPropertyFlags aims,
    uint32_t* memory_type_out) {
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(vk_physical, &mem_properties);

  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
    if ((memory_type_bits & (0x1u << i)) != 0) {
      if ((mem_properties.memoryTypes[i].propertyFlags & aims) == aims) {
        *memory_type_out = i;
        return true;
      }
    }
  }
  return false;
}

bool VgfRepr::init_timestamp_queries() {
  const char* enable = std::getenv("EXECUTORCH_VGF_ENABLE_TIMESTAMP_QUERIES");
  if (enable == nullptr || enable[0] == '\0') {
    ET_LOG(Info, "VGF timestamp queries disabled");
    return true;
  }

  if (timestamp_queries_enabled || vk_timestamp_query_pool != VK_NULL_HANDLE) {
    return true;
  }

  if (vk_queue_family_index == UINT32_MAX) {
    ET_LOG(Info, "VGF timestamp queries disabled: unknown queue family index");
    return true;
  }

  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(
      vk_physical, &queue_family_count, nullptr);

  if (vk_queue_family_index >= queue_family_count) {
    ET_LOG(
        Info,
        "VGF timestamp queries disabled: queue family index %u is out of range",
        vk_queue_family_index);
    return true;
  }

  vector<VkQueueFamilyProperties> queue_family_properties(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(
      vk_physical, &queue_family_count, queue_family_properties.data());

  timestamp_valid_bits =
      queue_family_properties[vk_queue_family_index].timestampValidBits;

  if (timestamp_valid_bits == 0) {
    ET_LOG(
        Info,
        "VGF timestamp queries disabled: queue family %u does not support timestamps",
        vk_queue_family_index);
    return true;
  }

  VkPhysicalDeviceProperties physical_device_properties;
  vkGetPhysicalDeviceProperties(vk_physical, &physical_device_properties);

  timestamp_period_ns =
      static_cast<double>(physical_device_properties.limits.timestampPeriod);

  if (timestamp_period_ns <= 0.0) {
    ET_LOG(
        Info,
        "VGF timestampPeriod is %.6f; using fallback 52.0 ns/tick",
        timestamp_period_ns);
    timestamp_period_ns = 52.0;
  }

  VkQueryPoolCreateInfo query_pool_info{
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = 2,
      .pipelineStatistics = 0,
  };

  VkResult result = vkCreateQueryPool(
      vk_device, &query_pool_info, nullptr, &vk_timestamp_query_pool);

  if (result != VK_SUCCESS) {
    ET_LOG(
        Info,
        "VGF timestamp queries disabled: vkCreateQueryPool failed with %d",
        result);
    vk_timestamp_query_pool = VK_NULL_HANDLE;
    return true;
  }

  timestamp_queries_enabled = true;

  ET_LOG(
      Info,
      "VGF timestamp queries enabled: queue_family=%u valid_bits=%u period_ns=%.6f",
      vk_queue_family_index,
      timestamp_valid_bits,
      timestamp_period_ns);

  return true;
}

void VgfRepr::read_timestamp_queries(
    executorch::runtime::EventTracer* event_tracer) {
  if (!timestamp_queries_enabled || vk_timestamp_query_pool == VK_NULL_HANDLE) {
    return;
  }

  uint64_t timestamps[2] = {0, 0};
  VkResult result;

  {
    VGF_PROFILE_SCOPE(event_tracer, "VGF_TIMESTAMP_QUERY_READBACK");

    result = vkGetQueryPoolResults(
        vk_device,
        vk_timestamp_query_pool,
        0,
        2,
        sizeof(timestamps),
        timestamps,
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
  }

  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to read VGF timestamp query results: %d", result);
    return;
  }

  uint64_t start = timestamps[0];
  uint64_t end = timestamps[1];

  uint64_t mask = std::numeric_limits<uint64_t>::max();
  if (timestamp_valid_bits < 64) {
    mask = (1ULL << timestamp_valid_bits) - 1ULL;
    start &= mask;
    end &= mask;
  }

  uint64_t delta_ticks;
  if (end >= start) {
    delta_ticks = end - start;
  } else {
    delta_ticks = (mask - start) + end + 1ULL;
  }

  const double duration_ns =
      static_cast<double>(delta_ticks) * timestamp_period_ns;
  const double duration_ms = duration_ns / 1000000.0;

  ET_LOG(
      Info,
      "VGF_DATA_GRAPH_DEVICE_TIME ticks=%llu duration_ns=%.3f duration_ms=%.6f",
      static_cast<unsigned long long>(delta_ticks),
      duration_ns,
      duration_ms);
}

static bool find_memory_index(
    VkPhysicalDevice vk_physical,
    VkMemoryRequirements2 memory_requirements,
    VkMemoryPropertyFlags aims,
    uint32_t* memory_type_out) {
  return find_memory_index_from_bits(
      vk_physical,
      memory_requirements.memoryRequirements.memoryTypeBits,
      aims,
      memory_type_out);
}

bool VgfRepr::map_persistent_io_memory() {
  unmap_persistent_io_memory();

  for (auto& io : IOs) {
    if (io.memory == VK_NULL_HANDLE) {
      ET_LOG(Error, "Cannot persistently map null Vulkan IO memory");
      unmap_persistent_io_memory();
      return false;
    }

    void* persistent_memory = nullptr;

    // IO resources may alias the same VkDeviceMemory. Vulkan memory must not be
    // mapped more than once at the same time, so map each unique memory once
    // and share the returned pointer across aliased IO entries.
    // Make sure that memory is HOST_VISIBLE and HOST_COHERENT.
    bool found_existing_mapping = false;
    auto mapped_memory_it = std::find_if(
        persistent_mapped_memories.begin(),
        persistent_mapped_memories.end(),
        [&](const auto& mapped_memory) {
          return mapped_memory.memory == io.memory;
        });

    if (mapped_memory_it != persistent_mapped_memories.end()) {
      persistent_memory = mapped_memory_it->data;
      found_existing_mapping = true;
    }

    if (!found_existing_mapping) {
      VkResult result = vkMapMemory(
          vk_device, io.memory, 0, VK_WHOLE_SIZE, 0, &persistent_memory);
      if (result != VK_SUCCESS) {
        ET_LOG(
            Error,
            "Failed to persistently map Vulkan IO memory, error %d",
            result);
        unmap_persistent_io_memory();
        return false;
      }

      persistent_mapped_memories.push_back(PersistentMappedMemory{
          .memory = io.memory,
          .data = persistent_memory,
      });
    }

    io.persistent_memory = persistent_memory;
  }

  return true;
}

void VgfRepr::unmap_persistent_io_memory() {
  for (const auto& mapped_memory : persistent_mapped_memories) {
    if (mapped_memory.memory != VK_NULL_HANDLE &&
        mapped_memory.data != nullptr) {
      vkUnmapMemory(vk_device, mapped_memory.memory);
    }
  }
  persistent_mapped_memories.clear();

  for (auto& io : IOs) {
    io.persistent_memory = nullptr;
  }
}

VkResult allocate_memory(
    VkPhysicalDevice physical,
    VkDevice device,
    VkMemoryRequirements2 memory_requirements,
    VkMemoryPropertyFlags aims,
    VkDeviceMemory* memory,
    uint32_t* memory_type_index_out = nullptr) {
  uint32_t memory_index = 0;
  if (!find_memory_index(physical, memory_requirements, aims, &memory_index)) {
    ET_LOG(
        Error,
        "Failed to find compatible Vulkan memory type for aims 0x%x",
        static_cast<unsigned int>(aims));
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  const VkMemoryAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = nullptr,
      .allocationSize = memory_requirements.memoryRequirements.size,
      .memoryTypeIndex = memory_index,
  };

  VkResult result = vkAllocateMemory(device, &allocate_info, nullptr, memory);
  if (result == VK_SUCCESS && memory_type_index_out != nullptr) {
    *memory_type_index_out = memory_index;
  }
  return result;
}

VkResult create_tensor_unbound(
    VkDevice device,
    VkFormat format,
    uint32_t shape_size,
    const int64_t* shape,
    uint32_t stride_size,
    const int64_t* strides,
    VkTensorDescriptionARM* description,
    VkTensorARM* tensor,
    VkMemoryRequirements2* memory_requirements) {
  *description = VkTensorDescriptionARM{
      .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
      .pNext = nullptr,
      .tiling = VK_TENSOR_TILING_LINEAR_ARM,
      .format = format,
      .dimensionCount = shape_size,
      .pDimensions = shape,
      .pStrides = (0 == stride_size ? nullptr : strides),
      .usage = VK_TENSOR_USAGE_SHADER_BIT_ARM |
          VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM |
          VK_TENSOR_USAGE_TRANSFER_DST_BIT_ARM |
          VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
  };

  const VkTensorCreateInfoARM create_info = {
      .sType = VK_STRUCTURE_TYPE_TENSOR_CREATE_INFO_ARM,
      .pNext = nullptr,
      .flags = 0,
      .pDescription = description,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
  };

  VkResult result = vkCreateTensorARM(device, &create_info, nullptr, tensor);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to CreateTensor, error %d", result);
    return result;
  }

  const VkTensorMemoryRequirementsInfoARM memory_requirements_info = {
      .sType = VK_STRUCTURE_TYPE_TENSOR_MEMORY_REQUIREMENTS_INFO_ARM,
      .pNext = nullptr,
      .tensor = *tensor,
  };
  *memory_requirements = VkMemoryRequirements2{
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
      .pNext = nullptr,
  };
  vkGetTensorMemoryRequirementsARM(
      device, &memory_requirements_info, memory_requirements);
  return VK_SUCCESS;
}

VkTensorDescriptionARM make_data_graph_descriptor(
    VkFormat format,
    uint32_t shape_size,
    const int64_t* shape,
    uint32_t stride_size,
    const int64_t* strides) {
  return VkTensorDescriptionARM{
      .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
      .pNext = nullptr,
      .tiling = VK_TENSOR_TILING_LINEAR_ARM,
      .format = format,
      .dimensionCount = shape_size,
      .pDimensions = shape,
      .pStrides = (0 == stride_size ? nullptr : strides),
      .usage = VK_TENSOR_USAGE_SHADER_BIT_ARM |
          VK_TENSOR_USAGE_TRANSFER_SRC_BIT_ARM |
          VK_TENSOR_USAGE_TRANSFER_DST_BIT_ARM |
          VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
  };
}

VkResult bind_tensor_memory_and_create_view(
    VkDevice device,
    VkFormat format,
    VkTensorARM tensor,
    VkDeviceMemory memory,
    VkTensorViewARM* tensor_view) {
  const VkBindTensorMemoryInfoARM bind_info = {
      .sType = VK_STRUCTURE_TYPE_BIND_TENSOR_MEMORY_INFO_ARM,
      .pNext = nullptr,
      .tensor = tensor,
      .memory = memory,
      .memoryOffset = 0,
  };
  VkResult result = vkBindTensorMemoryARM(device, 1, &bind_info);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to bind tensor memory, error %d", result);
    return result;
  }

  VkTensorViewCreateInfoARM tensor_view_info = {
      .sType = VK_STRUCTURE_TYPE_TENSOR_VIEW_CREATE_INFO_ARM,
      .pNext = nullptr,
      .flags = 0,
      .tensor = tensor,
      .format = format,
  };
  return vkCreateTensorViewARM(device, &tensor_view_info, nullptr, tensor_view);
}

VkResult create_buffer_unbound(
    VkDevice device,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkBuffer* buffer,
    VkMemoryRequirements2* memory_requirements) {
  VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
  };
  VkResult result = vkCreateBuffer(device, &buffer_info, nullptr, buffer);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to create buffer, error %d", result);
    return result;
  }

  VkMemoryRequirements memory_requirements1 = {};
  vkGetBufferMemoryRequirements(device, *buffer, &memory_requirements1);
  *memory_requirements = VkMemoryRequirements2{
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
      .pNext = nullptr,
      .memoryRequirements = memory_requirements1,
  };
  return VK_SUCCESS;
}

VkResult
bind_buffer_memory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory) {
  return vkBindBufferMemory(device, buffer, memory, 0);
}

VkResult create_image_unbound(
    VkDevice device,
    VkFormat format,
    VkExtent3D extent,
    VkImageUsageFlags usage,
    VkImage* image,
    VkMemoryRequirements2* memory_requirements) {
  const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = extent,
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  VkResult result = vkCreateImage(device, &image_info, nullptr, image);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to create image, error %d", result);
    return result;
  }

  VkMemoryRequirements reqs = {};
  vkGetImageMemoryRequirements(device, *image, &reqs);
  *memory_requirements = VkMemoryRequirements2{
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
      .pNext = nullptr,
      .memoryRequirements = reqs,
  };
  return VK_SUCCESS;
}

VkResult bind_image_memory_and_create_view(
    VkDevice device,
    VkFormat format,
    VkImage image,
    VkDeviceMemory memory,
    VkImageView* image_view) {
  VkResult result = vkBindImageMemory(device, image, memory, 0);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to bind image memory, error %d", result);
    return result;
  }

  const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .components =
          {
              .r = VK_COMPONENT_SWIZZLE_IDENTITY,
              .g = VK_COMPONENT_SWIZZLE_IDENTITY,
              .b = VK_COMPONENT_SWIZZLE_IDENTITY,
              .a = VK_COMPONENT_SWIZZLE_IDENTITY,
          },
      .subresourceRange =
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .baseMipLevel = 0,
              .levelCount = 1,
              .baseArrayLayer = 0,
              .layerCount = 1,
          },
  };
  return vkCreateImageView(device, &view_info, nullptr, image_view);
}

VkResult allocate_buffer(
    VkPhysicalDevice physical,
    VkDevice device,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkBuffer* buffer,
    VkDeviceMemory* memory) {
  VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
  };
  VkResult result = vkCreateBuffer(device, &buffer_info, nullptr, buffer);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to create buffer, error %d", result);
    return result;
  }

  VkMemoryRequirements memory_requirements = {};
  vkGetBufferMemoryRequirements(device, *buffer, &memory_requirements);
  VkMemoryRequirements2 memory_requirements2 = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
      .pNext = nullptr,
      .memoryRequirements = memory_requirements,
  };

  VkMemoryPropertyFlags aims = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  uint32_t memory_index = 0;
  if (!find_memory_index(physical, memory_requirements2, aims, &memory_index)) {
    ET_LOG(Error, "Failed to find buffer memory type");
    vkDestroyBuffer(device, *buffer, nullptr);
    *buffer = VK_NULL_HANDLE;
    return VK_ERROR_FEATURE_NOT_PRESENT;
  }

  const VkMemoryAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = nullptr,
      .allocationSize = memory_requirements.size,
      .memoryTypeIndex = memory_index,
  };
  result = vkAllocateMemory(device, &allocate_info, nullptr, memory);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to allocate buffer memory, error %d", result);
    return result;
  }

  result = vkBindBufferMemory(device, *buffer, *memory, 0);
  if (result != VK_SUCCESS) {
    ET_LOG(Error, "Failed to bind buffer memory, error %d", result);
    return result;
  }

  return VK_SUCCESS;
}

VkResult allocate_sampler(
    VkDevice device,
    VkFilter min_filter,
    VkFilter mag_filter,
    VkSamplerAddressMode address_mode_u,
    VkSamplerAddressMode address_mode_v,
    VkBorderColor border_color,
    VkSampler* sampler) {
  const VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .magFilter = mag_filter,
      .minFilter = min_filter,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = address_mode_u,
      .addressModeV = address_mode_v,
      .addressModeW = address_mode_v,
      .mipLodBias = 0.0f,
      .anisotropyEnable = VK_FALSE,
      .maxAnisotropy = 1.0f,
      .compareEnable = VK_FALSE,
      .compareOp = VK_COMPARE_OP_NEVER,
      .minLod = 0.0f,
      .maxLod = 0.0f,
      .borderColor = border_color,
      .unnormalizedCoordinates = VK_FALSE,
  };
  return vkCreateSampler(device, &sampler_info, nullptr, sampler);
}

static std::optional<uint32_t> get_resource_alias_group_id(
    const unique_ptr<vgflib::ModelResourceTableDecoder>& resource_decoder,
    uint32_t resource_index) {
#if EXECUTORCH_ARM_VGF_HAS_DECODER_V10_APIS
  auto alias_group = resource_decoder->getAliasGroupId(resource_index);
  if (!alias_group.has_value()) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(*alias_group);
#else
  (void)resource_decoder;
  (void)resource_index;
  return std::nullopt;
#endif
}

static bool allocate_resource_sampler(
    const unique_ptr<vgflib::ModelResourceTableDecoder>& resource_decoder,
    uint32_t resource_index,
    VkDevice device,
    VkSampler* sampler_out) {
#if EXECUTORCH_ARM_VGF_HAS_DECODER_V10_APIS
  auto sampler_config =
      resource_decoder->getSamplerConfigHandle(resource_index);
  if (sampler_config == nullptr) {
    ET_LOG(
        Error,
        "Missing sampler config for combined image sampler resource %u",
        resource_index);
    return false;
  }

  auto result = allocate_sampler(
      device,
      static_cast<VkFilter>(
          resource_decoder->getSamplerConfigMinFilter(sampler_config)),
      static_cast<VkFilter>(
          resource_decoder->getSamplerConfigMagFilter(sampler_config)),
      static_cast<VkSamplerAddressMode>(
          resource_decoder->getSamplerConfigAddressModeU(sampler_config)),
      static_cast<VkSamplerAddressMode>(
          resource_decoder->getSamplerConfigAddressModeV(sampler_config)),
      static_cast<VkBorderColor>(
          resource_decoder->getSamplerConfigBorderColor(sampler_config)),
      sampler_out);
#else
  (void)resource_decoder;
  auto result = allocate_sampler(
      device,
      VK_FILTER_LINEAR,
      VK_FILTER_LINEAR,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
      sampler_out);
#endif
  if (result != VK_SUCCESS) {
    ET_LOG(
        Error,
        "Failed to create sampler for VGF resource %u, error %d",
        resource_index,
        result);
    return false;
  }
  return true;
}

static auto get_module_spirv_code(
    unique_ptr<vgflib::ModuleTableDecoder>& module_decoder,
    uint32_t module_index) {
#if EXECUTORCH_ARM_VGF_HAS_DECODER_V10_APIS
  return module_decoder->getSPIRVModuleCode(module_index);
#else
  return module_decoder->getModuleCode(module_index);
#endif
}

static uint32_t get_segment_descriptor_set_index(
    const unique_ptr<vgflib::ModelSequenceTableDecoder>& sequence_decoder,
    uint32_t segment_index,
    uint32_t descriptor_index) {
#if EXECUTORCH_ARM_VGF_HAS_DECODER_V10_APIS
  return sequence_decoder->getSegmentDescriptorSetIndex(
      segment_index, descriptor_index);
#else
  (void)sequence_decoder;
  (void)segment_index;
  return descriptor_index;
#endif
}

VkResult transition_image_layout(
    VkDevice device,
    VkCommandPool command_pool,
    VkQueue queue,
    VkImage image,
    VkImageLayout old_layout,
    VkImageLayout new_layout) {
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  const VkCommandBufferAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = nullptr,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  VkResult result =
      vkAllocateCommandBuffers(device, &allocate_info, &command_buffer);
  if (result != VK_SUCCESS) {
    return result;
  }

  const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .pNext = nullptr,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      .pInheritanceInfo = nullptr,
  };
  result = vkBeginCommandBuffer(command_buffer, &begin_info);
  if (result != VK_SUCCESS) {
    vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
    return result;
  }

  const VkImageMemoryBarrier2 image_barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .pNext = nullptr,
      .srcStageMask = old_layout == VK_IMAGE_LAYOUT_UNDEFINED
          ? VK_PIPELINE_STAGE_2_NONE
          : (VK_PIPELINE_STAGE_2_TRANSFER_BIT |
             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
             VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM),
      .srcAccessMask = old_layout == VK_IMAGE_LAYOUT_UNDEFINED
          ? VK_ACCESS_2_NONE
          : (VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
             VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
             VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM |
             VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM),

      .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT |
          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
          VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM,
      .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT |
          VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT |
          VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM |
          VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange =
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .baseMipLevel = 0,
              .levelCount = 1,
              .baseArrayLayer = 0,
              .layerCount = 1,
          },
  };
  const VkDependencyInfo dependency_info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .pNext = nullptr,
      .memoryBarrierCount = 0,
      .pMemoryBarriers = nullptr,
      .bufferMemoryBarrierCount = 0,
      .pBufferMemoryBarriers = nullptr,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &image_barrier,
  };
  vkCmdPipelineBarrier2(command_buffer, &dependency_info);

  result = vkEndCommandBuffer(command_buffer);
  if (result != VK_SUCCESS) {
    vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
    return result;
  }

  const VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = nullptr,
      .waitSemaphoreCount = 0,
      .pWaitSemaphores = nullptr,
      .pWaitDstStageMask = nullptr,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
      .signalSemaphoreCount = 0,
      .pSignalSemaphores = nullptr,
  };

  // creates a temporary one-time command buffer, submits it once, waits, and
  // frees it immediately.
  result = submit_and_wait_with_fence(device, queue, &submit_info);

  vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
  return result;
}

static void debug_print_sequence(
    unique_ptr<vgflib::ModelSequenceTableDecoder>& sequence_decoder) {
  auto module_type_to_string = [](vgflib::ModuleType type) {
    switch (type) {
      case vgflib::ModuleType::GRAPH:
        return "GRAPH";
      case vgflib::ModuleType::COMPUTE:
        return "COMPUTE";
      default:
        return "UNKNOWN";
    }
  };
  ET_LOG(Info, "VGF Sequences:");
  for (int i = 0; i < sequence_decoder->modelSequenceTableSize(); i++) {
    ET_LOG(
        Info,
        "  Sequence(%d) '%s':",
        i,
        string(sequence_decoder->getSegmentName(i)).c_str());
    auto dispatch_shape = sequence_decoder->getSegmentDispatchShape(i);
    ET_LOG(
        Info,
        "    dispatch shape %d %d %d",
        dispatch_shape[0],
        dispatch_shape[1],
        dispatch_shape[2]);
    ET_LOG(
        Info,
        "    segment type %s",
        module_type_to_string(sequence_decoder->getSegmentType(i)));
    ET_LOG(
        Info,
        "    module index %d",
        sequence_decoder->getSegmentModuleIndex(i));
    auto input_names = sequence_decoder->getModelSequenceInputNamesHandle();
    ET_LOG(
        Info, "    names (%ld):", sequence_decoder->getNamesSize(input_names));
    for (int j = 0; j < sequence_decoder->getNamesSize(input_names); j++) {
      ET_LOG(
          Info,
          "      %d: %s",
          j,
          string(sequence_decoder->getName(input_names, j)).c_str());
    }
  }
}

template <typename Handle>
static const void* log_handle_ptr(Handle handle) {
  if constexpr (std::is_pointer_v<Handle>) {
    return handle;
  } else {
    return reinterpret_cast<const void*>(static_cast<uintptr_t>(handle));
  }
}

static void debug_print_modules(
    unique_ptr<vgflib::ModuleTableDecoder>& module_decoder) {
  auto module_type_to_string = [](vgflib::ModuleType type) {
    switch (type) {
      case vgflib::ModuleType::GRAPH:
        return "GRAPH";
      case vgflib::ModuleType::COMPUTE:
        return "COMPUTE";
      default:
        return "UNKNOWN";
    }
  };
  ET_LOG(Info, "VGF Modules:");
  for (int i = 0; i < module_decoder->size(); i++) {
    auto name = string(module_decoder->getModuleName(i));
    auto entrypoint = string(module_decoder->getModuleEntryPoint(i));
    auto type = module_decoder->getModuleType(i);
    auto spirv = module_decoder->getModuleCode(i);
    ET_LOG(Info, "  Module(%d) '%s':", i, name.c_str());
    ET_LOG(Info, "    type %s", module_type_to_string(type));
    ET_LOG(Info, "    entrypoint '%s'", entrypoint.c_str());
    ET_LOG(Info, "    has spirv %d", module_decoder->hasSPIRV(i));
    ET_LOG(
        Info, "    code size %lu", spirv.size()); // read the .begin() to .end()
  }
}

bool VgfRepr::process_vgf(
    const char* vgf_data,
    size_t vgf_size,
    ArrayRef<CompileSpec> specs,
    executorch::runtime::EventTracer* event_tracer) {
  VGF_PROFILE_SCOPE(event_tracer, "VGF_INIT_PROCESS_VGF");
  (void)specs;

  ET_LOG(Info, "Preparing VGF as Vulkan objects");

  VkResult result;

  unique_ptr<vgflib::HeaderDecoder> header_decoder;
  unique_ptr<vgflib::ModelSequenceTableDecoder> sequence_decoder;
  unique_ptr<vgflib::ModuleTableDecoder> module_decoder;
  unique_ptr<vgflib::ModelResourceTableDecoder> resource_decoder;
  unique_ptr<vgflib::ConstantDecoder> constant_decoder;

  {
    VGF_PROFILE_SCOPE(event_tracer, "VGF_INIT_DECODE_TABLES");

    // Prepare temporary decoders
    header_decoder =
        vgflib::CreateHeaderDecoder(vgf_data, vgflib::HeaderSize(), vgf_size);
    if (!header_decoder) {
      ET_LOG(Error, "Failed to create VGF header decoder");
      return false;
    }

    sequence_decoder = vgflib::CreateModelSequenceTableDecoder(
        vgf_data + header_decoder->GetModelSequenceTableOffset(),
        header_decoder->GetModelSequenceTableSize());
    module_decoder = vgflib::CreateModuleTableDecoder(
        vgf_data + header_decoder->GetModuleTableOffset(),
        header_decoder->GetModuleTableSize());
    resource_decoder = vgflib::CreateModelResourceTableDecoder(
        vgf_data + header_decoder->GetModelResourceTableOffset(),
        header_decoder->GetModelResourceTableSize());
    constant_decoder = vgflib::CreateConstantDecoder(
        vgf_data + header_decoder->GetConstantsOffset(),
        header_decoder->GetConstantsSize());
    // Check the VGF decoders
    if (not(header_decoder && module_decoder && sequence_decoder &&
            resource_decoder && constant_decoder && header_decoder->IsValid() &&
            header_decoder->CheckVersion())) {
      ET_LOG(Error, "Failed to process VGF file internalsr");
      return false;
    }
  }

  // Parse the sequences in the VGF (there can be multiple segments).
  debug_print_sequence(sequence_decoder);
  const int segment_count = sequence_decoder->modelSequenceTableSize();
  if (segment_count <= 0) {
    ET_LOG(Error, "Expected at least one segment");
    return false;
  }

  // Extract modules
  debug_print_modules(module_decoder);

  // Load our resource (tensors, constants) into their appropriate Vk objects
  struct ResourceBinding {
    VkDescriptorType descriptor_type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    VkTensorViewARM tensor_view = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkImageView image_view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDeviceSize buffer_size = 0;
  };
  vector<VkTensorDescriptionARM> descriptors(resource_decoder->size());
  vector<bool> descriptor_valid(resource_decoder->size(), false);
  vector<ResourceBinding> resource_bindings(resource_decoder->size());
  vector<int> resource_index_to_io_index(resource_decoder->size(), -1);
  struct AliasBacking {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize allocation_size = 0;
    uint32_t memory_type_bits = 0;
    uint32_t memory_type_index = UINT32_MAX;
    VkMemoryPropertyFlags required_memory_properties = 0;
    bool requirements_ready = false;
  };
  struct AliasGroupUsage {
    bool has_image = false;
    bool has_tensor_like = false;
  };
  struct AliasImageState {
    bool needs_tensor_aliasing = false;
    VkImageLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    vector<VkImage> images;
  };
  unordered_map<uint32_t, AliasBacking> alias_backings;
  unordered_map<uint32_t, AliasGroupUsage> alias_group_usage;
  unordered_map<uint32_t, AliasLogicalContract> alias_logical_contracts;
  unordered_map<uint32_t, AliasImageState> alias_image_states;
  int IO_count = resource_decoder->size();

  for (int i = 0; i < IO_count; i++) {
    auto alias_group = get_resource_alias_group_id(resource_decoder, i);
    if (!alias_group.has_value()) {
      continue;
    }
    auto& usage = alias_group_usage[*alias_group];
    auto descriptor_type = resolve_descriptor_type(resource_decoder, i);
    if (is_image_descriptor_type(descriptor_type)) {
      usage.has_image = true;
    }
    if (is_tensor_like_descriptor_type(descriptor_type)) {
      usage.has_tensor_like = true;
    }
  }

  auto alias_memory_properties_for_descriptor_type =
      [](VkDescriptorType descriptor_type) -> VkMemoryPropertyFlags {
    if (is_tensor_like_descriptor_type(descriptor_type)) {
      return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
    if (is_image_descriptor_type(descriptor_type)) {
      return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
    return 0;
  };

  for (int i = 0; i < IO_count; i++) {
    auto alias_group = get_resource_alias_group_id(resource_decoder, i);
    if (!alias_group.has_value()) {
      continue;
    }

    auto resource_type = resolve_descriptor_type(resource_decoder, i);
    auto resource_format = vgflib::ToVkFormat(resource_decoder->getVkFormat(i));
    auto shape = resource_decoder->getTensorShape(i);
    auto stride = resource_decoder->getTensorStride(i);
    const vector<int64_t> the_shape(shape.begin(), shape.end());
    const vector<int64_t> the_stride(stride.begin(), stride.end());

    if (!validate_alias_group_logical_contract(
            *alias_group,
            i,
            resource_type,
            resource_format,
            the_shape,
            the_stride,
            &alias_logical_contracts[*alias_group])) {
      return false;
    }

    VkMemoryRequirements2 memory_requirements = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = nullptr,
    };
    if (resource_type == VK_DESCRIPTOR_TYPE_TENSOR_ARM) {
      VkTensorDescriptionARM tensor_description;
      VkTensorARM tensor = VK_NULL_HANDLE;
      result = create_tensor_unbound(
          vk_device,
          resource_format,
          shape.size() == 0 ? 1 : static_cast<uint32_t>(shape.size()),
          shape.size() == 0 ? &kScalarSentinelDimension : shape.begin(),
          static_cast<uint32_t>(stride.size()),
          stride.begin(),
          &tensor_description,
          &tensor,
          &memory_requirements);
      if (result != VK_SUCCESS) {
        ET_LOG(
            Error,
            "Failed to query tensor memory requirements for VGF resource %d",
            i);
        return false;
      }
      destroy_tensor(vk_device, VK_NULL_HANDLE, tensor);
    } else if (resource_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
      const VkDeviceSize buffer_size = element_count_from_shape(the_shape) *
          get_format_size(resource_format);
      VkBuffer buffer = VK_NULL_HANDLE;
      result = create_buffer_unbound(
          vk_device,
          buffer_size,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          &buffer,
          &memory_requirements);
      if (result != VK_SUCCESS) {
        ET_LOG(
            Error,
            "Failed to query buffer memory requirements for VGF resource %d",
            i);
        return false;
      }
      destroy_buffer(vk_device, buffer);
    } else if (is_image_descriptor_type(resource_type)) {
      VkExtent3D image_extent = {};
      if (!validate_image_shape_and_format(
              the_shape, resource_format, &image_extent)) {
        return false;
      }
      const VkImageUsageFlags image_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
          VK_IMAGE_USAGE_TRANSFER_DST_BIT |
          ((alias_group_usage[*alias_group].has_tensor_like)
               ? VK_IMAGE_USAGE_TENSOR_ALIASING_BIT_ARM
               : 0) |
          ((resource_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
               ? VK_IMAGE_USAGE_STORAGE_BIT
               : VK_IMAGE_USAGE_SAMPLED_BIT);
      VkImage image = VK_NULL_HANDLE;
      result = create_image_unbound(
          vk_device,
          resource_format,
          image_extent,
          image_usage,
          &image,
          &memory_requirements);
      if (result != VK_SUCCESS) {
        ET_LOG(
            Error,
            "Failed to query image memory requirements for VGF resource %d",
            i);
        return false;
      }
      vkDestroyImage(vk_device, image, nullptr);
    } else {
      ET_LOG(
          Error,
          "Alias group %u contains unsupported resource %d",
          *alias_group,
          i);
      return false;
    }

    auto& alias_backing = alias_backings[*alias_group];
    if (!alias_backing.requirements_ready) {
      alias_backing.requirements_ready = true;
      alias_backing.allocation_size =
          memory_requirements.memoryRequirements.size;
      alias_backing.memory_type_bits =
          memory_requirements.memoryRequirements.memoryTypeBits;
      alias_backing.required_memory_properties =
          alias_memory_properties_for_descriptor_type(resource_type);
    } else {
      alias_backing.allocation_size = std::max(
          alias_backing.allocation_size,
          memory_requirements.memoryRequirements.size);
      alias_backing.memory_type_bits &=
          memory_requirements.memoryRequirements.memoryTypeBits;
      alias_backing.required_memory_properties |=
          alias_memory_properties_for_descriptor_type(resource_type);
    }
  }

  for (auto& [alias_group, alias_backing] : alias_backings) {
    if (!alias_backing.requirements_ready) {
      continue;
    }
    if (alias_backing.memory_type_bits == 0) {
      ET_LOG(
          Error,
          "Alias group %u has no common Vulkan memory type bits",
          alias_group);
      return false;
    }
    if (!find_memory_index_from_bits(
            vk_physical,
            alias_backing.memory_type_bits,
            alias_backing.required_memory_properties,
            &alias_backing.memory_type_index)) {
      ET_LOG(
          Error,
          "Alias group %u has no compatible Vulkan memory type",
          alias_group);
      return false;
    }
  }

  for (int i = 0; i < IO_count; i++) {
    auto resource_type = resolve_descriptor_type(resource_decoder, i);
    auto resource_format = vgflib::ToVkFormat(resource_decoder->getVkFormat(i));
    auto alias_group = get_resource_alias_group_id(resource_decoder, i);

    // Get tensor shape and strides
    auto shape = resource_decoder->getTensorShape(i);
    auto stride = resource_decoder->getTensorStride(i);
    const vector<int64_t> the_shape(shape.begin(), shape.end());
    const vector<int64_t> the_stride(stride.begin(), stride.end());
    const auto shape_size = shape.size();
    const bool uses_alias_group = alias_group.has_value();

    auto get_alias_backing = [&]() -> AliasBacking* {
      if (!uses_alias_group) {
        return nullptr;
      }
      return &alias_backings[*alias_group];
    };

    auto prepare_alias_memory =
        [&](const VkMemoryRequirements2& memory_requirements,
            const char* resource_kind,
            VkDeviceMemory* memory_out,
            bool* owns_memory_out) -> bool {
      auto* alias_backing = get_alias_backing();
      if (alias_backing == nullptr) {
        return false;
      }

      const uint32_t type_mask = 1u << alias_backing->memory_type_index;
      if ((memory_requirements.memoryRequirements.memoryTypeBits & type_mask) ==
              0 ||
          memory_requirements.memoryRequirements.size >
              alias_backing->allocation_size) {
        ET_LOG(
            Error,
            "Alias group %u is incompatible with %s resource %d",
            *alias_group,
            resource_kind,
            i);
        return false;
      }

      if (alias_backing->memory == VK_NULL_HANDLE) {
        const VkMemoryAllocateInfo allocate_info = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = alias_backing->allocation_size,
            .memoryTypeIndex = alias_backing->memory_type_index,
        };
        VkResult alias_alloc_result = vkAllocateMemory(
            vk_device, &allocate_info, nullptr, &alias_backing->memory);
        if (alias_alloc_result != VK_SUCCESS) {
          ET_LOG(
              Error,
              "Failed to allocate aliased %s memory for VGF resource %d",
              resource_kind,
              i);
          return false;
        }
        *owns_memory_out = true;
      } else {
        *owns_memory_out = false;
      }

      *memory_out = alias_backing->memory;
      return true;
    };

    switch (resource_decoder->getCategory(i)) {
      case vgflib::ResourceCategory::INPUT:
      case vgflib::ResourceCategory::OUTPUT: {
        size_t e_size = get_format_size(resource_format);
        if (0 == e_size) {
          ET_LOG(Error, "failed to get element size of VkFormat");
          return false;
        }

        bool is_in =
            resource_decoder->getCategory(i) == vgflib::ResourceCategory::INPUT;

        if (resource_type == VK_DESCRIPTOR_TYPE_TENSOR_ARM) {
          VkTensorARM tensor = VK_NULL_HANDLE;
          VkTensorViewARM tensor_view = VK_NULL_HANDLE;
          VkTensorDescriptionARM tensor_description;
          VkMemoryRequirements2 tensor_memory_requirements = {
              .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
              .pNext = nullptr,
          };
          result = create_tensor_unbound(
              vk_device,
              resource_format,
              shape_size == 0 ? 1 : static_cast<uint32_t>(shape_size),
              shape_size == 0 ? &kScalarSentinelDimension : shape.begin(),
              static_cast<uint32_t>(stride.size()),
              stride.begin(),
              &tensor_description,
              &tensor,
              &tensor_memory_requirements);
          if (result != VK_SUCCESS) {
            ET_LOG(Error, "Failed to allocate tensor for VGF resource %d", i);
            return false;
          }
          VkDeviceMemory tensor_memory = VK_NULL_HANDLE;
          bool owns_memory = true;
          auto* alias_backing = get_alias_backing();
          if (alias_backing != nullptr) {
            if (!prepare_alias_memory(
                    tensor_memory_requirements,
                    "tensor",
                    &tensor_memory,
                    &owns_memory)) {
              destroy_tensor(vk_device, VK_NULL_HANDLE, tensor);
              return false;
            }
          } else {
            const VkMemoryPropertyFlags aims =
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            result = allocate_memory(
                vk_physical,
                vk_device,
                tensor_memory_requirements,
                aims,
                &tensor_memory);
            if (result != VK_SUCCESS) {
              destroy_tensor(vk_device, VK_NULL_HANDLE, tensor);
              ET_LOG(
                  Error,
                  "Failed to allocate tensor memory for VGF resource %d",
                  i);
              return false;
            }
          }
          result = bind_tensor_memory_and_create_view(
              vk_device, resource_format, tensor, tensor_memory, &tensor_view);
          if (result != VK_SUCCESS) {
            if (owns_memory) {
              vkFreeMemory(vk_device, tensor_memory, nullptr);
            }
            destroy_tensor(vk_device, VK_NULL_HANDLE, tensor);
            ET_LOG(Error, "Failed to bind tensor for VGF resource %d", i);
            return false;
          }

          IOs.push_back(
              IO{the_shape,
                 the_stride,
                 e_size,
                 element_count_from_shape(the_shape) * e_size,
                 VK_DESCRIPTOR_TYPE_TENSOR_ARM,
                 tensor,
                 tensor_view,
                 VK_NULL_HANDLE,
                 VK_NULL_HANDLE,
                 VK_NULL_HANDLE,
                 VK_NULL_HANDLE,
                 VK_NULL_HANDLE,
                 tensor_memory,
                 {0, 0, 0},
                 nullptr,
                 owns_memory,
                 true,
                 is_in});
          resource_index_to_io_index[i] = static_cast<int>(IOs.size() - 1);

          resource_bindings[i] = ResourceBinding{
              .descriptor_type = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
              .tensor_view = tensor_view,
              .buffer = VK_NULL_HANDLE,
              .image_view = VK_NULL_HANDLE,
              .sampler = VK_NULL_HANDLE,
              .buffer_size = 0,
          };
          descriptors[i] = tensor_description;
          descriptor_valid[i] = true;
        } else if (resource_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
          VkDeviceSize buffer_size =
              element_count_from_shape(the_shape) * e_size;

          VkBuffer buffer = VK_NULL_HANDLE;
          VkMemoryRequirements2 buffer_memory_requirements = {
              .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
              .pNext = nullptr,
          };
          result = create_buffer_unbound(
              vk_device,
              buffer_size,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
              &buffer,
              &buffer_memory_requirements);
          if (result != VK_SUCCESS) {
            ET_LOG(Error, "Failed to allocate buffer for VGF resource %d", i);
            return false;
          }
          VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
          bool owns_memory = true;
          auto* alias_backing = get_alias_backing();
          if (alias_backing != nullptr) {
            if (!prepare_alias_memory(
                    buffer_memory_requirements,
                    "buffer",
                    &buffer_memory,
                    &owns_memory)) {
              destroy_buffer(vk_device, buffer);
              return false;
            }
          } else {
            const VkMemoryPropertyFlags aims =
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            result = allocate_memory(
                vk_physical,
                vk_device,
                buffer_memory_requirements,
                aims,
                &buffer_memory);
            if (result != VK_SUCCESS) {
              destroy_buffer(vk_device, buffer);
              ET_LOG(
                  Error,
                  "Failed to allocate buffer memory for VGF resource %d",
                  i);
              return false;
            }
          }
          result = bind_buffer_memory(vk_device, buffer, buffer_memory);
          if (result != VK_SUCCESS) {
            if (owns_memory) {
              vkFreeMemory(vk_device, buffer_memory, nullptr);
            }
            destroy_buffer(vk_device, buffer);
            ET_LOG(
                Error, "Failed to bind buffer memory for VGF resource %d", i);
            return false;
          }

          IOs.push_back(
              IO{the_shape,
                 the_stride,
                 e_size,
                 static_cast<size_t>(buffer_size),
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                 VK_NULL_HANDLE,
                 VK_NULL_HANDLE,
                 buffer,
                 VK_NULL_HANDLE,
                 VK_NULL_HANDLE,
                 VK_NULL_HANDLE,
                 VK_NULL_HANDLE,
                 buffer_memory,
                 {0, 0, 0},
                 nullptr,
                 owns_memory,
                 true,
                 is_in});
          resource_index_to_io_index[i] = static_cast<int>(IOs.size() - 1);

          resource_bindings[i] = ResourceBinding{
              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .tensor_view = VK_NULL_HANDLE,
              .buffer = buffer,
              .image_view = VK_NULL_HANDLE,
              .sampler = VK_NULL_HANDLE,
              .buffer_size = buffer_size,
          };
        } else if (
            resource_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
            resource_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
            resource_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
          VkExtent3D image_extent = {};
          size_t image_allocation_size = 0;
          if (!validate_image_shape_and_format(
                  the_shape,
                  resource_format,
                  &image_extent,
                  &image_allocation_size)) {
            return false;
          }
          const VkImageUsageFlags image_usage =
              VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
              VK_IMAGE_USAGE_TRANSFER_DST_BIT |
              ((uses_alias_group &&
                alias_group_usage[*alias_group].has_tensor_like)
                   ? VK_IMAGE_USAGE_TENSOR_ALIASING_BIT_ARM
                   : 0) |
              ((resource_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                   ? VK_IMAGE_USAGE_STORAGE_BIT
                   : VK_IMAGE_USAGE_SAMPLED_BIT);
          VkImage image = VK_NULL_HANDLE;
          VkImageView image_view = VK_NULL_HANDLE;
          VkMemoryRequirements2 image_memory_requirements = {
              .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
              .pNext = nullptr,
          };
          result = create_image_unbound(
              vk_device,
              resource_format,
              image_extent,
              image_usage,
              &image,
              &image_memory_requirements);
          if (result != VK_SUCCESS) {
            ET_LOG(Error, "Failed to allocate image for VGF resource %d", i);
            return false;
          }
          VkDeviceMemory image_memory = VK_NULL_HANDLE;
          bool owns_image_memory = true;
          auto* alias_backing = get_alias_backing();
          if (alias_backing != nullptr) {
            if (!prepare_alias_memory(
                    image_memory_requirements,
                    "image",
                    &image_memory,
                    &owns_image_memory)) {
              free_image(
                  vk_device,
                  VK_NULL_HANDLE,
                  image,
                  VK_NULL_HANDLE,
                  VK_NULL_HANDLE);
              return false;
            }
          } else {
            const VkMemoryPropertyFlags aims =
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            result = allocate_memory(
                vk_physical,
                vk_device,
                image_memory_requirements,
                aims,
                &image_memory);
            if (result != VK_SUCCESS) {
              free_image(
                  vk_device,
                  VK_NULL_HANDLE,
                  image,
                  VK_NULL_HANDLE,
                  VK_NULL_HANDLE);
              ET_LOG(
                  Error,
                  "Failed to allocate image memory for VGF resource %d",
                  i);
              return false;
            }
          }
          result = bind_image_memory_and_create_view(
              vk_device, resource_format, image, image_memory, &image_view);
          if (result != VK_SUCCESS) {
            free_image(
                vk_device,
                VK_NULL_HANDLE,
                image,
                VK_NULL_HANDLE,
                owns_image_memory ? image_memory : VK_NULL_HANDLE);
            ET_LOG(Error, "Failed to bind image for VGF resource %d", i);
            return false;
          }
          const bool needs_tensor_aliasing = uses_alias_group &&
              alias_group_usage[*alias_group].has_tensor_like;
          const VkImageLayout initial_layout = needs_tensor_aliasing
              ? VK_IMAGE_LAYOUT_TENSOR_ALIASING_ARM
              : VK_IMAGE_LAYOUT_GENERAL;
          result = transition_image_layout(
              vk_device,
              vk_command_pool,
              vk_queue,
              image,
              VK_IMAGE_LAYOUT_UNDEFINED,
              initial_layout);
          if (result != VK_SUCCESS) {
            ET_LOG(Error, "Failed to transition image for VGF resource %d", i);
            free_image(
                vk_device,
                image_view,
                image,
                VK_NULL_HANDLE,
                owns_image_memory ? image_memory : VK_NULL_HANDLE);
            return false;
          }

          VkSampler sampler = VK_NULL_HANDLE;
          if (resource_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            if (!allocate_resource_sampler(
                    resource_decoder, i, vk_device, &sampler)) {
              free_image(
                  vk_device,
                  image_view,
                  image,
                  VK_NULL_HANDLE,
                  owns_image_memory ? image_memory : VK_NULL_HANDLE);
              return false;
            }
          }
          if (uses_alias_group) {
            auto& alias_state = alias_image_states[*alias_group];
            alias_state.needs_tensor_aliasing = needs_tensor_aliasing;
            alias_state.current_layout = initial_layout;
            alias_state.images.push_back(image);
          }
          VkBuffer staging_buffer = VK_NULL_HANDLE;
          VkDeviceMemory staging_memory = VK_NULL_HANDLE;
          result = allocate_buffer(
              vk_physical,
              vk_device,
              image_allocation_size,
              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              &staging_buffer,
              &staging_memory);
          if (result != VK_SUCCESS) {
            ET_LOG(
                Error,
                "Failed to allocate staging buffer for image VGF resource %d",
                i);
            free_image(
                vk_device,
                image_view,
                image,
                sampler,
                owns_image_memory ? image_memory : VK_NULL_HANDLE);
            return false;
          }

          IOs.push_back(
              IO{the_shape,
                 the_stride,
                 e_size,
                 image_allocation_size,
                 resource_type,
                 VK_NULL_HANDLE,
                 VK_NULL_HANDLE,
                 staging_buffer,
                 image,
                 image_view,
                 sampler,
                 image_memory,
                 staging_memory,
                 image_extent,
                 nullptr,
                 true,
                 owns_image_memory,
                 is_in});
          resource_index_to_io_index[i] = static_cast<int>(IOs.size() - 1);

          resource_bindings[i] = ResourceBinding{
              .descriptor_type = resource_type,
              .tensor_view = VK_NULL_HANDLE,
              .buffer = VK_NULL_HANDLE,
              .image_view = image_view,
              .sampler = sampler,
              .buffer_size = image_allocation_size,
          };
          descriptors[i] = make_data_graph_descriptor(
              resource_format,
              shape_size == 0 ? 1 : static_cast<uint32_t>(shape_size),
              shape_size == 0 ? &kScalarSentinelDimension : shape.begin(),
              static_cast<uint32_t>(stride.size()),
              stride.begin());
          descriptor_valid[i] = true;
        } else {
          ET_LOG(Error, "Unsupported descriptor type %u", resource_type);
          return false;
        }
        break;
      }
      case vgflib::ResourceCategory::CONSTANT:
        // Constants just need a descriptor; only graph segments can bind them.
        descriptors[i] = VkTensorDescriptionARM{
            .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
            .pNext = nullptr,
            .tiling = VK_TENSOR_TILING_LINEAR_ARM,
            .format = resource_format,
            .dimensionCount =
                shape_size == 0 ? 1 : static_cast<uint32_t>(shape_size),
            .pDimensions =
                shape_size == 0 ? &kScalarSentinelDimension : shape.begin(),
            // Note: stride_data of 0's causes size==0, null means stride==size
            .pStrides = (0 == stride.size() ? nullptr : stride.begin()),
            .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
        };
        descriptor_valid[i] = true;
        break;
      case vgflib::ResourceCategory::INTERMEDIATE: {
        size_t e_size = get_format_size(resource_format);
        if (0 == e_size) {
          ET_LOG(Error, "failed to get element size of VkFormat");
          return false;
        }
        if (resource_type == VK_DESCRIPTOR_TYPE_TENSOR_ARM) {
          VkTensorARM tensor = VK_NULL_HANDLE;
          VkTensorViewARM tensor_view = VK_NULL_HANDLE;
          VkTensorDescriptionARM tensor_description;
          VkMemoryRequirements2 tensor_memory_requirements = {
              .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
              .pNext = nullptr,
          };
          result = create_tensor_unbound(
              vk_device,
              resource_format,
              shape_size == 0 ? 1 : static_cast<uint32_t>(shape_size),
              shape_size == 0 ? &kScalarSentinelDimension : shape.begin(),
              static_cast<uint32_t>(stride.size()),
              stride.begin(),
              &tensor_description,
              &tensor,
              &tensor_memory_requirements);
          if (result != VK_SUCCESS) {
            ET_LOG(Error, "Failed to allocate tensor for VGF resource %d", i);
            return false;
          }
          VkDeviceMemory tensor_memory = VK_NULL_HANDLE;
          bool owns_memory = true;
          auto* alias_backing = get_alias_backing();
          if (alias_backing != nullptr) {
            if (!prepare_alias_memory(
                    tensor_memory_requirements,
                    "tensor",
                    &tensor_memory,
                    &owns_memory)) {
              destroy_tensor(vk_device, VK_NULL_HANDLE, tensor);
              return false;
            }
          } else {
            const VkMemoryPropertyFlags aims =
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            result = allocate_memory(
                vk_physical,
                vk_device,
                tensor_memory_requirements,
                aims,
                &tensor_memory);
            if (result != VK_SUCCESS) {
              destroy_tensor(vk_device, VK_NULL_HANDLE, tensor);
              ET_LOG(
                  Error,
                  "Failed to allocate tensor memory for VGF resource %d",
                  i);
              return false;
            }
          }
          result = bind_tensor_memory_and_create_view(
              vk_device, resource_format, tensor, tensor_memory, &tensor_view);
          if (result != VK_SUCCESS) {
            if (owns_memory) {
              vkFreeMemory(vk_device, tensor_memory, nullptr);
            }
            destroy_tensor(vk_device, VK_NULL_HANDLE, tensor);
            ET_LOG(Error, "Failed to bind tensor for VGF resource %d", i);
            return false;
          }

          extra_allocs.push_back(ResourceAlloc{
              .descriptor_type = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
              .tensor = tensor,
              .tensor_view = tensor_view,
              .buffer = VK_NULL_HANDLE,
              .image = VK_NULL_HANDLE,
              .image_view = VK_NULL_HANDLE,
              .sampler = VK_NULL_HANDLE,
              .image_memory = VK_NULL_HANDLE,
              .memory = tensor_memory,
              .owns_memory = owns_memory,
              .owns_image_memory = true,
          });

          resource_bindings[i] = ResourceBinding{
              .descriptor_type = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
              .tensor_view = tensor_view,
              .buffer = VK_NULL_HANDLE,
              .image_view = VK_NULL_HANDLE,
              .sampler = VK_NULL_HANDLE,
              .buffer_size = 0,
          };
          descriptors[i] = tensor_description;
          descriptor_valid[i] = true;
        } else if (resource_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
          VkDeviceSize buffer_size =
              element_count_from_shape(the_shape) * e_size;

          VkBuffer buffer = VK_NULL_HANDLE;
          VkMemoryRequirements2 buffer_memory_requirements = {
              .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
              .pNext = nullptr,
          };
          result = create_buffer_unbound(
              vk_device,
              buffer_size,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
              &buffer,
              &buffer_memory_requirements);
          if (result != VK_SUCCESS) {
            ET_LOG(Error, "Failed to allocate buffer for VGF resource %d", i);
            return false;
          }
          VkDeviceMemory buffer_memory = VK_NULL_HANDLE;
          bool owns_memory = true;
          auto* alias_backing = get_alias_backing();
          if (alias_backing != nullptr) {
            if (!prepare_alias_memory(
                    buffer_memory_requirements,
                    "buffer",
                    &buffer_memory,
                    &owns_memory)) {
              destroy_buffer(vk_device, buffer);
              return false;
            }
          } else {
            const VkMemoryPropertyFlags aims =
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            result = allocate_memory(
                vk_physical,
                vk_device,
                buffer_memory_requirements,
                aims,
                &buffer_memory);
            if (result != VK_SUCCESS) {
              destroy_buffer(vk_device, buffer);
              ET_LOG(
                  Error,
                  "Failed to allocate buffer memory for VGF resource %d",
                  i);
              return false;
            }
          }
          result = bind_buffer_memory(vk_device, buffer, buffer_memory);
          if (result != VK_SUCCESS) {
            if (owns_memory) {
              vkFreeMemory(vk_device, buffer_memory, nullptr);
            }
            destroy_buffer(vk_device, buffer);
            ET_LOG(
                Error, "Failed to bind buffer memory for VGF resource %d", i);
            return false;
          }

          extra_allocs.push_back(ResourceAlloc{
              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .tensor = VK_NULL_HANDLE,
              .tensor_view = VK_NULL_HANDLE,
              .buffer = buffer,
              .image = VK_NULL_HANDLE,
              .image_view = VK_NULL_HANDLE,
              .sampler = VK_NULL_HANDLE,
              .image_memory = VK_NULL_HANDLE,
              .memory = buffer_memory,
              .owns_memory = owns_memory,
              .owns_image_memory = true,
          });

          resource_bindings[i] = ResourceBinding{
              .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .tensor_view = VK_NULL_HANDLE,
              .buffer = buffer,
              .image_view = VK_NULL_HANDLE,
              .sampler = VK_NULL_HANDLE,
              .buffer_size = buffer_size,
          };
        } else if (
            resource_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
            resource_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
            resource_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
          VkExtent3D image_extent = {};
          if (!validate_image_shape_and_format(
                  the_shape, resource_format, &image_extent)) {
            return false;
          }
          const VkImageUsageFlags image_usage =
              VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
              VK_IMAGE_USAGE_TRANSFER_DST_BIT |
              ((uses_alias_group &&
                alias_group_usage[*alias_group].has_tensor_like)
                   ? VK_IMAGE_USAGE_TENSOR_ALIASING_BIT_ARM
                   : 0) |
              ((resource_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                   ? VK_IMAGE_USAGE_STORAGE_BIT
                   : VK_IMAGE_USAGE_SAMPLED_BIT);
          VkImage image = VK_NULL_HANDLE;
          VkImageView image_view = VK_NULL_HANDLE;
          VkMemoryRequirements2 image_memory_requirements = {
              .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
              .pNext = nullptr,
          };
          result = create_image_unbound(
              vk_device,
              resource_format,
              image_extent,
              image_usage,
              &image,
              &image_memory_requirements);
          if (result != VK_SUCCESS) {
            ET_LOG(Error, "Failed to allocate image for VGF resource %d", i);
            return false;
          }
          VkDeviceMemory image_memory = VK_NULL_HANDLE;
          bool owns_image_memory = true;
          auto* alias_backing = get_alias_backing();
          if (alias_backing != nullptr) {
            if (!prepare_alias_memory(
                    image_memory_requirements,
                    "image",
                    &image_memory,
                    &owns_image_memory)) {
              free_image(
                  vk_device,
                  VK_NULL_HANDLE,
                  image,
                  VK_NULL_HANDLE,
                  VK_NULL_HANDLE);
              return false;
            }
          } else {
            const VkMemoryPropertyFlags aims =
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            result = allocate_memory(
                vk_physical,
                vk_device,
                image_memory_requirements,
                aims,
                &image_memory);
            if (result != VK_SUCCESS) {
              free_image(
                  vk_device,
                  VK_NULL_HANDLE,
                  image,
                  VK_NULL_HANDLE,
                  VK_NULL_HANDLE);
              ET_LOG(
                  Error,
                  "Failed to allocate image memory for VGF resource %d",
                  i);
              return false;
            }
          }
          result = bind_image_memory_and_create_view(
              vk_device, resource_format, image, image_memory, &image_view);
          if (result != VK_SUCCESS) {
            free_image(
                vk_device,
                VK_NULL_HANDLE,
                image,
                VK_NULL_HANDLE,
                owns_image_memory ? image_memory : VK_NULL_HANDLE);
            ET_LOG(Error, "Failed to bind image for VGF resource %d", i);
            return false;
          }
          const bool needs_tensor_aliasing = uses_alias_group &&
              alias_group_usage[*alias_group].has_tensor_like;
          const VkImageLayout initial_layout = needs_tensor_aliasing
              ? VK_IMAGE_LAYOUT_TENSOR_ALIASING_ARM
              : VK_IMAGE_LAYOUT_GENERAL;
          result = transition_image_layout(
              vk_device,
              vk_command_pool,
              vk_queue,
              image,
              VK_IMAGE_LAYOUT_UNDEFINED,
              initial_layout);
          if (result != VK_SUCCESS) {
            ET_LOG(Error, "Failed to transition image for VGF resource %d", i);
            free_image(
                vk_device,
                image_view,
                image,
                VK_NULL_HANDLE,
                owns_image_memory ? image_memory : VK_NULL_HANDLE);
            return false;
          }

          VkSampler sampler = VK_NULL_HANDLE;
          if (resource_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            if (!allocate_resource_sampler(
                    resource_decoder, i, vk_device, &sampler)) {
              free_image(
                  vk_device,
                  image_view,
                  image,
                  VK_NULL_HANDLE,
                  owns_image_memory ? image_memory : VK_NULL_HANDLE);
              return false;
            }
          }
          if (uses_alias_group) {
            auto& alias_state = alias_image_states[*alias_group];
            alias_state.needs_tensor_aliasing = needs_tensor_aliasing;
            alias_state.current_layout = initial_layout;
            alias_state.images.push_back(image);
          }

          extra_allocs.push_back(ResourceAlloc{
              .descriptor_type = resource_type,
              .tensor = VK_NULL_HANDLE,
              .tensor_view = VK_NULL_HANDLE,
              .buffer = VK_NULL_HANDLE,
              .image = image,
              .image_view = image_view,
              .sampler = sampler,
              .image_memory = image_memory,
              .memory = VK_NULL_HANDLE,
              .owns_memory = true,
              .owns_image_memory = owns_image_memory,
          });

          resource_bindings[i] = ResourceBinding{
              .descriptor_type = resource_type,
              .tensor_view = VK_NULL_HANDLE,
              .buffer = VK_NULL_HANDLE,
              .image_view = image_view,
              .sampler = sampler,
              .buffer_size = 0,
          };
          descriptors[i] = make_data_graph_descriptor(
              resource_format,
              shape_size == 0 ? 1 : static_cast<uint32_t>(shape_size),
              shape_size == 0 ? &kScalarSentinelDimension : shape.begin(),
              static_cast<uint32_t>(stride.size()),
              stride.begin());
          descriptor_valid[i] = true;
        } else {
          ET_LOG(Error, "Unsupported descriptor type %u", resource_type);
          return false;
        }
      } break;
      default:
        ET_LOG(Info, "Unsupported resource category UNKNOWN");
        return false;
    }
  }

  // Build per-segment pipelines and descriptor sets.
  segments.clear();
  segments.reserve(segment_count);
  for (int segment_id = 0; segment_id < segment_count; ++segment_id) {
    const auto segment_type = sequence_decoder->getSegmentType(segment_id);
    if (segment_type != vgflib::ModuleType::GRAPH &&
        segment_type != vgflib::ModuleType::COMPUTE) {
      ET_LOG(Error, "Unsupported segment type");
      return false;
    }

    SegmentState segment;
    segment.segment_id = segment_id;
    segment.use_data_graph_pipeline =
        (segment_type == vgflib::ModuleType::GRAPH);
    auto dispatch_shape = sequence_decoder->getSegmentDispatchShape(segment_id);
    segment.dispatch_shape = {
        dispatch_shape[0], dispatch_shape[1], dispatch_shape[2]};

    auto segment_name = string(sequence_decoder->getSegmentName(segment_id));
    auto segment_module = sequence_decoder->getSegmentModuleIndex(segment_id);
    ET_LOG(
        Info,
        "VGF segment '%s' module=%u type=%s dispatch=[%u,%u,%u]",
        segment_name.c_str(),
        segment_module,
        segment.use_data_graph_pipeline ? "GRAPH" : "COMPUTE",
        dispatch_shape[0],
        dispatch_shape[1],
        dispatch_shape[2]);

    auto segment_m_name = string(module_decoder->getModuleName(segment_module));
    auto segment_m_entrypoint =
        string(module_decoder->getModuleEntryPoint(segment_module));
    ET_LOG(
        Info,
        "VGF module '%s' entrypoint='%s' type=%s has_spirv=%d",
        segment_m_name.c_str(),
        segment_m_entrypoint.c_str(),
        (module_decoder->getModuleType(segment_module) ==
                 vgflib::ModuleType::GRAPH
             ? "GRAPH"
             : "COMPUTE"),
        module_decoder->hasSPIRV(segment_module));
    if (!module_decoder->hasSPIRV(segment_module)) {
      ET_LOG(Error, "Module %d does not contain SPIR-V code", segment_module);
      return false;
    }
    auto segment_m_spirv =
        get_module_spirv_code(module_decoder, segment_module);
    ET_LOG(Info, "SPIR-V code size (words) %zu", segment_m_spirv.size());

    VkShaderModuleCreateInfo smci{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = segment_m_spirv.size() * sizeof(uint32_t),
        .pCode = segment_m_spirv.begin(),
    };
    result =
        vkCreateShaderModule(vk_device, &smci, nullptr, &segment.vk_shader);
    if (result != VK_SUCCESS) {
      ET_LOG(Error, "Failed to load shader from segment %d", segment_module);
      return false;
    }

    // Constants table (graph segments only)
    vector<VkDataGraphPipelineConstantARM> constants;
    auto constant_indexes =
        sequence_decoder->getSegmentConstantIndexes(segment_id);
    if (!segment.use_data_graph_pipeline && !constant_indexes.empty()) {
      ET_LOG(Error, "Constants are not supported with compute segments");
      return false;
    }
    if (segment.use_data_graph_pipeline) {
      for (uint32_t i : constant_indexes) {
        auto mrt_i = constant_decoder->getConstantMrtIndex(i);
        if (!descriptor_valid[mrt_i]) {
          ET_LOG(Error, "Missing descriptor for constant MRT index %u", mrt_i);
          return false;
        }
        auto constant_data = constant_decoder->getConstant(i);
        constants.push_back(VkDataGraphPipelineConstantARM{
            .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM,
            .pNext = &descriptors[mrt_i],
            .id = i,
            .pConstantData = constant_data.begin(),
        });
      }
    }

    // Prepare layout bindings from this segment's information
    vector<VkDescriptorSetLayoutBinding> layout_bindings;
    vector<VkDataGraphPipelineResourceInfoARM> data_graph_resources;
    auto set_count =
        sequence_decoder->getSegmentDescriptorSetInfosSize(segment_id);
    if (set_count != 1) {
      ET_LOG(
          Error,
          "Only a single descriptor set is currently supported, got %zu for segment %d",
          set_count,
          segment_id);
      return false;
    }
    for (uint32_t d_idx = 0; d_idx < set_count; d_idx++) {
      auto handle =
          sequence_decoder->getDescriptorBindingSlotsHandle(segment_id, d_idx);
      auto binding_count = sequence_decoder->getBindingsSize(handle);
      for (int binding = 0; binding < binding_count; binding++) {
        auto binding_index =
            sequence_decoder->getBindingSlotBinding(handle, binding);
        auto MRT_index =
            sequence_decoder->getBindingSlotMrtIndex(handle, binding);
        auto MRT_type = resolve_descriptor_type(resource_decoder, MRT_index);

        if (segment.use_data_graph_pipeline &&
            MRT_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
          ET_LOG(
              Error, "Storage buffers are not supported with graph segments");
          return false;
        }

        const VkDescriptorSetLayoutBinding layout_binding{
            .binding = binding_index,
            .descriptorType = MRT_type,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_ALL,
            .pImmutableSamplers = nullptr,
        };
        layout_bindings.push_back(layout_binding);

        if (segment.use_data_graph_pipeline) {
          if (!descriptor_valid[MRT_index]) {
            ET_LOG(Error, "Missing descriptor for MRT index %u", MRT_index);
            return false;
          }
          const VkDataGraphPipelineResourceInfoARM resource{
              .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
              .pNext = &descriptors[MRT_index],
              .descriptorSet = d_idx,
              .binding = binding_index,
              .arrayElement = 0,
          };
          data_graph_resources.push_back(resource);
        }
      }
    }

    const VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = static_cast<uint32_t>(layout_bindings.size()),
        .pBindings = layout_bindings.data(),
    };
    result = vkCreateDescriptorSetLayout(
        vk_device, &layout_info, nullptr, &segment.vk_layout);
    if (result != VK_SUCCESS) {
      ET_LOG(Error, "Failed to create descriptor layout");
      return false;
    }

    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.reserve(layout_bindings.size());
    for (const auto& b : layout_bindings) {
      bool found = false;
      for (size_t idx = 0; idx < poolSizes.size(); ++idx) {
        if (poolSizes[idx].type == b.descriptorType) {
          poolSizes[idx].descriptorCount += b.descriptorCount;
          found = true;
          break;
        }
      }
      if (!found) {
        poolSizes.push_back({b.descriptorType, b.descriptorCount});
      }
    }

    const VkDescriptorPoolCreateInfo descriptor_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = static_cast<uint32_t>(set_count),
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    result = vkCreateDescriptorPool(
        vk_device, &descriptor_pool_info, nullptr, &segment.vk_descriptor_pool);
    if (result != VK_SUCCESS) {
      ET_LOG(Error, "Failed to create descriptor pool");
      return false;
    }

    const VkDescriptorSetAllocateInfo descriptor_set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = segment.vk_descriptor_pool,
        .descriptorSetCount = static_cast<uint32_t>(set_count),
        .pSetLayouts = &segment.vk_layout,
    };

    segment.descriptor_sets.resize(set_count);
    result = vkAllocateDescriptorSets(
        vk_device, &descriptor_set_info, segment.descriptor_sets.data());
    if (result != VK_SUCCESS) {
      ET_LOG(Error, "Failed to allocate descriptor sets");
      return false;
    }

    for (uint32_t d_idx = 0; d_idx < set_count; d_idx++) {
      const auto set_index =
          get_segment_descriptor_set_index(sequence_decoder, segment_id, d_idx);
      if (set_index != d_idx) {
        ET_LOG(
            Error,
            "Explicit descriptor set index %u is not supported for segment %d descriptor %u",
            set_index,
            segment_id,
            d_idx);
        return false;
      }

      auto descriptor_slots =
          sequence_decoder->getDescriptorBindingSlotsHandle(segment_id, d_idx);
      auto descriptor_count =
          sequence_decoder->getBindingsSize(descriptor_slots);
      ET_LOG(
          Info, "VGF descriptor set %u bindings: %zu", d_idx, descriptor_count);
      for (uint32_t i = 0; i < descriptor_count; i++) {
        auto binding =
            sequence_decoder->getBindingSlotBinding(descriptor_slots, i);
        auto mrt_i =
            sequence_decoder->getBindingSlotMrtIndex(descriptor_slots, i);
        const auto& binding_info = resource_bindings[mrt_i];
        if (binding_info.descriptor_type == VK_DESCRIPTOR_TYPE_TENSOR_ARM) {
          ET_LOG(
              Info,
              "Updating descriptor: segment=%u set=%u binding=%u mrt=%u type=VK_DESCRIPTOR_TYPE_TENSOR_ARM",
              segment_id,
              d_idx,
              binding,
              mrt_i);
          VkWriteDescriptorSetTensorARM write_desc = {
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM,
              .pNext = nullptr,
              .tensorViewCount = 1,
              .pTensorViews = &binding_info.tensor_view,
          };
          VkWriteDescriptorSet desc_set = {
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .pNext = &write_desc,
              .dstSet = segment.descriptor_sets[d_idx],
              .dstBinding = binding,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
              .pImageInfo = nullptr,
              .pBufferInfo = nullptr,
              .pTexelBufferView = nullptr,
          };
          vkUpdateDescriptorSets(vk_device, 1, &desc_set, 0, nullptr);
        } else if (
            binding_info.descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
          ET_LOG(
              Info,
              "Updating descriptor: segment=%u set=%u binding=%u mrt=%u type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
              segment_id,
              d_idx,
              binding,
              mrt_i);
          VkDescriptorBufferInfo buffer_info = {
              .buffer = binding_info.buffer,
              .offset = 0,
              .range = binding_info.buffer_size,
          };
          VkWriteDescriptorSet desc_set = {
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .pNext = nullptr,
              .dstSet = segment.descriptor_sets[d_idx],
              .dstBinding = binding,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .pImageInfo = nullptr,
              .pBufferInfo = &buffer_info,
              .pTexelBufferView = nullptr,
          };
          vkUpdateDescriptorSets(vk_device, 1, &desc_set, 0, nullptr);
        } else if (
            binding_info.descriptor_type ==
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
            binding_info.descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
            binding_info.descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
          const char* type_name = binding_info.descriptor_type ==
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
              ? "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
              : (binding_info.descriptor_type ==
                         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                     ? "VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE"
                     : "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE");
          ET_LOG(
              Info,
              "Updating descriptor: segment=%u set=%u binding=%u mrt=%u type=%s image_view=%p sampler=%p",
              segment_id,
              d_idx,
              binding,
              mrt_i,
              type_name,
              log_handle_ptr(binding_info.image_view),
              log_handle_ptr(binding_info.sampler));
          VkDescriptorImageInfo image_info = {
              .sampler = binding_info.sampler,
              .imageView = binding_info.image_view,
              .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
          };
          VkWriteDescriptorSet desc_set = {
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .pNext = nullptr,
              .dstSet = segment.descriptor_sets[d_idx],
              .dstBinding = binding,
              .dstArrayElement = 0,
              .descriptorCount = 1,
              .descriptorType = binding_info.descriptor_type,
              .pImageInfo = &image_info,
              .pBufferInfo = nullptr,
              .pTexelBufferView = nullptr,
          };
          vkUpdateDescriptorSets(vk_device, 1, &desc_set, 0, nullptr);
        } else {
          ET_LOG(
              Error,
              "Unsupported descriptor type %u for descriptor binding",
              binding_info.descriptor_type);
          return false;
        }
      }
    }

    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &segment.vk_layout,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };
    result = vkCreatePipelineLayout(
        vk_device, &pipeline_layout_info, nullptr, &segment.vk_pipeline_layout);
    if (result != VK_SUCCESS) {
      ET_LOG(Error, "Failed to create pipeline layout");
      return false;
    }

    if (segment.use_data_graph_pipeline) {
      VkDataGraphPipelineShaderModuleCreateInfoARM shader_info{
          .sType =
              VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM,
          .pNext = nullptr,
          .module = segment.vk_shader,
          .pName = segment_m_entrypoint.c_str(),
          .pSpecializationInfo = nullptr,
          .constantCount = static_cast<uint32_t>(constants.size()),
          .pConstants = constants.data(),
      };

      VkDataGraphPipelineCreateInfoARM graph_pipeline_info{
          .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM,
          .pNext = &shader_info,
          .flags = VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT |
              VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR,
          .layout = segment.vk_pipeline_layout,
          .resourceInfoCount =
              static_cast<uint32_t>(data_graph_resources.size()),
          .pResourceInfos = data_graph_resources.data(),
      };

      result = vkCreateDataGraphPipelinesARM(
          vk_device,
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          1,
          &graph_pipeline_info,
          nullptr,
          &segment.vk_pipeline);
      if (result != VK_SUCCESS) {
        ET_LOG(Error, "Failed to create DataGraphPipeline");
        return false;
      }

      VkDataGraphPipelineSessionCreateInfoARM pipeline_session_info{
          .sType =
              VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
          .pNext = nullptr,
          .flags = 0,
          .dataGraphPipeline = segment.vk_pipeline,
      };
      result = vkCreateDataGraphPipelineSessionARM(
          vk_device, &pipeline_session_info, nullptr, &segment.vk_session);
      if (result != VK_SUCCESS) {
        ET_LOG(Error, "Failed to create DataGraphPipelineSession");
        return false;
      }

      VkDataGraphPipelineSessionBindPointRequirementsInfoARM
          bind_point_requirements_info = {
              .sType =
                  VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM,
              .pNext = nullptr,
              .session = segment.vk_session,
          };

      uint32_t bind_point_count = 0;
      result = vkGetDataGraphPipelineSessionBindPointRequirementsARM(
          vk_device, &bind_point_requirements_info, &bind_point_count, nullptr);
      if (result != VK_SUCCESS) {
        ET_LOG(Error, "Failed to get session bind point count");
        return false;
      }

      vector<VkDataGraphPipelineSessionBindPointRequirementARM>
          bind_point_requirements;
      bind_point_requirements.resize(bind_point_count);
      result = vkGetDataGraphPipelineSessionBindPointRequirementsARM(
          vk_device,
          &bind_point_requirements_info,
          &bind_point_count,
          bind_point_requirements.data());
      if (result != VK_SUCCESS) {
        ET_LOG(Error, "Failed to get session bind point requirements");
        return false;
      }

      for (const auto& bind_point_requirement : bind_point_requirements) {
        if (bind_point_requirement.bindPointType !=
            VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM) {
          ET_LOG(
              Error,
              "Expected VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM");
          return false;
        }
        if (bind_point_requirement.bindPoint !=
            VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM) {
          ET_LOG(
              Error,
              "Expected VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM");
          return false;
        }
        if (bind_point_requirement.numObjects != 1) {
          ET_LOG(Error, "Expected only one object for the bindpoint");
          return false;
        }

        VkDataGraphPipelineSessionMemoryRequirementsInfoARM
            memory_requirements_info = {
                .sType =
                    VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM,
                .pNext = nullptr,
                .session = segment.vk_session,
                .bindPoint = bind_point_requirement.bindPoint,
                .objectIndex = 0,
            };
        VkMemoryRequirements2 memory_requirements = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
            .pNext = nullptr,
        };
        vkGetDataGraphPipelineSessionMemoryRequirementsARM(
            vk_device, &memory_requirements_info, &memory_requirements);

        VkMemoryPropertyFlags aims = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32_t memory_index = 0;
        if (!find_memory_index(
                vk_physical, memory_requirements, aims, &memory_index)) {
          ET_LOG(
              Error,
              "Failed to find data-graph session memory type for segment %d",
              segment.segment_id);
          return false;
        }

        VkMemoryAllocateInfo memory_allocate_info = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = memory_requirements.memoryRequirements.size,
            .memoryTypeIndex = memory_index,
        };

        VkDeviceMemory memory;
        result = vkAllocateMemory(
            vk_device, &memory_allocate_info, nullptr, &memory);
        if (result != VK_SUCCESS) {
          ET_LOG(Error, "Failed to allocate memory for intermediates");
          return false;
        }
        intermediates.push_back(memory);

        VkBindDataGraphPipelineSessionMemoryInfoARM bind_info = {
            .sType =
                VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM,
            .pNext = nullptr,
            .session = segment.vk_session,
            .bindPoint = bind_point_requirement.bindPoint,
            .objectIndex = 0,
            .memory = memory,
            .memoryOffset = 0,
        };
        result =
            vkBindDataGraphPipelineSessionMemoryARM(vk_device, 1, &bind_info);
        if (result != VK_SUCCESS) {
          ET_LOG(Error, "Failed to bind intermediates memory");
          return false;
        }
      }
    } else {
      VkPipelineShaderStageCreateInfo stage_info{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .pNext = nullptr,
          .flags = 0,
          .stage = VK_SHADER_STAGE_COMPUTE_BIT,
          .module = segment.vk_shader,
          .pName = segment_m_entrypoint.c_str(),
          .pSpecializationInfo = nullptr,
      };
      VkComputePipelineCreateInfo compute_info{
          .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
          .pNext = nullptr,
          .flags = 0,
          .stage = stage_info,
          .layout = segment.vk_pipeline_layout,
          .basePipelineHandle = VK_NULL_HANDLE,
          .basePipelineIndex = -1,
      };
      result = vkCreateComputePipelines(
          vk_device,
          VK_NULL_HANDLE,
          1,
          &compute_info,
          nullptr,
          &segment.vk_pipeline);
      if (result != VK_SUCCESS) {
        ET_LOG(Error, "Failed to create compute pipeline");
        return false;
      }
    }

    segments.push_back(std::move(segment));
  }

  // Map model sequence inputs/outputs to IO indices
  auto input_handle =
      sequence_decoder->getModelSequenceInputBindingSlotsHandle();
  auto output_handle =
      sequence_decoder->getModelSequenceOutputBindingSlotsHandle();
  auto input_names_handle =
      sequence_decoder->getModelSequenceInputNamesHandle();
  auto output_names_handle =
      sequence_decoder->getModelSequenceOutputNamesHandle();
  const size_t model_input_count =
      sequence_decoder->getNamesSize(input_names_handle);
  const size_t model_output_count =
      sequence_decoder->getNamesSize(output_names_handle);
  this->model_input_count = model_input_count;
  this->model_output_count = model_output_count;
  model_input_io_index.assign(model_input_count, -1);
  model_output_io_index.assign(model_output_count, -1);

  const size_t input_binding_count =
      sequence_decoder->getBindingsSize(input_handle);
  const size_t output_binding_count =
      sequence_decoder->getBindingsSize(output_handle);
  for (size_t i = 0; i < input_binding_count && i < model_input_count; ++i) {
    auto mrt_i = sequence_decoder->getBindingSlotMrtIndex(input_handle, i);
    if (mrt_i < resource_index_to_io_index.size()) {
      model_input_io_index[i] = resource_index_to_io_index[mrt_i];
    }
  }
  for (size_t i = 0; i < output_binding_count && i < model_output_count; ++i) {
    auto mrt_i = sequence_decoder->getBindingSlotMrtIndex(output_handle, i);
    if (mrt_i < resource_index_to_io_index.size()) {
      model_output_io_index[i] = resource_index_to_io_index[mrt_i];
    }
  }
  ET_LOG(
      Info,
      "Model IO mapping: inputs=%zu outputs=%zu (bindings in=%zu out=%zu)",
      model_input_count,
      model_output_count,
      input_binding_count,
      output_binding_count);
  for (size_t i = 0; i < model_input_count; ++i) {
    ET_LOG(Info, "  input[%zu] -> IO[%d]", i, model_input_io_index[i]);
  }
  for (size_t i = 0; i < model_output_count; ++i) {
    ET_LOG(Info, "  output[%zu] -> IO[%d]", i, model_output_io_index[i]);
  }

  {
    VGF_PROFILE_SCOPE(event_tracer, "VGF_INIT_ALLOCATE_COMMAND_BUFFER");

    VkCommandBufferAllocateInfo buffer_allocate_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = vk_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};

    result = vkAllocateCommandBuffers(
        vk_device, &buffer_allocate_info, &vk_execute_cmd);
    if (result != VK_SUCCESS) {
      ET_LOG(Error, "Failed to allocate command buffers");
      return false;
    }

    const VkFenceCreateInfo fence_info{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    result = vkCreateFence(vk_device, &fence_info, nullptr, &vk_execute_fence);
    if (result != VK_SUCCESS) {
      ET_LOG(Error, "Failed to create VGF execute fence, error %d", result);
      vkFreeCommandBuffers(vk_device, vk_command_pool, 1, &vk_execute_cmd);
      vk_execute_cmd = VK_NULL_HANDLE;
      return false;
    }
  }

  {
    VGF_PROFILE_SCOPE(event_tracer, "VGF_INIT_TIMESTAMP_QUERIES");

    if (!init_timestamp_queries()) {
      ET_LOG(Error, "Failed to initialize VGF timestamp queries");
      return false;
    }
  }

  {
    VGF_PROFILE_SCOPE(event_tracer, "VGF_INIT_RECORD_COMMAND_BUFFER");

    // Populate command once with our dispatch information
    VkCommandBufferBeginInfo beginInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(vk_execute_cmd, &beginInfo);

    // Sync what will be the data coming in from host
    VkMemoryBarrier2 barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
        .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
        .dstStageMask =
            VK_PIPELINE_STAGE_2_TRANSFER_BIT | vgf_execution_stage_mask(),
        .dstAccessMask =
            VK_ACCESS_2_TRANSFER_READ_BIT | vgf_execution_read_access_mask(),
    };
    VkDependencyInfo dependency_info = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(vk_execute_cmd, &dependency_info);

    bool has_input_image = false;
    for (const auto& io : IOs) {
      if (io.is_input &&
          (io.descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
           io.descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
           io.descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)) {
        has_input_image = true;
        const VkBufferImageCopy copy_region = {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .imageOffset = {0, 0, 0},
            .imageExtent = io.image_extent,
        };
        vkCmdCopyBufferToImage(
            vk_execute_cmd,
            io.buffer,
            io.image,
            VK_IMAGE_LAYOUT_GENERAL,
            1,
            &copy_region);
      }
    }

    if (has_input_image) {
      VkMemoryBarrier2 input_image_barrier = {
          .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
          .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
          .dstStageMask = vgf_execution_stage_mask(),
          .dstAccessMask = vgf_execution_read_access_mask() |
              vgf_execution_write_access_mask(),
      };
      VkDependencyInfo input_image_dependency = {
          .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
          .memoryBarrierCount = 1,
          .pMemoryBarriers = &input_image_barrier,
      };
      vkCmdPipelineBarrier2(vk_execute_cmd, &input_image_dependency);
    }

    if (timestamp_queries_enabled &&
        vk_timestamp_query_pool != VK_NULL_HANDLE) {
      vkCmdResetQueryPool(vk_execute_cmd, vk_timestamp_query_pool, 0, 2);

      if (vkCmdWriteTimestamp2) {
        vkCmdWriteTimestamp2(
            vk_execute_cmd,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            vk_timestamp_query_pool,
            0);
      } else {
        vkCmdWriteTimestamp(
            vk_execute_cmd,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            vk_timestamp_query_pool,
            0);
      }
    }

    // Bind and dispatch each segment in order.
    for (size_t seg_idx = 0; seg_idx < segments.size(); ++seg_idx) {
      const auto& segment = segments[seg_idx];
      unordered_map<uint32_t, VkImageLayout> desired_alias_layouts;
      auto set_count = sequence_decoder->getSegmentDescriptorSetInfosSize(
          segment.segment_id);
      for (uint32_t d_idx = 0; d_idx < set_count; d_idx++) {
        auto descriptor_slots =
            sequence_decoder->getDescriptorBindingSlotsHandle(
                segment.segment_id, d_idx);
        auto descriptor_count =
            sequence_decoder->getBindingsSize(descriptor_slots);
        for (uint32_t i = 0; i < descriptor_count; i++) {
          auto mrt_i =
              sequence_decoder->getBindingSlotMrtIndex(descriptor_slots, i);
          auto alias_group =
              get_resource_alias_group_id(resource_decoder, mrt_i);
          if (!alias_group.has_value()) {
            continue;
          }
          auto alias_state_it = alias_image_states.find(*alias_group);
          if (alias_state_it == alias_image_states.end() ||
              !alias_state_it->second.needs_tensor_aliasing) {
            continue;
          }
          const auto descriptor_type = resource_bindings[mrt_i].descriptor_type;
          const auto desired_layout = is_image_descriptor_type(descriptor_type)
              ? VK_IMAGE_LAYOUT_GENERAL
              : VK_IMAGE_LAYOUT_TENSOR_ALIASING_ARM;
          auto desired_it = desired_alias_layouts.find(*alias_group);
          if (desired_it == desired_alias_layouts.end()) {
            desired_alias_layouts[*alias_group] = desired_layout;
          } else if (desired_it->second != desired_layout) {
            ET_LOG(
                Error,
                "Alias group %u mixes image and tensor-like descriptor use in segment %d",
                *alias_group,
                segment.segment_id);
            return false;
          }
        }
      }
      for (auto& [alias_group, desired_layout] : desired_alias_layouts) {
        auto& alias_state = alias_image_states[alias_group];
        if (alias_state.current_layout == desired_layout) {
          continue;
        }
        for (auto image : alias_state.images) {
          record_image_layout_transition(
              vk_execute_cmd,
              image,
              alias_state.current_layout,
              desired_layout);
        }
        alias_state.current_layout = desired_layout;
      }

      VkPipelineBindPoint bind_point = segment.use_data_graph_pipeline
          ? VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM
          : VK_PIPELINE_BIND_POINT_COMPUTE;
      vkCmdBindPipeline(vk_execute_cmd, bind_point, segment.vk_pipeline);

      vkCmdBindDescriptorSets(
          vk_execute_cmd,
          bind_point,
          segment.vk_pipeline_layout,
          0, // first set
          1,
          segment.descriptor_sets.data(),
          0,
          nullptr);

      if (segment.use_data_graph_pipeline) {
        vkCmdDispatchDataGraphARM(vk_execute_cmd, segment.vk_session, nullptr);
      } else {
        vkCmdDispatch(
            vk_execute_cmd,
            segment.dispatch_shape[0],
            segment.dispatch_shape[1],
            segment.dispatch_shape[2]);
      }

      if (seg_idx + 1 < segments.size()) {
        VkMemoryBarrier2 segment_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = vgf_execution_stage_mask(),
            .srcAccessMask = vgf_execution_write_access_mask(),
            .dstStageMask = vgf_execution_stage_mask(),
            .dstAccessMask = vgf_execution_read_access_mask() |
                vgf_execution_write_access_mask(),
        };
        VkDependencyInfo segment_dep = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &segment_barrier,
        };
        vkCmdPipelineBarrier2(vk_execute_cmd, &segment_dep);
      }
    }

    if (timestamp_queries_enabled &&
        vk_timestamp_query_pool != VK_NULL_HANDLE) {
      if (vkCmdWriteTimestamp2) {
        vkCmdWriteTimestamp2(
            vk_execute_cmd,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            vk_timestamp_query_pool,
            1);
      } else {
        vkCmdWriteTimestamp(
            vk_execute_cmd,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            vk_timestamp_query_pool,
            1);
      }
    }

    // Sync data back
    const bool has_output_image =
        std::any_of(IOs.begin(), IOs.end(), [](const auto& io) {
          return !io.is_input &&
              (io.descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
               io.descriptor_type ==
                   VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
               io.descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
        });

    if (has_output_image) {
      VkMemoryBarrier2 output_image_barrier = {
          .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
          .srcStageMask = vgf_execution_stage_mask(),
          .srcAccessMask = vgf_execution_write_access_mask(),
          .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
      };
      VkDependencyInfo output_image_dependency = {
          .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
          .memoryBarrierCount = 1,
          .pMemoryBarriers = &output_image_barrier,
      };
      vkCmdPipelineBarrier2(vk_execute_cmd, &output_image_dependency);

      for (const auto& io : IOs) {
        if (!io.is_input &&
            (io.descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
             io.descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
             io.descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)) {
          const VkBufferImageCopy copy_region = {
              .bufferOffset = 0,
              .bufferRowLength = 0,
              .bufferImageHeight = 0,
              .imageSubresource =
                  {
                      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                      .mipLevel = 0,
                      .baseArrayLayer = 0,
                      .layerCount = 1,
                  },
              .imageOffset = {0, 0, 0},
              .imageExtent = io.image_extent,
          };
          vkCmdCopyImageToBuffer(
              vk_execute_cmd,
              io.image,
              VK_IMAGE_LAYOUT_GENERAL,
              io.buffer,
              1,
              &copy_region);
        }
      }
    }

    VkMemoryBarrier2 barrier_2 = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask =
            VK_PIPELINE_STAGE_2_TRANSFER_BIT | vgf_execution_stage_mask(),
        .srcAccessMask =
            VK_ACCESS_2_TRANSFER_WRITE_BIT | vgf_execution_write_access_mask(),
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
        .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
    };
    VkDependencyInfo dependency_info_2 = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier_2,
    };
    vkCmdPipelineBarrier2(vk_execute_cmd, &dependency_info_2);

    // end the command buffer
    vkEndCommandBuffer(vk_execute_cmd);
  }

  {
    VGF_PROFILE_SCOPE(event_tracer, "VGF_INIT_MAP_IO_MEMORY");

    if (!map_persis

            tent_io_memory()) {
      ET_LOG(Error, "Failed to persistently map VGF IO memory");
      return false;
    }
  }

  return true;
}

bool VgfRepr::execute_vgf(executorch::runtime::EventTracer* event_tracer) {
  ET_LOG(Info, "Executing vgf");

  VkSubmitInfo submit{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = nullptr,
      .waitSemaphoreCount = 0,
      .pWaitSemaphores = nullptr,
      .pWaitDstStageMask = nullptr,
      .commandBufferCount = 1,
      .pCommandBuffers = &vk_execute_cmd,
      .signalSemaphoreCount = 0,
      .pSignalSemaphores = nullptr,
  };

  VkResult result;

  {
    VGF_PROFILE_SCOPE(event_tracer, "VGF_QUEUE_SUBMIT_AND_WAIT_FENCE");

    if (vk_execute_fence == VK_NULL_HANDLE) {
      ET_LOG(Error, "VGF execute fence is not initialized");
      return false;
    }

    result = vkResetFences(vk_device, 1, &vk_execute_fence);
    if (result != VK_SUCCESS) {
      ET_LOG(Error, "VGF/VkFence reset failed, error %d", result);
      return false;
    }

    result = vkQueueSubmit(vk_queue, 1, &submit, vk_execute_fence);
    if (result != VK_SUCCESS) {
      ET_LOG(Error, "VGF/VkFence wait failed, error %d", result);
      return false;
    }

    result = vkWaitForFences(
        vk_device,
        1,
        &vk_execute_fence,
        VK_TRUE,
        std::numeric_limits<uint64_t>::max());
  }

  if (result != VK_SUCCESS) {
    ET_LOG(
        Error, "VGF/VkCommandBuffer command submission or fence wait failed");
    return false;
  }

  read_timestamp_queries(event_tracer);

  return true;
}

void VgfRepr::free_vgf() {
  unmap_persistent_io_memory();

  if (vk_timestamp_query_pool != VK_NULL_HANDLE) {
    vkDestroyQueryPool(vk_device, vk_timestamp_query_pool, nullptr);
    vk_timestamp_query_pool = VK_NULL_HANDLE;
  }

  if (vk_execute_fence != VK_NULL_HANDLE) {
    vkDestroyFence(vk_device, vk_execute_fence, nullptr);
    vk_execute_fence = VK_NULL_HANDLE;
  }

  vkFreeCommandBuffers(vk_device, vk_command_pool, 1, &vk_execute_cmd);
  vector<VkDeviceMemory> owned_memory;
  auto remember_owned_memory = [&](VkDeviceMemory memory) {
    if (memory == VK_NULL_HANDLE) {
      return;
    }
    if (find(owned_memory.begin(), owned_memory.end(), memory) ==
        owned_memory.end()) {
      owned_memory.push_back(memory);
    }
  };
  for (auto& segment : segments) {
    if (segment.use_data_graph_pipeline &&
        segment.vk_session != VK_NULL_HANDLE) {
      vkDestroyDataGraphPipelineSessionARM(
          vk_device, segment.vk_session, nullptr);
    }
    if (segment.vk_pipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(vk_device, segment.vk_pipeline, nullptr);
    }
    if (segment.vk_pipeline_layout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(vk_device, segment.vk_pipeline_layout, nullptr);
    }
    if (segment.vk_descriptor_pool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(vk_device, segment.vk_descriptor_pool, nullptr);
    }
    if (segment.vk_layout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(vk_device, segment.vk_layout, nullptr);
    }
    if (segment.vk_shader != VK_NULL_HANDLE) {
      vkDestroyShaderModule(vk_device, segment.vk_shader, nullptr);
    }
  }
  segments.clear();
  for (int i = 0; i < IOs.size(); i++) {
    if (IOs[i].descriptor_type == VK_DESCRIPTOR_TYPE_TENSOR_ARM) {
      if (IOs[i].owns_memory) {
        remember_owned_memory(IOs[i].memory);
      }
      destroy_tensor(vk_device, IOs[i].tensor_view, IOs[i].tensor);
    } else if (IOs[i].descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
      if (IOs[i].owns_memory) {
        remember_owned_memory(IOs[i].memory);
      }
      destroy_buffer(vk_device, IOs[i].buffer);
    } else if (
        IOs[i].descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
        IOs[i].descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
        IOs[i].descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
      if (IOs[i].owns_memory) {
        remember_owned_memory(IOs[i].memory);
      }
      destroy_buffer(vk_device, IOs[i].buffer);
      if (IOs[i].owns_image_memory) {
        remember_owned_memory(IOs[i].image_memory);
      }
      free_image(
          vk_device,
          IOs[i].image_view,
          IOs[i].image,
          IOs[i].sampler,
          VK_NULL_HANDLE);
    }
  }
  IOs.clear();
  for (const auto& alloc : extra_allocs) {
    if (alloc.descriptor_type == VK_DESCRIPTOR_TYPE_TENSOR_ARM) {
      if (alloc.owns_memory) {
        remember_owned_memory(alloc.memory);
      }
      destroy_tensor(vk_device, alloc.tensor_view, alloc.tensor);
    } else if (alloc.descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
      if (alloc.owns_memory) {
        remember_owned_memory(alloc.memory);
      }
      destroy_buffer(vk_device, alloc.buffer);
    } else if (
        alloc.descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
        alloc.descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
        alloc.descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
      if (alloc.owns_image_memory) {
        remember_owned_memory(alloc.image_memory);
      }
      free_image(
          vk_device,
          alloc.image_view,
          alloc.image,
          alloc.sampler,
          VK_NULL_HANDLE);
    }
  }
  extra_allocs.clear();
  for (auto memory : owned_memory) {
    vkFreeMemory(vk_device, memory, nullptr);
  }
  for (auto memory : intermediates) {
    vkFreeMemory(vk_device, memory, nullptr);
  }
}

static uint32_t get_format_size(VkFormat format) {
  // Note: While this is a small subset of VkFormat, this supports all base
  //       types for tensors coming from the compiler flow. Tensor formats only
  //       specify single element type.
  switch (format) {
    case VK_FORMAT_R8_BOOL_ARM:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT:
      return 1;
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8_SINT:
      return 2;
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
      return 4;
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R64_SINT:
      return 8;
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return 16;
    default:
      ET_LOG(Error, "Unknown tensor format");
      return 0;
  }
}

} // namespace vgf
} // namespace backends
} // namespace executorch
