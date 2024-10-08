/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <executorch/backends/vulkan/runtime/vk_api/VkUtils.h>

namespace vkcompute {
namespace utils {
namespace {

uint32_t find_mem_type_idx(
    api::Context* context,
    uint32_t mem_type_bits,
    const VkFlags flags) {
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(
      context->adapter_ptr()->physical_handle(), &mem_props);

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
    if ((mem_type_bits & 1u) &&
        (mem_props.memoryTypes[i].propertyFlags & flags) == flags) {
      return i;
    }
    mem_type_bits >>= 1u;
  }
  VK_THROW("Failed to find memory type index for AHB");
}

VkImage create_image(
    api::Context* context,
    AHardwareBuffer* ahb,
    const VkImageUsageFlags usage) {
  AHardwareBuffer_acquire(ahb);
  AHardwareBuffer_Desc desc;
  AHardwareBuffer_describe(ahb, &desc);

  VkAndroidHardwareBufferFormatPropertiesANDROID format_info = {
      .sType =
          VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID,
      .pNext = nullptr,
  };

  VkAndroidHardwareBufferPropertiesANDROID ahb_props = {
      .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID,
      .pNext = &format_info,
  };

  VK_CHECK(vkGetAndroidHardwareBufferPropertiesANDROID(
      context->device(), ahb, &ahb_props));

  const VkExternalMemoryImageCreateInfo ext_create_info = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .pNext = nullptr,
      .handleTypes =
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
  };
  const auto format = VK_FORMAT_R8G8B8A8_UNORM;
  const auto tiling = VK_IMAGE_TILING_LINEAR;
  const VkImageCreateInfo createInfo = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &ext_create_info,
      .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {desc.width, desc.height, 1u},
      .mipLevels = 1u,
      .arrayLayers = 1u,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = tiling,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  VkImage image;
  VK_CHECK(vkCreateImage(context->device(), &createInfo, nullptr, &image));

  const auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(
      context->adapter_ptr()->physical_handle(), &mem_props);

  const auto mem_type_idx =
      find_mem_type_idx(context, ahb_props.memoryTypeBits, flags);

  const VkImportAndroidHardwareBufferInfoANDROID androidHardwareBufferInfo = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
      .pNext = nullptr,
      .buffer = ahb,
  };
  const VkMemoryDedicatedAllocateInfo mem_alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .pNext = &androidHardwareBufferInfo,
      .image = image,
      .buffer = VK_NULL_HANDLE,
  };
  const VkMemoryAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &mem_alloc_info,
      .allocationSize = ahb_props.allocationSize,
      .memoryTypeIndex = mem_type_idx,
  };

  VkDeviceMemory memory;
  VK_CHECK(vkAllocateMemory(context->device(), &alloc_info, nullptr, &memory));
  VK_CHECK(vkBindImageMemory(context->device(), image, memory, 0));

  return image;
}

VkImageView create_image_view(api::Context* context, const VkImage& image) {
  const VkImageViewCreateInfo image_view_ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .components =
          {
              VK_COMPONENT_SWIZZLE_IDENTITY,
              VK_COMPONENT_SWIZZLE_IDENTITY,
              VK_COMPONENT_SWIZZLE_IDENTITY,
              VK_COMPONENT_SWIZZLE_IDENTITY,
          },
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
  };

  VkImageView image_view;
  VK_CHECK(vkCreateImageView(
      context->device(), &image_view_ci, nullptr, &image_view));
  return image_view;
}

VkSampler create_sampler(api::Context* context) {
  vkapi::ImageSampler::Properties sampler_props{
      VK_FILTER_LINEAR,
      VK_SAMPLER_MIPMAP_MODE_LINEAR,
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
  };
  auto sampler =
      context->adapter_ptr()->sampler_cache().retrieve(sampler_props);
  return sampler;
}

} // namespace

vkapi::VulkanImage create_image_from_ahb(
    api::Context* context,
    AHardwareBuffer* ahb,
    const uint32_t width,
    const uint32_t height,
    const VkImageUsageFlags usage) {
  const auto image_type = VK_IMAGE_TYPE_2D;
  const auto image_format = VK_FORMAT_R8G8B8A8_UNORM;
  const utils::uvec3 extents({width, height, 1});

  const vkapi::VulkanImage::ImageProperties image_props = {
      image_type, image_format, vkapi::create_extent3d(extents), usage};

  auto image = create_image(context, ahb, usage);
  auto image_view = create_image_view(context, image);
  auto sampler = (usage & VK_IMAGE_USAGE_SAMPLED_BIT) ? create_sampler(context)
                                                      : VK_NULL_HANDLE;

  return vkapi::VulkanImage(
      context->device(), image_props, image, image_view, sampler);
}

void add_rgba_to_image_node(
    ComputeGraph& graph,
    const ValueRef in_rgba,
    const ValueRef out_tensor) {
  std::string kernel_name("rgba_to_image");
  kernel_name.reserve(kShaderNameReserve);

  const auto global_wg_size = graph.create_global_wg_size(out_tensor);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      // Input and Outputs
      {{out_tensor, vkapi::MemoryAccessType::WRITE},
       {in_rgba, vkapi::MemoryAccessType::READ}},
      // Parameter Buffers
      {graph.logical_limits_ubo(out_tensor)}));
}

void add_image_to_rgba_node(
    ComputeGraph& graph,
    const ValueRef in_tensor,
    const ValueRef out_rgba) {
  std::string kernel_name("image_to_rgba");
  kernel_name.reserve(kShaderNameReserve);

  const auto global_wg_size = graph.create_global_wg_size(out_rgba);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      // Input and Outputs
      {{out_rgba, vkapi::MemoryAccessType::WRITE},
       {in_tensor, vkapi::MemoryAccessType::READ}},
      // Parameter Buffers
      {graph.logical_limits_ubo(out_rgba)}));
}

} // namespace utils
} // namespace vkcompute
