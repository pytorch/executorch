/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/vk_api/memory/Image.h>

namespace vkcompute {
namespace vkapi {

//
// ImageSampler
//

bool operator==(
    const ImageSampler::Properties& _1,
    const ImageSampler::Properties& _2) {
  return (
      _1.filter == _2.filter && _1.mipmap_mode == _2.mipmap_mode &&
      _1.address_mode == _2.address_mode && _1.border_color == _2.border_color);
}

ImageSampler::ImageSampler(
    VkDevice device,
    const ImageSampler::Properties& props)
    : device_(device), handle_(VK_NULL_HANDLE) {
  const VkSamplerCreateInfo sampler_create_info{
      VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      props.filter, // magFilter
      props.filter, // minFilter
      props.mipmap_mode, // mipmapMode
      props.address_mode, // addressModeU
      props.address_mode, // addressModeV
      props.address_mode, // addressModeW
      0.0f, // mipLodBias
      VK_FALSE, // anisotropyEnable
      1.0f, // maxAnisotropy,
      VK_FALSE, // compareEnable
      VK_COMPARE_OP_NEVER, // compareOp
      0.0f, // minLod
      VK_LOD_CLAMP_NONE, // maxLod
      props.border_color, // borderColor
      VK_FALSE, // unnormalizedCoordinates
  };

  VK_CHECK(vkCreateSampler(device_, &sampler_create_info, nullptr, &handle_));
}

ImageSampler::ImageSampler(ImageSampler&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ImageSampler::~ImageSampler() {
  if (handle_ == VK_NULL_HANDLE) {
    return;
  }
  vkDestroySampler(device_, handle_, nullptr);
}

size_t ImageSampler::Hasher::operator()(
    const ImageSampler::Properties& props) const {
  size_t seed = 0;
  seed = utils::hash_combine(seed, std::hash<VkFilter>()(props.filter));
  seed = utils::hash_combine(
      seed, std::hash<VkSamplerMipmapMode>()(props.mipmap_mode));
  seed = utils::hash_combine(
      seed, std::hash<VkSamplerAddressMode>()(props.address_mode));
  seed =
      utils::hash_combine(seed, std::hash<VkBorderColor>()(props.border_color));
  return seed;
}

void swap(ImageSampler& lhs, ImageSampler& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkSampler tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// VulkanImage
//

VulkanImage::VulkanImage()
    : device_{VK_NULL_HANDLE},
      image_properties_{},
      view_properties_{},
      sampler_properties_{},
      allocator_(VK_NULL_HANDLE),
      memory_{},
      owns_memory_(false),
      memory_bundled_(false),
      owns_view_(false),
      is_copy_(false),
      handles_{
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
      },
      layout_{} {}

VulkanImage::VulkanImage(
    VkDevice device,
    VmaAllocator vma_allocator,
    const VmaAllocationCreateInfo& allocation_create_info,
    const ImageProperties& image_props,
    const ViewProperties& view_props,
    const SamplerProperties& sampler_props,
    VkSampler sampler,
    const VkImageLayout layout,
    const bool allocate_memory)
    : device_{device},
      image_properties_(image_props),
      view_properties_(view_props),
      sampler_properties_(sampler_props),
      allocator_(vma_allocator),
      memory_{},
      owns_memory_{allocate_memory},
      memory_bundled_(allocate_memory),
      owns_view_(false),
      is_copy_(false),
      handles_{
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          sampler,
      },
      layout_(layout) {
  VmaAllocatorInfo allocator_info{};
  vmaGetAllocatorInfo(allocator_, &allocator_info);

  // If any dims are zero, then allocate a 1x1x1 image texture. This is to
  // ensure that there will be some resource that can be bound to a shader.
  if (image_props.image_extents.width == 0 ||
      image_props.image_extents.height == 0 ||
      image_props.image_extents.depth == 0) {
    image_properties_.image_extents.width = 1u;
    image_properties_.image_extents.height = 1u;
    image_properties_.image_extents.depth = 1u;
  }

  const VkImageCreateInfo image_create_info{
      VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      image_properties_.image_type, // imageType
      image_properties_.image_format, // format
      image_properties_.image_extents, // extents
      1u, // mipLevels
      1u, // arrayLayers
      VK_SAMPLE_COUNT_1_BIT, // samples
      image_properties_.image_tiling, // tiling
      image_properties_.image_usage, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
      layout_, // initialLayout
  };

  if (allocate_memory) {
    VK_CHECK(vmaCreateImage(
        allocator_,
        &image_create_info,
        &allocation_create_info,
        &(handles_.image),
        &(memory_.allocation),
        nullptr));
    // Only create the image view if the image has been bound to memory
    owns_view_ = true;
    create_image_view();
  } else {
    VK_CHECK(vkCreateImage(
        allocator_info.device, &image_create_info, nullptr, &(handles_.image)));
  }
}

VulkanImage::VulkanImage(
    VkDevice device,
    const ImageProperties& image_props,
    VkImage image,
    VkImageView image_view,
    VkSampler sampler,
    const VkImageLayout layout)
    : device_{device},
      image_properties_{image_props},
      view_properties_{},
      sampler_properties_{},
      allocator_(VK_NULL_HANDLE),
      memory_{},
      owns_memory_(false),
      memory_bundled_(false),
      is_copy_(false),
      handles_{
          image,
          image_view,
          sampler,
      },
      layout_{layout} {}

VulkanImage::VulkanImage(const VulkanImage& other) noexcept
    : device_(other.device_),
      image_properties_(other.image_properties_),
      view_properties_(other.view_properties_),
      sampler_properties_(other.sampler_properties_),
      allocator_(other.allocator_),
      memory_(other.memory_),
      owns_memory_{false},
      owns_view_{false},
      is_copy_(true),
      handles_(other.handles_),
      layout_(other.layout_) {}

VulkanImage::VulkanImage(VulkanImage&& other) noexcept
    : device_(other.device_),
      image_properties_(other.image_properties_),
      view_properties_(other.view_properties_),
      sampler_properties_(other.sampler_properties_),
      allocator_(other.allocator_),
      memory_(std::move(other.memory_)),
      owns_memory_(other.owns_memory_),
      memory_bundled_(other.memory_bundled_),
      owns_view_(other.owns_view_),
      is_copy_(other.is_copy_),
      handles_(other.handles_),
      layout_(other.layout_) {
  other.handles_.image = VK_NULL_HANDLE;
  other.handles_.image_view = VK_NULL_HANDLE;
  other.handles_.sampler = VK_NULL_HANDLE;
  other.owns_memory_ = false;
  other.memory_bundled_ = false;
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other) noexcept {
  VkImage tmp_image = handles_.image;
  VkImageView tmp_image_view = handles_.image_view;
  bool tmp_owns_memory = owns_memory_;
  bool tmp_memory_bundled = memory_bundled_;

  device_ = other.device_;
  image_properties_ = other.image_properties_;
  view_properties_ = other.view_properties_;
  sampler_properties_ = other.sampler_properties_;
  allocator_ = other.allocator_;
  memory_ = std::move(other.memory_);
  owns_memory_ = other.owns_memory_;
  memory_bundled_ = other.memory_bundled_;
  is_copy_ = other.is_copy_;
  handles_ = other.handles_;
  layout_ = other.layout_;

  other.handles_.image = tmp_image;
  other.handles_.image_view = tmp_image_view;
  other.owns_memory_ = tmp_owns_memory;
  other.memory_bundled_ = tmp_memory_bundled;

  return *this;
}

VulkanImage::~VulkanImage() {
  if (owns_view_ && handles_.image_view != VK_NULL_HANDLE) {
    vkDestroyImageView(this->device(), handles_.image_view, nullptr);
  }

  // Do not destroy any resources if this class instance is a copy of another
  // class instance, since this means that this class instance does not have
  // ownership of the underlying resource.
  if (is_copy_) {
    return;
  }

  if (handles_.image != VK_NULL_HANDLE) {
    if (owns_memory_) {
      if (memory_bundled_) {
        vmaDestroyImage(allocator_, handles_.image, memory_.allocation);
        // Prevent the underlying memory allocation from being freed; it was
        // freed by vmaDestroyImage
        memory_.allocation = VK_NULL_HANDLE;
      } else {
        vkDestroyImage(this->device(), handles_.image, nullptr);
        // Allow underlying memory allocation to be freed by the destructor of
        // Allocation class
      }
    } else {
      vkDestroyImage(this->device(), handles_.image, nullptr);
      // Prevent the underlying memory allocation from being freed since this
      // object doesn't own it
      memory_.allocation = VK_NULL_HANDLE;
    }
  }
}

void VulkanImage::create_image_view() {
  VmaAllocatorInfo allocator_info{};
  vmaGetAllocatorInfo(allocator_, &allocator_info);

  const VkComponentMapping component_mapping{
      VK_COMPONENT_SWIZZLE_IDENTITY, // r
      VK_COMPONENT_SWIZZLE_IDENTITY, // g
      VK_COMPONENT_SWIZZLE_IDENTITY, // b
      VK_COMPONENT_SWIZZLE_IDENTITY, // a
  };

  const VkImageSubresourceRange subresource_range{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // baseMipLevel
      VK_REMAINING_MIP_LEVELS, // levelCount
      0u, // baseArrayLayer
      VK_REMAINING_ARRAY_LAYERS, // layerCount
  };

  const VkImageViewCreateInfo image_view_create_info{
      VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      handles_.image, // image
      view_properties_.view_type, // viewType
      view_properties_.view_format, // format
      component_mapping, // components
      subresource_range, // subresourceRange
  };

  VK_CHECK(vkCreateImageView(
      allocator_info.device,
      &(image_view_create_info),
      nullptr,
      &(handles_.image_view)));
}

void VulkanImage::bind_allocation_impl(const Allocation& memory) {
  VK_CHECK_COND(!memory_, "Cannot bind an already bound allocation!");
  // To prevent multiple instances of binding the same VkImage to a memory
  // block, do not actually bind memory if this VulkanImage is a copy. Assume
  // that the original VulkanImage is responsible for binding the image.
  if (!is_copy_) {
    VK_CHECK(vmaBindImageMemory(allocator_, memory.allocation, handles_.image));
  }

  // Only create the image view if the image has been bound to memory
  owns_view_ = true;
  create_image_view();
}

void VulkanImage::bind_allocation(const Allocation& memory) {
  bind_allocation_impl(memory);
  memory_.allocation = memory.allocation;
}

void VulkanImage::acquire_allocation(Allocation&& memory) {
  bind_allocation_impl(memory);
  memory_ = std::move(memory);
  owns_memory_ = true;
}

VkMemoryRequirements VulkanImage::get_memory_requirements() const {
  VkMemoryRequirements memory_requirements;
  vkGetImageMemoryRequirements(
      this->device(), handles_.image, &memory_requirements);
  return memory_requirements;
}

//
// ImageMemoryBarrier
//

ImageMemoryBarrier::ImageMemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags,
    const VkImageLayout src_layout_flags,
    const VkImageLayout dst_layout_flags,
    const VulkanImage& image)
    : handle{
          VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
          src_layout_flags, // oldLayout
          dst_layout_flags, // newLayout
          VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
          VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
          image.handles_.image, // image
          {
              // subresourceRange
              VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
              0u, // baseMipLevel
              VK_REMAINING_MIP_LEVELS, // levelCount
              0u, // baseArrayLayer
              VK_REMAINING_ARRAY_LAYERS, // layerCount
          },
      } {}

//
// SamplerCache
//

SamplerCache::SamplerCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

SamplerCache::SamplerCache(SamplerCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}

SamplerCache::~SamplerCache() {
  purge();
}

VkSampler SamplerCache::retrieve(const SamplerCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    it = cache_.insert({key, SamplerCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void SamplerCache::purge() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.clear();
}

} // namespace vkapi
} // namespace vkcompute
