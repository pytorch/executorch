/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

#include <executorch/backends/vulkan/runtime/utils/VecUtils.h>

#include <executorch/backends/vulkan/runtime/vk_api/memory/vma_api.h>

#include <executorch/backends/vulkan/runtime/vk_api/memory/Allocation.h>

#include <mutex>
#include <unordered_map>

namespace vkcompute {

// Forward declare vTensor classes such that they can be set as friend classes
namespace api {
class vTensorStorage;
} // namespace api

namespace vkapi {

class ImageSampler final {
 public:
  struct Properties final {
    VkFilter filter;
    VkSamplerMipmapMode mipmap_mode;
    VkSamplerAddressMode address_mode;
    VkBorderColor border_color;
  };

  explicit ImageSampler(VkDevice, const Properties&);

  ImageSampler(const ImageSampler&) = delete;
  ImageSampler& operator=(const ImageSampler&) = delete;

  ImageSampler(ImageSampler&&) noexcept;
  ImageSampler& operator=(ImageSampler&&) = delete;

  ~ImageSampler();

 private:
  VkDevice device_;
  VkSampler handle_;

 public:
  VkSampler handle() const {
    return handle_;
  }

  struct Hasher {
    size_t operator()(const Properties&) const;
  };

  // We need to define a custom swap function since this class
  // does not allow for move assignment. The swap function will
  // be used in the hash map.
  friend void swap(ImageSampler& lhs, ImageSampler& rhs) noexcept;
};

class VulkanImage final {
 public:
  struct ImageProperties final {
    VkImageType image_type;
    VkFormat image_format;
    VkExtent3D image_extents;
    VkImageUsageFlags image_usage;
  };

  struct ViewProperties final {
    VkImageViewType view_type;
    VkFormat view_format;
  };

  using SamplerProperties = ImageSampler::Properties;

  struct Handles final {
    VkImage image;
    VkImageView image_view;
    VkSampler sampler;
  };

  explicit VulkanImage();

  explicit VulkanImage(
      const VmaAllocator,
      const VmaAllocationCreateInfo&,
      const ImageProperties&,
      const ViewProperties&,
      const SamplerProperties&,
      const VkImageLayout layout,
      VkSampler,
      const bool allocate_memory = true);

 protected:
  /*
   * The Copy constructor allows for creation of a class instance that are
   * "aliases" of another class instance. The resulting class instance will not
   * have ownership of the underlying VkImage.
   *
   * This behaviour is analogous to creating a copy of a pointer, thus it is
   * unsafe, as the original class instance may be destroyed before the copy.
   * These constructors are therefore marked protected so that they may be used
   * only in situations where the lifetime of the original class instance is
   * guaranteed to exceed, or at least be the same as, the lifetime of the
   * copied class instance.
   */
  VulkanImage(const VulkanImage& other) noexcept;

 public:
  // To discourage creating copies, the assignment operator is still deleted.
  VulkanImage& operator=(const VulkanImage&) = delete;

  VulkanImage(VulkanImage&&) noexcept;
  VulkanImage& operator=(VulkanImage&&) noexcept;

  ~VulkanImage();

  struct Package final {
    VkImage handle;
    VkImageLayout image_layout;
    VkImageView image_view;
    VkSampler image_sampler;
  };

  friend struct ImageMemoryBarrier;

 private:
  ImageProperties image_properties_;
  ViewProperties view_properties_;
  SamplerProperties sampler_properties_;
  // The allocator object this was allocated from
  VmaAllocator allocator_;
  // Handles to the allocated memory
  Allocation memory_;
  // Indicates whether the underlying memory is owned by this resource
  bool owns_memory_;
  // Indicates whether this VulkanImage was copied from another VulkanImage,
  // thus it does not have ownership of the underlying VKBuffer
  bool is_copy_;
  Handles handles_;
  // Layout
  VkImageLayout layout_;

 public:
  void create_image_view();

  inline VkDevice device() const {
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(allocator_, &allocator_info);
    return allocator_info.device;
  }

  inline VmaAllocator vma_allocator() const {
    return allocator_;
  }

  inline VmaAllocation allocation() const {
    return memory_.allocation;
  }

  inline VmaAllocationCreateInfo allocation_create_info() const {
    return VmaAllocationCreateInfo(memory_.create_info);
  }

  inline VkFormat format() const {
    return image_properties_.image_format;
  }

  inline VkExtent3D extents() const {
    return image_properties_.image_extents;
  }

  inline VkImage handle() const {
    return handles_.image;
  }

  inline VkImageView image_view() const {
    return handles_.image_view;
  }

  inline VkSampler sampler() const {
    return handles_.sampler;
  }

  Package package() const {
    return {
        handles_.image,
        layout_,
        handles_.image_view,
        handles_.sampler,
    };
  }

  inline VkImageLayout layout() const {
    return layout_;
  }

  inline void set_layout(const VkImageLayout layout) {
    layout_ = layout;
  }

  inline bool has_memory() const {
    return (memory_.allocation != VK_NULL_HANDLE);
  }

  inline bool owns_memory() const {
    return owns_memory_;
  }

  inline bool is_copy() const {
    return is_copy_;
  }

  inline operator bool() const {
    return (handles_.image != VK_NULL_HANDLE);
  }

  inline bool is_copy_of(const VulkanImage& other) const {
    return (handles_.image == other.handles_.image) && is_copy_;
  }

  inline void bind_allocation(const Allocation& memory) {
    VK_CHECK_COND(!memory_, "Cannot bind an already bound allocation!");
    VK_CHECK(vmaBindImageMemory(allocator_, memory.allocation, handles_.image));
    memory_.allocation = memory.allocation;

    // Only create the image view if the image has been bound to memory
    create_image_view();
  }

  VkMemoryRequirements get_memory_requirements() const;

  friend class api::vTensorStorage;
};

struct ImageMemoryBarrier final {
  VkImageMemoryBarrier handle;

  ImageMemoryBarrier(
      const VkAccessFlags src_access_flags,
      const VkAccessFlags dst_access_flags,
      const VkImageLayout src_layout_flags,
      const VkImageLayout dst_layout_flags,
      const VulkanImage& image);
};

class SamplerCache final {
 public:
  explicit SamplerCache(VkDevice device);

  SamplerCache(const SamplerCache&) = delete;
  SamplerCache& operator=(const SamplerCache&) = delete;

  SamplerCache(SamplerCache&&) noexcept;
  SamplerCache& operator=(SamplerCache&&) = delete;

  ~SamplerCache();

  using Key = ImageSampler::Properties;
  using Value = ImageSampler;
  using Hasher = ImageSampler::Hasher;

 private:
  // Multiple threads could potentially be adding entries into the cache, so use
  // a mutex to manage access
  std::mutex cache_mutex_;

  VkDevice device_;
  std::unordered_map<Key, Value, Hasher> cache_;

 public:
  VkSampler retrieve(const Key&);
  void purge();
};

} // namespace vkapi
} // namespace vkcompute
