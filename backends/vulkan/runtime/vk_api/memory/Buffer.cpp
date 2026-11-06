/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/vk_api/memory/Buffer.h>

#ifdef VK_ANDROID_external_memory_android_hardware_buffer
#include <dlfcn.h>
#endif /* VK_ANDROID_external_memory_android_hardware_buffer */

namespace vkcompute {
namespace vkapi {

//
// VulkanBuffer
//

VulkanBuffer::VulkanBuffer()
    : buffer_properties_{},
      allocator_(VK_NULL_HANDLE),
      memory_{},
      owns_memory_(false),
      memory_bundled_(false),
      is_copy_(false),
      handle_(VK_NULL_HANDLE),
      external_memory_(VK_NULL_HANDLE) {}

VulkanBuffer::VulkanBuffer(
    VmaAllocator vma_allocator,
    const VkDeviceSize size,
    const VmaAllocationCreateInfo& allocation_create_info,
    const VkBufferUsageFlags usage,
    const bool allocate_memory)
    : buffer_properties_({size, 0u, size}),
      allocator_(vma_allocator),
      memory_{},
      owns_memory_(allocate_memory),
      memory_bundled_(allocate_memory),
      is_copy_(false),
      handle_(VK_NULL_HANDLE),
      external_memory_(VK_NULL_HANDLE) {
  // If the buffer size is 0, allocate a buffer with a size of 1 byte. This is
  // to ensure that there will be some resource that can be bound to a shader.
  if (size == 0) {
    buffer_properties_.size = 1u;
    buffer_properties_.mem_range = 1u;
  }

  const VkBufferCreateInfo buffer_create_info{
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      buffer_properties_.size, // size
      usage, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
  };

  if (allocate_memory) {
    VK_CHECK(vmaCreateBuffer(
        allocator_,
        &buffer_create_info,
        &allocation_create_info,
        &handle_,
        &(memory_.allocation),
        nullptr));
  } else {
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(allocator_, &allocator_info);
    VK_CHECK(vkCreateBuffer(
        allocator_info.device, &buffer_create_info, nullptr, &handle_));
  }
}

VulkanBuffer::VulkanBuffer(
    const VulkanBuffer& other,
    const VkDeviceSize offset,
    const VkDeviceSize range) noexcept
    : buffer_properties_(other.buffer_properties_),
      allocator_(other.allocator_),
      memory_(other.memory_),
      owns_memory_(false),
      memory_bundled_(false),
      is_copy_(true),
      handle_(other.handle_),
      external_memory_(VK_NULL_HANDLE) {
  // TODO: set the offset and range appropriately
  buffer_properties_.mem_offset = other.buffer_properties_.mem_offset + offset;
  if (range != VK_WHOLE_SIZE) {
    buffer_properties_.mem_range = range;
  }
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : buffer_properties_(other.buffer_properties_),
      allocator_(other.allocator_),
      memory_(std::move(other.memory_)),
      owns_memory_(other.owns_memory_),
      memory_bundled_(other.memory_bundled_),
      is_copy_(other.is_copy_),
      handle_(other.handle_),
      external_memory_(other.external_memory_) {
  other.handle_ = VK_NULL_HANDLE;
  other.external_memory_ = VK_NULL_HANDLE;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
  VkBuffer tmp_buffer = handle_;
  bool tmp_owns_memory = owns_memory_;
  bool tmp_memory_bundled = memory_bundled_;
  VkDeviceMemory tmp_external_memory = external_memory_;

  buffer_properties_ = other.buffer_properties_;
  allocator_ = other.allocator_;
  // Allocation's move assignment swaps, so `other` takes this buffer's previous
  // allocation along with the handle and flags swapped below, and frees them.
  memory_ = std::move(other.memory_);
  owns_memory_ = other.owns_memory_;
  memory_bundled_ = other.memory_bundled_;
  is_copy_ = other.is_copy_;
  handle_ = other.handle_;
  external_memory_ = other.external_memory_;

  other.handle_ = tmp_buffer;
  other.external_memory_ = tmp_external_memory;
  other.owns_memory_ = tmp_owns_memory;
  other.memory_bundled_ = tmp_memory_bundled;

  return *this;
}

VulkanBuffer::~VulkanBuffer() {
  if (external_memory_ != VK_NULL_HANDLE) {
    vkDestroyBuffer(this->device(), handle_, nullptr);
    vkFreeMemory(this->device(), external_memory_, nullptr);
    return;
  }
  // Do not destroy the VkBuffer if this class instance is a copy of another
  // class instance, since this means that this class instance does not have
  // ownership of the underlying resource.
  if (handle_ != VK_NULL_HANDLE && !is_copy_) {
    if (owns_memory_) {
      if (memory_bundled_) {
        vmaDestroyBuffer(allocator_, handle_, memory_.allocation);
        // Prevent the underlying memory allocation from being freed; it was
        // freed by vmaDestroyImage
        memory_.allocation = VK_NULL_HANDLE;
      } else {
        vkDestroyBuffer(this->device(), handle_, nullptr);
        // Allow underlying memory allocation to be freed by the destructor of
        // Allocation class
      }
    } else {
      vkDestroyBuffer(this->device(), handle_, nullptr);
      // Prevent the underlying memory allocation from being freed since this
      // object doesn't own it
      memory_.allocation = VK_NULL_HANDLE;
    }
  }
}

#ifdef VK_ANDROID_external_memory_android_hardware_buffer
namespace {

// Importing anything but a BLOB buffer with GPU_DATA_BUFFER usage into a
// VkBuffer is invalid usage, which a driver may answer with a crash rather
// than an error. AHardwareBuffer_describe is looked up at runtime because it
// needs API 26 at compile time; every device with AHardwareBuffer has it.
void check_importable_as_buffer(const AHardwareBuffer* ahb) {
  using DescribeFn = void (*)(const AHardwareBuffer*, AHardwareBuffer_Desc*);
  static const auto describe = reinterpret_cast<DescribeFn>(
      dlsym(RTLD_DEFAULT, "AHardwareBuffer_describe"));
  VK_CHECK_COND(describe != nullptr, "AHardwareBuffer_describe is unavailable");
  AHardwareBuffer_Desc desc{};
  describe(ahb, &desc);
  VK_CHECK_COND(
      desc.format == AHARDWAREBUFFER_FORMAT_BLOB &&
          (desc.usage & AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER) != 0,
      "AHardwareBuffer must be a BLOB buffer with GPU_DATA_BUFFER usage");
}

} // namespace

VulkanBuffer VulkanBuffer::import_android_hardware_buffer(
    const VmaAllocator vma_allocator,
    AHardwareBuffer* ahb,
    const VkBufferUsageFlags usage) {
  VulkanBuffer buffer;
  buffer.allocator_ = vma_allocator;
  const VkDevice device = buffer.device();

  VK_CHECK_COND(ahb != nullptr, "null AHardwareBuffer");
  check_importable_as_buffer(ahb);
  // volk leaves this null when the device was created without the extension.
  VK_CHECK_COND(
      vkGetAndroidHardwareBufferPropertiesANDROID != nullptr,
      "VK_ANDROID_external_memory_android_hardware_buffer is not enabled on "
      "this device");
  // Sized from Vulkan's view of the AHardwareBuffer: for a BLOB buffer the
  // allocation covers at least the requested width.
  VkAndroidHardwareBufferPropertiesANDROID props{
      VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
  VK_CHECK(vkGetAndroidHardwareBufferPropertiesANDROID(device, ahb, &props));
  buffer.buffer_properties_ = {props.allocationSize, 0u, props.allocationSize};

  const VkExternalMemoryBufferCreateInfo external_info{
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
      nullptr,
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
  };
  const VkBufferCreateInfo buffer_create_info{
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      &external_info,
      0u,
      props.allocationSize,
      usage,
      VK_SHARING_MODE_EXCLUSIVE,
      0u,
      nullptr,
  };
  VK_CHECK(vkCreateBuffer(device, &buffer_create_info, nullptr, &buffer.handle_));

  // Until the memory is bound the destructor does not know this buffer is
  // external, so a failure below cleans up here.
  try {
    bind_android_hardware_buffer(buffer, ahb, props);
  } catch (...) {
    if (buffer.external_memory_ != VK_NULL_HANDLE) {
      vkFreeMemory(device, buffer.external_memory_, nullptr);
      buffer.external_memory_ = VK_NULL_HANDLE;
    }
    vkDestroyBuffer(device, buffer.handle_, nullptr);
    buffer.handle_ = VK_NULL_HANDLE;
    throw;
  }
  return buffer;
}

void VulkanBuffer::bind_android_hardware_buffer(
    VulkanBuffer& buffer,
    AHardwareBuffer* ahb,
    const VkAndroidHardwareBufferPropertiesANDROID& props) {
  const VkDevice device = buffer.device();
  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device, buffer.handle_, &requirements);
  const uint32_t type_bits = props.memoryTypeBits & requirements.memoryTypeBits;
  VK_CHECK_COND(type_bits != 0, "no memory type fits the AHardwareBuffer");
  uint32_t memory_type = 0;
  while (!(type_bits & (1u << memory_type))) {
    ++memory_type;
  }

  const VkImportAndroidHardwareBufferInfoANDROID import_info{
      VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
      nullptr,
      ahb,
  };
  const VkMemoryAllocateInfo allocate_info{
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      &import_info,
      props.allocationSize,
      memory_type,
  };
  VK_CHECK(vkAllocateMemory(
      device, &allocate_info, nullptr, &buffer.external_memory_));
  VK_CHECK(vkBindBufferMemory(device, buffer.handle_, buffer.external_memory_, 0));
}
#endif /* VK_ANDROID_external_memory_android_hardware_buffer */

VmaAllocationInfo VulkanBuffer::allocation_info() const {
  VmaAllocationInfo info;
  vmaGetAllocationInfo(allocator_, memory_.allocation, &info);
  return info;
}

void VulkanBuffer::bind_allocation_impl(const Allocation& memory) {
  VK_CHECK_COND(!memory_, "Cannot bind an already bound allocation!");
  if (!is_copy_) {
    VK_CHECK(vmaBindBufferMemory(allocator_, memory.allocation, handle_));
  }
}

void VulkanBuffer::bind_allocation(const Allocation& memory) {
  bind_allocation_impl(memory);
  memory_.allocation = memory.allocation;
}

void VulkanBuffer::acquire_allocation(Allocation&& memory) {
  bind_allocation_impl(memory);
  memory_ = std::move(memory);
  owns_memory_ = true;
}

VkMemoryRequirements VulkanBuffer::get_memory_requirements() const {
  VkMemoryRequirements memory_requirements;
  vkGetBufferMemoryRequirements(this->device(), handle_, &memory_requirements);
  return memory_requirements;
}

//
// MemoryMap
//

MemoryMap::MemoryMap(const VulkanBuffer& buffer, const uint8_t access)
    : access_(access),
      allocator_(buffer.vma_allocator()),
      allocation_(buffer.allocation()),
      data_(nullptr),
      data_len_{buffer.mem_size()} {
  if (allocation_) {
    VK_CHECK(vmaMapMemory(allocator_, allocation_, &data_));
  }
}

MemoryMap::MemoryMap(MemoryMap&& other) noexcept
    : access_(other.access_),
      allocator_(other.allocator_),
      allocation_(other.allocation_),
      data_(other.data_),
      data_len_{other.data_len_} {
  other.allocation_ = VK_NULL_HANDLE;
  other.data_ = nullptr;
}

MemoryMap::~MemoryMap() {
  if (!data_) {
    return;
  }

  if (allocation_) {
    if (access_ & MemoryAccessType::WRITE) {
      // Call will be ignored by implementation if the memory type this
      // allocation belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is
      // the behavior we want. Don't check the result here as the destructor
      // cannot throw.
      vmaFlushAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE);
    }

    vmaUnmapMemory(allocator_, allocation_);
  }
}

void MemoryMap::invalidate() {
  if (access_ & MemoryAccessType::READ && allocation_) {
    // Call will be ignored by implementation if the memory type this allocation
    // belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is the behavior
    // we want.
    VK_CHECK(
        vmaInvalidateAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE));
  }
}

//
// BufferMemoryBarrier
//

BufferMemoryBarrier::BufferMemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags,
    const VulkanBuffer& buffer)
    : handle{
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
          VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
          VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
          buffer.handle_, // buffer
          buffer.buffer_properties_.mem_offset, // offset
          buffer.buffer_properties_.mem_range, // size
      } {}

} // namespace vkapi
} // namespace vkcompute
