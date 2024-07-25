/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/containers/Tensor.h>

#include <executorch/backends/vulkan/runtime/vk_api/VkUtils.h>

namespace vkcompute {
namespace api {

std::vector<int64_t> calculate_strides(
    const std::vector<int64_t>& sizes,
    const utils::GPUMemoryLayout memory_layout,
    const bool texel_strides) {
  const int64_t dim_offset =
      utils::to_packed_dim_nchw_offset<int64_t>(memory_layout);
  const int64_t last_dim = sizes.size() - dim_offset;
  VK_CHECK_COND(last_dim >= 0);

  size_t ndim = sizes.size();
  std::vector<int64_t> strides(ndim);

  const int64_t last_dim_size =
      texel_strides ? utils::div_up_4(sizes.at(last_dim)) : sizes.at(last_dim);

  for (int stride_d = ndim - 1; stride_d >= 0; stride_d--) {
    strides.at(stride_d) = 1;
    if (stride_d == last_dim) {
      continue;
    }
    strides.at(stride_d) = last_dim_size;
    for (int size_d = ndim - 1; size_d > stride_d; size_d--) {
      if (size_d != last_dim) {
        strides.at(stride_d) *= sizes.at(size_d);
      }
    }
  }
  return strides;
}

std::vector<int64_t> calculate_padded_sizes(
    const std::vector<int64_t>& sizes,
    const utils::GPUMemoryLayout memory_layout) {
  int64_t ndim = sizes.size();
  if (ndim == 0) {
    ndim = 1;
  }

  // Tensor sizes will be unsqueezed up to the next multiple of 4
  const int64_t ndim_up4 = utils::align_up_4(ndim);
  std::vector<int64_t> padded_sizes(ndim_up4);
  for (int64_t i = 0; i < ndim_up4; ++i) {
    padded_sizes.at(i) = utils::val_at(i - ndim_up4, sizes);
  }

  // Pad the packed dim to the next multiple of 4.
  const int64_t dim_offset =
      utils::to_packed_dim_nchw_offset<int64_t>(memory_layout);
  const int64_t padded_dim_size = utils::val_at(-dim_offset, sizes);
  padded_sizes.at(ndim_up4 - dim_offset) = utils::align_up_4(padded_dim_size);

  return padded_sizes;
}

utils::uvec3 calculate_image_extents(
    const std::vector<int64_t>& padded_sizes,
    const utils::GPUMemoryLayout memory_layout) {
  VK_CHECK_COND(padded_sizes.size() == 4);

  uint32_t N = utils::safe_downcast<uint32_t>(padded_sizes.at(0));
  uint32_t C = utils::safe_downcast<uint32_t>(padded_sizes.at(1));
  uint32_t H = utils::safe_downcast<uint32_t>(padded_sizes.at(2));
  uint32_t W = utils::safe_downcast<uint32_t>(padded_sizes.at(3));

  switch (memory_layout) {
    case utils::kWidthPacked:
      VK_CHECK_COND(W % 4 == 0);
      W /= 4;
      break;
    case utils::kHeightPacked:
      VK_CHECK_COND(H % 4 == 0);
      H /= 4;
      break;
    case utils::kChannelsPacked:
      VK_CHECK_COND(C % 4 == 0);
      C /= 4;
      break;
  }

  return {W, H, C * N};
}

//
// vTensor
//

vTensor::vTensor(
    Context* const context,
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout memory_layout,
    const bool allocate_memory)
    : dtype_(dtype),
      memory_layout_(memory_layout),
      // Calculate sizes and strides
      sizes_(sizes.begin(), sizes.end()),
      padded_sizes_{calculate_padded_sizes(sizes, memory_layout_)},
      texture_limits_{{0, 0, 0}},
      // Utility Uniform Buffers that can be passed to shaders as arguments
      sizes_uniform_(),
      texture_limits_uniform_(),
      texel_strides_uniform_(),
      ntexels_uniform_(),
      // Construct Tensor storage
      storage_(
          context,
          storage_type,
          memory_layout_,
          padded_sizes_,
          dtype_,
          allocate_memory) {
  if (storage_type != utils::kBuffer) {
    texture_limits_.limits = utils::ivec3{
        utils::safe_downcast<int32_t>(storage_.image_extents_.data[0]),
        utils::safe_downcast<int32_t>(storage_.image_extents_.data[1]),
        utils::safe_downcast<int32_t>(storage_.image_extents_.data[2])};
  }

  if (dtype == vkapi::kHalf) {
    VK_CHECK_COND(
        api::context()->adapter_ptr()->has_16bit_storage(),
        "Half dtype is only available if the physical device supports float16 "
        "storage buffers!");
  }
}

vkapi::VulkanImage& vTensor::image(
    vkapi::PipelineBarrier& pipeline_barrier,
    const vkapi::PipelineStageFlags stage) & {
  storage_.transition(pipeline_barrier, stage, vkapi::MemoryAccessType::READ);
  return storage_.image_;
}

vkapi::VulkanImage& vTensor::image(
    vkapi::PipelineBarrier& pipeline_barrier,
    const vkapi::PipelineStageFlags stage,
    const vkapi::MemoryAccessFlags access) & {
  storage_.transition(pipeline_barrier, stage, access);
  return storage_.image_;
}

vkapi::VulkanBuffer& vTensor::buffer(
    vkapi::PipelineBarrier& pipeline_barrier,
    const vkapi::PipelineStageFlags stage) & {
  storage_.transition(pipeline_barrier, stage, vkapi::MemoryAccessType::READ);
  return storage_.buffer_;
}

vkapi::VulkanBuffer& vTensor::buffer(
    vkapi::PipelineBarrier& pipeline_barrier,
    const vkapi::PipelineStageFlags stage,
    const vkapi::MemoryAccessFlags access) & {
  storage_.transition(pipeline_barrier, stage, access);
  return storage_.buffer_;
}

const vkapi::BufferBindInfo vTensor::sizes_ubo() {
  if (!sizes_uniform_.buffer()) {
    sizes_uniform_ =
        ParamsBuffer(storage_.context_, utils::make_whcn_ivec4(sizes_));
  }
  return vkapi::BufferBindInfo(sizes_uniform_.buffer());
}

const vkapi::BufferBindInfo vTensor::texture_limits_ubo() {
  if (!texture_limits_uniform_.buffer()) {
    texture_limits_uniform_ = ParamsBuffer(storage_.context_, texture_limits_);
  }
  return vkapi::BufferBindInfo(texture_limits_uniform_.buffer());
}

const vkapi::BufferBindInfo vTensor::texel_strides_ubo() {
  if (!texel_strides_uniform_.buffer()) {
    texel_strides_uniform_ = ParamsBuffer(
        storage_.context_,
        utils::make_whcn_ivec4(
            calculate_strides(padded_sizes_, memory_layout_)));
  }
  return vkapi::BufferBindInfo(texel_strides_uniform_.buffer());
}

const vkapi::BufferBindInfo vTensor::ntexels_ubo() {
  if (!ntexels_uniform_.buffer()) {
    ntexels_uniform_ = ParamsBuffer(storage_.context_, texel_numel());
  }
  return vkapi::BufferBindInfo(ntexels_uniform_.buffer());
}

VmaAllocationCreateInfo vTensor::get_allocation_create_info() const {
  switch (storage_type()) {
    case utils::kBuffer:
      return storage_.buffer_.allocation_create_info();
    case utils::kTexture2D:
    case utils::kTexture3D:
      return storage_.image_.allocation_create_info();
  }
  return {};
}

VkMemoryRequirements vTensor::get_memory_requirements() const {
  switch (storage_type()) {
    case utils::kBuffer:
      return storage_.buffer_.get_memory_requirements();
    case utils::kTexture2D:
    case utils::kTexture3D:
      return storage_.image_.get_memory_requirements();
  }
  return {};
}

void vTensor::bind_allocation(const vkapi::Allocation& allocation) {
  switch (storage_type()) {
    case utils::kBuffer:
      storage_.buffer_.bind_allocation(allocation);
      break;
    case utils::kTexture2D:
    case utils::kTexture3D:
      storage_.image_.bind_allocation(allocation);
      break;
  }
}

void vTensor::update_size_metadata(const std::vector<int64_t>& new_sizes) {
  sizes_ = new_sizes;
  padded_sizes_ = calculate_padded_sizes(sizes_, memory_layout_);

  // Calculate the extents of the image texture that would have been required
  // for a tensor of the new sizes.
  utils::uvec3 virtual_extents =
      calculate_image_extents(padded_sizes_, memory_layout_);

  // Update the texture limits to reflect the new virtual extents.
  texture_limits_.limits = utils::ivec3{
      utils::safe_downcast<int32_t>(virtual_extents.data[0]),
      utils::safe_downcast<int32_t>(virtual_extents.data[1]),
      utils::safe_downcast<int32_t>(virtual_extents.data[2])};

  if (sizes_uniform_.buffer()) {
    sizes_uniform_.update(utils::make_whcn_ivec4(sizes_));
  }
  if (texture_limits_uniform_.buffer()) {
    texture_limits_uniform_.update(texture_limits_);
  }
  if (texel_strides_uniform_.buffer()) {
    texel_strides_uniform_.update(utils::make_whcn_ivec4(
        calculate_strides(padded_sizes_, memory_layout_)));
  }
  if (ntexels_uniform_.buffer()) {
    ntexels_uniform_.update(texel_numel());
  }
}

void vTensor::reallocate(const std::vector<int64_t>& new_sizes) {
  update_size_metadata(new_sizes);
  storage_.discard_and_reallocate(
      calculate_padded_sizes(new_sizes, memory_layout_),
      memory_layout_,
      dtype_);
}

void vTensor::virtual_resize(const std::vector<int64_t>& new_sizes) {
  if (storage_type() != utils::kBuffer) {
    // For texture storage check that the current texture is large enough for
    // the new sizes of the tensor.
    utils::uvec3 virtual_extents =
        calculate_image_extents(padded_sizes_, memory_layout_);

    bool valid_resize = virtual_extents.data[0] <= image_extents().data[0];
    valid_resize =
        valid_resize && virtual_extents.data[1] <= image_extents().data[1];
    valid_resize =
        valid_resize && virtual_extents.data[2] <= image_extents().data[2];

    VK_CHECK_COND(
        valid_resize,
        "Cannot use virtual resize if new sizes requires a larger texture.");
  }

  update_size_metadata(new_sizes);
}

//
// vTensorStorage
//

vkapi::VulkanImage allocate_image(
    Context* const context_ptr,
    utils::uvec3& image_extents,
    const utils::StorageType storage_type,
    const VkFormat image_format,
    const bool allocate_memory) {
  vkapi::Adapter* adapter_ptr = context_ptr->adapter_ptr();

  vkapi::ImageSampler::Properties sampler_props{
      VK_FILTER_NEAREST,
      VK_SAMPLER_MIPMAP_MODE_NEAREST,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
  };

  VkImageType image_type = VK_IMAGE_TYPE_3D;
  VkImageViewType image_view_type;

  switch (storage_type) {
    case utils::kTexture3D:
      image_type = VK_IMAGE_TYPE_3D;
      image_view_type = VK_IMAGE_VIEW_TYPE_3D;
      break;
    case utils::kTexture2D:
      image_type = VK_IMAGE_TYPE_2D;
      image_view_type = VK_IMAGE_VIEW_TYPE_2D;
      break;
    default:
      // Return an empty VulkanImage by default
      return vkapi::VulkanImage();
  }

  VkSampler sampler = adapter_ptr->sampler_cache().retrieve(sampler_props);

  return adapter_ptr->vma().create_image(
      vkapi::create_extent3d(image_extents),
      image_format,
      image_type,
      image_view_type,
      sampler_props,
      sampler,
      /*allow_transfer = */ true,
      /*allocate_memory = */ allocate_memory);
}

vkapi::VulkanBuffer allocate_buffer(
    Context* const context_ptr,
    const int64_t numel,
    const utils::StorageType storage_type,
    const vkapi::ScalarType dtype,
    const bool allocate_memory) {
  vkapi::Adapter* adapter_ptr = context_ptr->adapter_ptr();

  switch (storage_type) {
    case utils::kBuffer:
      break;
    default:
      // Return an empty VulkanBuffer if Buffer storage is not used
      return vkapi::VulkanBuffer();
  }

  return adapter_ptr->vma().create_storage_buffer(
      element_size(dtype) * numel, /*gpu_only = */ true, allocate_memory);
}

vTensorStorage::vTensorStorage(
    Context* const context,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout gpu_memory_layout,
    const std::vector<int64_t>& padded_sizes,
    const vkapi::ScalarType dtype,
    const bool allocate_memory)
    : context_(context),
      storage_type_{storage_type},
      image_extents_(calculate_image_extents(padded_sizes, gpu_memory_layout)),
      buffer_length_{utils::multiply_integers(padded_sizes)},
      image_(allocate_image(
          context_,
          image_extents_,
          storage_type_,
          to_vkformat(dtype),
          allocate_memory)),
      buffer_(allocate_buffer(
          context_,
          buffer_length_,
          storage_type_,
          dtype,
          allocate_memory)),
      last_access_{} {}

vTensorStorage::~vTensorStorage() {
  flush();
}

void vTensorStorage::flush() {
  if (image_) {
    context_->register_image_cleanup(image_);
  } else if (buffer_) {
    context_->register_buffer_cleanup(buffer_);
  }
  last_access_ = {};
}

void vTensorStorage::transition(
    vkapi::PipelineBarrier& pipeline_barrier,
    const vkapi::PipelineStageFlags cur_stage,
    const vkapi::MemoryAccessFlags cur_access) {
  // Get last stage access
  vkapi::PipelineStageFlags prev_stage = last_access_.stage;
  vkapi::MemoryAccessFlags prev_access = last_access_.access;

  const bool prev_written = (prev_access & vkapi::MemoryAccessType::WRITE) != 0;

  VkImageLayout cur_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout new_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  bool layout_changed = false;
  if (image_) {
    cur_layout = image_.layout();
    new_layout = vkapi::vk_layout(cur_stage, cur_access);

    layout_changed = cur_layout != new_layout;
  }

  if (prev_written || layout_changed) {
    VkPipelineStageFlags src_stage = vkapi::vk_stage(prev_stage);
    if (0u == src_stage) {
      src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
    VkPipelineStageFlags dst_stage = vkapi::vk_stage(cur_stage);
    if (0u == dst_stage) {
      dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    pipeline_barrier.stage.src |= src_stage;
    pipeline_barrier.stage.dst |= dst_stage;

    if (image_) {
      pipeline_barrier.images.emplace_back(
          vkapi::vk_access(prev_stage, prev_access),
          vkapi::vk_access(cur_stage, cur_access),
          cur_layout,
          new_layout,
          image_);

      image_.set_layout(new_layout);
    } else if (buffer_) {
      pipeline_barrier.buffers.emplace_back(
          vkapi::vk_access(prev_stage, prev_access),
          vkapi::vk_access(cur_stage, cur_access),
          buffer_);
    }
  }

  last_access_.stage = cur_stage;
  last_access_.access = cur_access;
}

void vTensorStorage::discard_and_reallocate(
    const std::vector<int64_t>& padded_sizes,
    const utils::GPUMemoryLayout gpu_memory_layout,
    const vkapi::ScalarType dtype) {
  const bool image_owns_memory = image_.owns_memory();
  const bool buffer_owns_memory = buffer_.owns_memory();

  flush();

  image_extents_ = calculate_image_extents(padded_sizes, gpu_memory_layout);
  image_ = allocate_image(
      context_,
      image_extents_,
      storage_type_,
      to_vkformat(dtype),
      image_owns_memory);

  buffer_length_ = utils::multiply_integers(padded_sizes);
  buffer_ = allocate_buffer(
      context_, buffer_length_, storage_type_, dtype, buffer_owns_memory);
}

} // namespace api
} // namespace vkcompute
