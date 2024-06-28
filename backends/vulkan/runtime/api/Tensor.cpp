/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/Tensor.h>
#include <executorch/backends/vulkan/runtime/api/Utils.h>

namespace vkcompute {

std::vector<int64_t> calculate_strides(
    const std::vector<int64_t>& sizes,
    const api::GPUMemoryLayout memory_layout,
    const bool texel_strides) {
  const int64_t dim_offset =
      api::to_packed_dim_nchw_offset<int64_t>(memory_layout);
  const int64_t last_dim = sizes.size() - dim_offset;
  VK_CHECK_COND(last_dim >= 0);

  size_t ndim = sizes.size();
  std::vector<int64_t> strides(ndim);

  const int64_t last_dim_size = texel_strides
      ? api::utils::div_up_4(sizes.at(last_dim))
      : sizes.at(last_dim);

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
    const api::GPUMemoryLayout memory_layout) {
  int64_t ndim = sizes.size();
  if (ndim == 0) {
    ndim = 1;
  }

  // Tensor sizes will be unsqueezed up to the next multiple of 4
  const int64_t ndim_up4 = api::utils::align_up_4(ndim);
  std::vector<int64_t> padded_sizes(ndim_up4);
  for (int64_t i = 0; i < ndim_up4; ++i) {
    padded_sizes.at(i) = api::utils::val_at(i - ndim_up4, sizes);
  }

  // Pad the packed dim to the next multiple of 4.
  const int64_t dim_offset =
      api::to_packed_dim_nchw_offset<int64_t>(memory_layout);
  const int64_t padded_dim_size = api::utils::val_at(-dim_offset, sizes);
  padded_sizes.at(ndim_up4 - dim_offset) =
      api::utils::align_up_4(padded_dim_size);

  return padded_sizes;
}

api::utils::uvec3 calculate_image_extents(
    const std::vector<int64_t>& padded_sizes,
    const api::GPUMemoryLayout memory_layout) {
  VK_CHECK_COND(padded_sizes.size() == 4);

  uint32_t N = api::utils::safe_downcast<uint32_t>(padded_sizes.at(0));
  uint32_t C = api::utils::safe_downcast<uint32_t>(padded_sizes.at(1));
  uint32_t H = api::utils::safe_downcast<uint32_t>(padded_sizes.at(2));
  uint32_t W = api::utils::safe_downcast<uint32_t>(padded_sizes.at(3));

  switch (memory_layout) {
    case api::kWidthPacked:
      VK_CHECK_COND(W % 4 == 0);
      W /= 4;
      break;
    case api::kHeightPacked:
      VK_CHECK_COND(H % 4 == 0);
      H /= 4;
      break;
    case api::kChannelsPacked:
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
    api::Context* const context,
    const std::vector<int64_t>& sizes,
    const api::ScalarType dtype,
    const api::StorageType storage_type,
    const api::GPUMemoryLayout memory_layout,
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
  if (storage_type != api::kBuffer) {
    texture_limits_.limits = api::utils::ivec3{
        api::utils::safe_downcast<int32_t>(storage_.image_extents_.data[0]),
        api::utils::safe_downcast<int32_t>(storage_.image_extents_.data[1]),
        api::utils::safe_downcast<int32_t>(storage_.image_extents_.data[2])};
  }

  if (dtype == api::kHalf) {
    VK_CHECK_COND(
        api::context()->adapter_ptr()->has_16bit_storage(),
        "Half dtype is only available if the physical device supports float16 "
        "storage buffers!");
  }
}

api::VulkanImage& vTensor::image(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage) & {
  storage_.transition(pipeline_barrier, stage, api::MemoryAccessType::READ);
  return storage_.image_;
}

api::VulkanImage& vTensor::image(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage,
    const api::MemoryAccessFlags access) & {
  storage_.transition(pipeline_barrier, stage, access);
  return storage_.image_;
}

api::VulkanBuffer& vTensor::buffer(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage) & {
  storage_.transition(pipeline_barrier, stage, api::MemoryAccessType::READ);
  return storage_.buffer_;
}

api::VulkanBuffer& vTensor::buffer(
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags stage,
    const api::MemoryAccessFlags access) & {
  storage_.transition(pipeline_barrier, stage, access);
  return storage_.buffer_;
}

const api::BufferBindInfo vTensor::sizes_ubo() {
  if (!sizes_uniform_.buffer()) {
    sizes_uniform_ = api::UniformParamsBuffer(
        storage_.context_, api::utils::make_whcn_ivec4(sizes_));
  }
  return api::BufferBindInfo(sizes_uniform_.buffer());
}

const api::BufferBindInfo vTensor::texture_limits_ubo() {
  if (!texture_limits_uniform_.buffer()) {
    texture_limits_uniform_ =
        api::UniformParamsBuffer(storage_.context_, texture_limits_);
  }
  return api::BufferBindInfo(texture_limits_uniform_.buffer());
}

const api::BufferBindInfo vTensor::texel_strides_ubo() {
  if (!texel_strides_uniform_.buffer()) {
    texel_strides_uniform_ = api::UniformParamsBuffer(
        storage_.context_,
        api::utils::make_whcn_ivec4(
            calculate_strides(padded_sizes_, memory_layout_)));
  }
  return api::BufferBindInfo(texel_strides_uniform_.buffer());
}

const api::BufferBindInfo vTensor::ntexels_ubo() {
  if (!ntexels_uniform_.buffer()) {
    ntexels_uniform_ =
        api::UniformParamsBuffer(storage_.context_, texel_numel());
  }
  return api::BufferBindInfo(ntexels_uniform_.buffer());
}

VmaAllocationCreateInfo vTensor::get_allocation_create_info() const {
  switch (storage_type()) {
    case api::kBuffer:
      return storage_.buffer_.allocation_create_info();
    case api::kTexture2D:
    case api::kTexture3D:
      return storage_.image_.allocation_create_info();
  }
  return {};
}

VkMemoryRequirements vTensor::get_memory_requirements() const {
  switch (storage_type()) {
    case api::kBuffer:
      return storage_.buffer_.get_memory_requirements();
    case api::kTexture2D:
    case api::kTexture3D:
      return storage_.image_.get_memory_requirements();
  }
  return {};
}

void vTensor::bind_allocation(const api::Allocation& allocation) {
  switch (storage_type()) {
    case api::kBuffer:
      storage_.buffer_.bind_allocation(allocation);
      break;
    case api::kTexture2D:
    case api::kTexture3D:
      storage_.image_.bind_allocation(allocation);
      break;
  }
}

void vTensor::update_size_metadata(const std::vector<int64_t>& new_sizes) {
  sizes_ = new_sizes;
  padded_sizes_ = calculate_padded_sizes(sizes_, memory_layout_);

  // Calculate the extents of the image texture that would have been required
  // for a tensor of the new sizes.
  api::utils::uvec3 virtual_extents =
      calculate_image_extents(padded_sizes_, memory_layout_);

  // Update the texture limits to reflect the new virtual extents.
  texture_limits_.limits = api::utils::ivec3{
      api::utils::safe_downcast<int32_t>(virtual_extents.data[0]),
      api::utils::safe_downcast<int32_t>(virtual_extents.data[1]),
      api::utils::safe_downcast<int32_t>(virtual_extents.data[2])};

  if (sizes_uniform_.buffer()) {
    sizes_uniform_.update(api::utils::make_whcn_ivec4(sizes_));
  }
  if (texture_limits_uniform_.buffer()) {
    texture_limits_uniform_.update(texture_limits_);
  }
  if (texel_strides_uniform_.buffer()) {
    texel_strides_uniform_.update(api::utils::make_whcn_ivec4(
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
  if (storage_type() != api::kBuffer) {
    // For texture storage check that the current texture is large enough for
    // the new sizes of the tensor.
    api::utils::uvec3 virtual_extents =
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

api::VulkanImage allocate_image(
    api::Context* const context_ptr,
    api::utils::uvec3& image_extents,
    const api::StorageType storage_type,
    const VkFormat image_format,
    const bool allocate_memory) {
  api::Adapter* adapter_ptr = context_ptr->adapter_ptr();

  api::ImageSampler::Properties sampler_props{
      VK_FILTER_NEAREST,
      VK_SAMPLER_MIPMAP_MODE_NEAREST,
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
  };

  VkImageType image_type = VK_IMAGE_TYPE_3D;
  VkImageViewType image_view_type;

  switch (storage_type) {
    case api::kTexture3D:
      image_type = VK_IMAGE_TYPE_3D;
      image_view_type = VK_IMAGE_VIEW_TYPE_3D;
      break;
    case api::kTexture2D:
      image_type = VK_IMAGE_TYPE_2D;
      image_view_type = VK_IMAGE_VIEW_TYPE_2D;
      break;
    default:
      // Return an empty VulkanImage by default
      return api::VulkanImage();
  }

  VkSampler sampler = adapter_ptr->sampler_cache().retrieve(sampler_props);

  return adapter_ptr->vma().create_image(
      api::create_extent3d(image_extents),
      image_format,
      image_type,
      image_view_type,
      sampler_props,
      sampler,
      /*allow_transfer = */ true,
      /*allocate_memory = */ allocate_memory);
}

api::VulkanBuffer allocate_buffer(
    api::Context* const context_ptr,
    const int64_t numel,
    const api::StorageType storage_type,
    const api::ScalarType dtype,
    const bool allocate_memory) {
  api::Adapter* adapter_ptr = context_ptr->adapter_ptr();

  switch (storage_type) {
    case api::kBuffer:
      break;
    default:
      // Return an empty VulkanBuffer if Buffer storage is not used
      return api::VulkanBuffer();
  }

  return adapter_ptr->vma().create_storage_buffer(
      api::element_size(dtype) * numel, /*gpu_only = */ true, allocate_memory);
}

vTensorStorage::vTensorStorage(
    api::Context* const context,
    const api::StorageType storage_type,
    const api::GPUMemoryLayout gpu_memory_layout,
    const std::vector<int64_t>& padded_sizes,
    const api::ScalarType dtype,
    const bool allocate_memory)
    : context_(context),
      storage_type_{storage_type},
      image_extents_(calculate_image_extents(padded_sizes, gpu_memory_layout)),
      buffer_length_{api::utils::multiply_integers(padded_sizes)},
      image_(allocate_image(
          context_,
          image_extents_,
          storage_type_,
          api::to_vkformat(dtype),
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
    api::PipelineBarrier& pipeline_barrier,
    const api::PipelineStageFlags cur_stage,
    const api::MemoryAccessFlags cur_access) {
  // Get last stage access
  api::PipelineStageFlags prev_stage = last_access_.stage;
  api::MemoryAccessFlags prev_access = last_access_.access;

  const bool prev_written = (prev_access & api::MemoryAccessType::WRITE) != 0;

  VkImageLayout cur_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout new_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  bool layout_changed = false;
  if (image_) {
    cur_layout = image_.layout();
    new_layout = api::vk_layout(cur_stage, cur_access);

    layout_changed = cur_layout != new_layout;
  }

  if (prev_written || layout_changed) {
    VkPipelineStageFlags src_stage = api::vk_stage(prev_stage);
    if (0u == src_stage) {
      src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
    VkPipelineStageFlags dst_stage = api::vk_stage(cur_stage);
    if (0u == dst_stage) {
      dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    pipeline_barrier.stage.src |= src_stage;
    pipeline_barrier.stage.dst |= dst_stage;

    if (image_) {
      pipeline_barrier.images.emplace_back(
          api::vk_access(prev_stage, prev_access),
          api::vk_access(cur_stage, cur_access),
          cur_layout,
          new_layout,
          image_);

      image_.set_layout(new_layout);
    } else if (buffer_) {
      pipeline_barrier.buffers.emplace_back(
          api::vk_access(prev_stage, prev_access),
          api::vk_access(cur_stage, cur_access),
          buffer_);
    }
  }

  last_access_.stage = cur_stage;
  last_access_.access = cur_access;
}

void vTensorStorage::discard_and_reallocate(
    const std::vector<int64_t>& padded_sizes,
    const api::GPUMemoryLayout gpu_memory_layout,
    const api::ScalarType dtype) {
  const bool image_owns_memory = image_.owns_memory();
  const bool buffer_owns_memory = buffer_.owns_memory();

  flush();

  image_extents_ = calculate_image_extents(padded_sizes, gpu_memory_layout);
  image_ = allocate_image(
      context_,
      image_extents_,
      storage_type_,
      api::to_vkformat(dtype),
      image_owns_memory);

  buffer_length_ = api::utils::multiply_integers(padded_sizes);
  buffer_ = allocate_buffer(
      context_, buffer_length_, storage_type_, dtype, buffer_owns_memory);
}

} // namespace vkcompute
