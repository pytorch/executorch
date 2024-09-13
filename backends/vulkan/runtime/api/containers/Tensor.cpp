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

/*
 * Given the strides of a buffer-backed tensor, estimate the equivalent memory
 * layout enum value by identifying the fastest moving dimension.
 */
utils::GPUMemoryLayout estimate_memory_layout(
    const std::vector<int64_t>& dim_order) {
  int64_t fastest_dim_whcn = dim_order.size() - 1 - dim_order.back();
  if (fastest_dim_whcn >= 0 && fastest_dim_whcn < 3) {
    return utils::GPUMemoryLayout(fastest_dim_whcn);
  }

  // TODO(ssjia) find a way to gracefully recover from this case by i.e. adding
  // a UNKOWN GPUMemoryLayout. This is not high priority though because we don't
  // expect this to ever come up in practice.
  VK_THROW("No compatible GPUMemoryLayout value");
}

std::vector<int64_t> calculate_dim_order(
    const size_t ndim,
    const utils::GPUMemoryLayout memory_layout) {
  // Special case for zero dim tensors
  if (ndim == 0) {
    return {0};
  }
  std::vector<int64_t> dim_order(ndim);
  int64_t last_dim =
      ndim - utils::to_packed_dim_nchw_offset<int64_t>(memory_layout);

  int64_t cur_dim = 0;
  for (int d = 0; d < ndim; ++d) {
    if (d == last_dim) {
      cur_dim++;
    }
    dim_order[d] = cur_dim;
    cur_dim++;
  }
  if (last_dim >= 0) {
    dim_order[ndim - 1] = last_dim;
  }

  return dim_order;
}

std::vector<int64_t> calculate_strides(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& dim_order) {
  // For zero dim tensors
  if (sizes.size() == 0) {
    return {1};
  }

  size_t ndim = sizes.size();
  std::vector<int64_t> strides(ndim);

  strides[dim_order[ndim - 1]] = 1;
  for (int32_t i = ndim - 2; i >= 0; --i) {
    if (sizes[dim_order[i + 1]] == 0) {
      strides[dim_order[i]] = strides[dim_order[i + 1]];
    } else {
      strides[dim_order[i]] =
          strides[dim_order[i + 1]] * sizes[dim_order[i + 1]];
    }
  }

  return strides;
}

/*
 * Axis mapping is somewhat analogous to strides for texture backed tensors.
 *
 * The axis mapping is normalized to 4 dimensions, similar to the padded sizes.
 * The first 3 values of the axis mapping indicate the (X,Y,Z) image texture
 * axis that corresponds to the width, height, and channels dimension of the
 * tensor. Thus the axis mapping can be considered to be in WHCN dimension
 * order.
 *
 * The last value `axis_map.at(3)` indicates the WHCN index of the tensor
 * dimension along which batches will be concatenated. This dimension can be
 * referred to as the "inner dimension" To determine which image texture axis is
 * used for the concatenation, a double lookup will need to be performed
 * (axis_map.at(axis_map.at(3))).
 *
 * The reason for strucuring axis mapping this way is because for the batch dim,
 * two things need to be easily derived:
 *
 * 1. The dim idx of the inner dimension, so that the size of the inner
 *    dimension can be easily determined.
 * 2. The texture axis used to concatenate batches
 *
 * By storing the dim index of the inner dimension instead of the texture axis
 * it maps to, both pieces of information are readily available.
 *
 * The axis mapping allows for permuted views of texture-backed tensors.
 */
std::vector<int64_t> default_axis_map() {
  // Currently, all compute shaders have an assumption that the channels dim is
  // used to combine with the batch dim of a tensor. However, once dim mapping
  // is integrated into the tensor indexing logic for each compute shader, we
  // can be more flexible with mapping the batch dim to different texture axes
  // in order to improve performance or memory footprint.
  return {0, 1, 2, 2};
}

bool dim_order_is_valid(const std::vector<int64_t>& dim_order) {
  int64_t sum = 0;
  for (size_t i = 0; i < dim_order.size(); ++i) {
    if (dim_order[i] < 0 || dim_order[i] >= dim_order.size()) {
      return false;
    }
    sum += dim_order[i];
  }
  int64_t n = static_cast<int64_t>(dim_order.size() - 1);
  // Sanity check that the sum of the indices in the vector is equal to the sum
  // of 0 + 1 + 2 + ... + (ndim - 1)
  return sum == n * (n + 1) / 2;
}

std::vector<int64_t> unsqueeze_strides(
    const std::vector<int64_t>& strides,
    const int64_t numel) {
  const size_t ndim = strides.size();
  const size_t ndim_up4 = utils::align_up_4(strides.size());
  std::vector<int64_t> unsqueezed_strides(ndim_up4);
  for (int32_t i = 1; i <= ndim; ++i) {
    int64_t dim_stride = strides.at(ndim - i);
    unsqueezed_strides.at(ndim_up4 - i) = dim_stride;
  }

  for (int32_t i = ndim + 1; i <= ndim_up4; ++i) {
    unsqueezed_strides.at(ndim_up4 - i) = numel;
  }
  return unsqueezed_strides;
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
    const std::vector<int64_t>& axis_map,
    const utils::GPUMemoryLayout memory_layout) {
  VK_CHECK_COND(padded_sizes.size() == 4);
  VK_CHECK_COND(axis_map.size() == 4);

  utils::uvec3 extents({1, 1, 1});
  // First three elements of axis_map indicate which (X,Y,Z) image axis the
  // width, height, and channels dim of the tensor maps to.
  for (int whcn_dim = 0; whcn_dim < 3; ++whcn_dim) {
    const int64_t axis = axis_map.at(whcn_dim);
    const int64_t dim = padded_sizes.size() - 1 - whcn_dim;
    extents[axis] = utils::safe_downcast<uint32_t>(padded_sizes.at(dim));
  }

  // axis_map[3] indicates the WHCN index of the dimension used for batch
  // concatenation. Thus a double lookup is required to determine the image axis
  // used for batch concatenation.
  const int64_t concatted_whcn_dim = axis_map.at(3);
  const int64_t batch_axis = axis_map.at(concatted_whcn_dim);
  // Multiply the extents of the batch axis by the batch size.
  extents[batch_axis] *= padded_sizes.at(0);

  switch (memory_layout) {
    case utils::kWidthPacked:
      VK_CHECK_COND(extents[axis_map.at(0)] % 4 == 0);
      extents[axis_map.at(0)] /= 4;
      break;
    case utils::kHeightPacked:
      VK_CHECK_COND(extents[axis_map.at(1)] % 4 == 0);
      extents[axis_map.at(1)] /= 4;
      break;
    case utils::kChannelsPacked:
      VK_CHECK_COND(extents[axis_map.at(2)] % 4 == 0);
      extents[axis_map.at(2)] /= 4;
      break;
  }

  return extents;
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
      // Calculate tensor metadata
      sizes_(sizes.begin(), sizes.end()),
      dim_order_(calculate_dim_order(sizes_.size(), memory_layout_)),
      axis_map_(default_axis_map()),
      strides_(calculate_strides(sizes, dim_order_)),
      numel_(utils::multiply_integers(sizes_)),
      padded_sizes_{calculate_padded_sizes(sizes, memory_layout_)},
      unsqueezed_strides_{unsqueeze_strides(strides_, numel_)},
      padded_numel_(utils::multiply_integers(padded_sizes_)),
      texture_limits_{{0, 0, 0}},
      logical_limits_{{0, 0, 0}},
      // Utility Uniform Buffers that can be passed to shaders as arguments
      sizes_uniform_(),
      strides_uniform_(),
      numel_uniform_(),
      axis_map_uniform_(),
      texture_limits_uniform_(),
      logical_limits_uniform_(),
      // Construct Tensor storage
      storage_(
          context,
          storage_type,
          memory_layout_,
          axis_map_,
          padded_sizes_,
          dtype_,
          allocate_memory) {
  VK_CHECK_COND(
      dim_order_is_valid(dim_order_), "computed dim order is invalid");

  if (storage_type != utils::kBuffer) {
    texture_limits_.limits = utils::ivec3{
        utils::safe_downcast<int32_t>(storage_.image_extents_[0]),
        utils::safe_downcast<int32_t>(storage_.image_extents_[1]),
        utils::safe_downcast<int32_t>(storage_.image_extents_[2])};

    update_logical_limits();
  }

  if (dtype == vkapi::kHalf) {
    VK_CHECK_COND(
        api::context()->adapter_ptr()->has_16bit_storage(),
        "Half dtype is only available if the physical device supports float16 "
        "storage buffers!");
  }
}

vTensor::vTensor(const vTensor& other)
    : dtype_(other.dtype_),
      memory_layout_(other.memory_layout_),
      // Copy tensor size metadata
      sizes_(other.sizes_.begin(), other.sizes_.end()),
      dim_order_(other.dim_order_.begin(), other.dim_order_.end()),
      axis_map_(other.axis_map_.begin(), other.axis_map_.end()),
      strides_(other.strides_.begin(), other.strides_.end()),
      numel_(other.numel_),
      padded_sizes_{other.padded_sizes_.begin(), other.padded_sizes_.end()},
      unsqueezed_strides_{
          other.unsqueezed_strides_.begin(),
          other.unsqueezed_strides_.end()},
      padded_numel_(other.padded_numel_),
      texture_limits_{other.texture_limits_},
      logical_limits_{other.logical_limits_},
      // Empty initialize Utility Uniform Buffers
      sizes_uniform_(),
      strides_uniform_(),
      numel_uniform_(),
      axis_map_uniform_(),
      texture_limits_uniform_(),
      logical_limits_uniform_(),
      // Copy Tensor storage
      storage_(other.storage_) {}

vTensor::vTensor(
    const vTensor& other,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& dim_order,
    const int64_t offset_numel)
    : dtype_(other.dtype_),
      memory_layout_(estimate_memory_layout(dim_order)),
      // Copy tensor size metadata
      sizes_(sizes.begin(), sizes.end()),
      dim_order_(dim_order.begin(), dim_order.end()),
      axis_map_(default_axis_map()),
      strides_(calculate_strides(sizes_, dim_order_)),
      numel_(utils::multiply_integers(sizes_)),
      padded_sizes_{calculate_padded_sizes(sizes, memory_layout_)},
      unsqueezed_strides_{unsqueeze_strides(strides_, numel_)},
      padded_numel_(utils::multiply_integers(padded_sizes_)),
      texture_limits_{other.texture_limits_},
      logical_limits_(other.logical_limits_),
      // Empty initialize Utility Uniform Buffers
      sizes_uniform_(),
      strides_uniform_(),
      numel_uniform_(),
      axis_map_uniform_(),
      texture_limits_uniform_(),
      logical_limits_uniform_(),
      // Copy Tensor storage
      storage_(other.storage_, vkapi::element_size(dtype_) * offset_numel) {
  VK_CHECK_COND(
      dim_order_is_valid(dim_order_), "new dim order provided is invalid");
  VK_CHECK_COND(
      offset_numel + numel_ <= other.numel(),
      "Tensor alias cannot access more elements than available in the original"
      "tensor");
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

void vTensor::update_logical_limits() {
  logical_limits_.limits[0] = texture_limits_.limits[axis_map_.at(0)];
  logical_limits_.limits[1] = texture_limits_.limits[axis_map_.at(1)];
  logical_limits_.limits[2] = texture_limits_.limits[axis_map_.at(2)];
}

utils::uvec3 vTensor::logical_extents() const {
  utils::uvec3 logical_extents(
      {utils::safe_downcast<uint32_t>(logical_limits_.limits[0]),
       utils::safe_downcast<uint32_t>(logical_limits_.limits[1]),
       utils::safe_downcast<uint32_t>(logical_limits_.limits[2])});
  return logical_extents;
}

const vkapi::BufferBindInfo vTensor::sizes_ubo() {
  if (!sizes_uniform_.buffer()) {
    sizes_uniform_ =
        ParamsBuffer(storage_.context_, utils::make_whcn_ivec4(sizes_));
  }
  return vkapi::BufferBindInfo(sizes_uniform_.buffer());
}

const vkapi::BufferBindInfo vTensor::strides_ubo() {
  if (!strides_uniform_.buffer()) {
    strides_uniform_ = ParamsBuffer(
        storage_.context_, utils::make_whcn_ivec4(unsqueezed_strides_));
  }
  return vkapi::BufferBindInfo(strides_uniform_.buffer());
}

const vkapi::BufferBindInfo vTensor::axis_map_ubo() {
  if (!axis_map_uniform_.buffer()) {
    axis_map_uniform_ =
        ParamsBuffer(storage_.context_, utils::make_ivec4(axis_map_));
  }
  return vkapi::BufferBindInfo(axis_map_uniform_.buffer());
}

const vkapi::BufferBindInfo vTensor::texture_limits_ubo() {
  if (!texture_limits_uniform_.buffer()) {
    texture_limits_uniform_ = ParamsBuffer(storage_.context_, texture_limits_);
  }
  return vkapi::BufferBindInfo(texture_limits_uniform_.buffer());
}

const vkapi::BufferBindInfo vTensor::logical_limits_ubo() {
  if (!logical_limits_uniform_.buffer()) {
    logical_limits_uniform_ = ParamsBuffer(storage_.context_, logical_limits_);
  }
  return vkapi::BufferBindInfo(logical_limits_uniform_.buffer());
}

const vkapi::BufferBindInfo vTensor::numel_ubo() {
  if (!numel_uniform_.buffer()) {
    numel_uniform_ = ParamsBuffer(storage_.context_, numel_);
  }
  return vkapi::BufferBindInfo(numel_uniform_.buffer());
}

size_t vTensor::staging_buffer_numel() const {
  const bool is_int8 = dtype_ == vkapi::kChar;
  const bool int8_supported =
      storage_.context_->adapter_ptr()->has_full_int8_buffers_support();
  if (is_int8 && !int8_supported) {
    return utils::align_up_4(numel_);
  }
  if (storage_type() == utils::kBuffer) {
    return numel_;
  }
  return padded_numel_;
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

void vTensor::update_metadata() {
  strides_ = calculate_strides(sizes_, dim_order_);
  // Only update the memory layout for buffer-backed tensors. Strides are
  // meaningless for texture-backed tensors and do not impact the memory layout.
  if (storage_type() == utils::kBuffer) {
    memory_layout_ = estimate_memory_layout(dim_order_);
  }
  numel_ = utils::multiply_integers(sizes_);

  padded_sizes_ = calculate_padded_sizes(sizes_, memory_layout_);
  unsqueezed_strides_ = unsqueeze_strides(strides_, numel_);
  padded_numel_ = utils::multiply_integers(padded_sizes_);

  // Calculate the extents of the image texture that would have been required
  // for a tensor of the new sizes.
  utils::uvec3 virtual_extents =
      calculate_image_extents(padded_sizes_, axis_map_, memory_layout_);

  // Update the texture limits to reflect the new virtual extents.
  texture_limits_.limits = utils::ivec3{
      utils::safe_downcast<int32_t>(virtual_extents[0]),
      utils::safe_downcast<int32_t>(virtual_extents[1]),
      utils::safe_downcast<int32_t>(virtual_extents[2])};

  update_logical_limits();

  if (sizes_uniform_.buffer()) {
    sizes_uniform_.update(utils::make_whcn_ivec4(sizes_));
  }
  if (strides_uniform_.buffer()) {
    strides_uniform_.update(utils::make_whcn_ivec4(unsqueezed_strides_));
  }
  if (numel_uniform_.buffer()) {
    numel_uniform_.update(numel_);
  }
  if (axis_map_uniform_.buffer()) {
    axis_map_uniform_.update(utils::make_ivec4(axis_map_));
  }
  if (texture_limits_uniform_.buffer()) {
    texture_limits_uniform_.update(texture_limits_);
  }
  if (logical_limits_uniform_.buffer()) {
    logical_limits_uniform_.update(logical_limits_);
  }
}

void vTensor::check_sizes(const std::vector<int64_t>& sizes) const {
  if (storage_type() != utils::kBuffer) {
    // For texture storage check that the current texture is large enough for
    // the new sizes of the tensor.
    utils::uvec3 virtual_extents =
        calculate_image_extents(padded_sizes_, axis_map_, memory_layout_);

    bool valid_resize = virtual_extents[0] <= image_extents()[0];
    valid_resize = valid_resize && virtual_extents[1] <= image_extents()[1];
    valid_resize = valid_resize && virtual_extents[2] <= image_extents()[2];

    VK_CHECK_COND(
        valid_resize,
        "tensor sizes requires a larger texture than the current one.");
  } else {
    // For buffer storage check that the current buffer is large enough for the
    // new sizes of the tensor.
    int64_t numel = utils::multiply_integers(sizes);
    bool valid_resize =
        numel + storage_.buffer_offset_ <= storage_.buffer_length_;
    VK_CHECK_COND(
        valid_resize,
        "tensor sizes requires a larger buffer than the current one.");
  }
}

void vTensor::virtual_reconfigure(
    const std::vector<int64_t>& new_sizes,
    const std::vector<int64_t>& new_dim_order) {
  VK_CHECK_COND(
      storage_type() == utils::kBuffer,
      "virtual_reconfigure is only applicable for buffer backed tensors");
  VK_CHECK_COND(new_sizes.size() == new_dim_order.size());
  VK_CHECK_COND(dim_order_is_valid(new_dim_order));

  check_sizes(new_sizes);
  sizes_ = new_sizes;
  dim_order_ = new_dim_order;
  update_metadata();
}

void vTensor::virtual_resize(const std::vector<int64_t>& new_sizes) {
  VK_CHECK_COND(
      new_sizes.size() == dim_order_.size(),
      "new sizes cannot modify the dimensionality of the tensor ");

  check_sizes(new_sizes);
  sizes_ = new_sizes;
  update_metadata();
}

void vTensor::reallocate(const std::vector<int64_t>& new_sizes) {
  sizes_ = new_sizes;
  update_metadata();
  storage_.discard_and_reallocate(
      calculate_padded_sizes(new_sizes, memory_layout_),
      axis_map_,
      memory_layout_,
      dtype_);
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
      element_size(dtype) * numel, allocate_memory);
}

vTensorStorage::vTensorStorage(
    Context* const context,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout gpu_memory_layout,
    const std::vector<int64_t>& axis_map,
    const std::vector<int64_t>& padded_sizes,
    const vkapi::ScalarType dtype,
    const bool allocate_memory)
    : context_(context),
      storage_type_{storage_type},
      image_extents_(
          calculate_image_extents(padded_sizes, axis_map, gpu_memory_layout)),
      buffer_length_{utils::multiply_integers(padded_sizes)},
      buffer_offset_{0},
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

vTensorStorage::vTensorStorage(
    const vTensorStorage& other,
    const int64_t buffer_offset)
    : context_(other.context_),
      storage_type_{other.storage_type_},
      image_extents_(other.image_extents_),
      buffer_length_{other.buffer_length_},
      buffer_offset_{buffer_offset},
      image_(other.image_),
      buffer_(other.buffer_, buffer_offset),
      last_access_{other.last_access_} {}

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

bool vTensorStorage::is_copy_of(const vTensorStorage& other) const {
  if (storage_type_ == utils::kBuffer) {
    return buffer_.is_copy_of(other.buffer_);
  }
  return image_.is_copy_of(other.image_);
}

void vTensorStorage::discard_and_reallocate(
    const std::vector<int64_t>& padded_sizes,
    const std::vector<int64_t>& axis_map,
    const utils::GPUMemoryLayout gpu_memory_layout,
    const vkapi::ScalarType dtype) {
  const bool image_owns_memory = image_.owns_memory();
  const bool buffer_owns_memory = buffer_.owns_memory();

  flush();

  image_extents_ =
      calculate_image_extents(padded_sizes, axis_map, gpu_memory_layout);
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
