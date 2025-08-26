/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/api/containers/Tensor.h>
#include <algorithm>
#include <cassert>
#include <cstring>

namespace vkcompute {
namespace api {

/*
 * Used to infer the sizes of a tensor that would correspond to a given
 * VulkanImage.
 */
std::vector<int64_t> calculate_sizes(
    const vkapi::VulkanImage& image,
    const utils::GPUMemoryLayout memory_layout) {
  auto sizes = std::vector<int64_t>{
      image.extents().width, image.extents().height, image.extents().depth};
  const auto packed_dim = utils::to_packed_dim<int32_t>(memory_layout);
  sizes.at(packed_dim) *= 4;
  return sizes;
}

std::vector<int64_t> calculate_dim_order(
    const size_t ndim,
    const int32_t packed_dim) {
  // Special case for zero dim tensors
  if (ndim == 0) {
    return {0};
  }
  std::vector<int64_t> dim_order(ndim);
  // Explicitly convert ndim to signed to prevent underflow
  int64_t last_dim = int64_t(ndim) - 1 - packed_dim;

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
std::vector<int64_t> calculate_axis_map(
    const std::vector<int64_t>& sizes,
    utils::AxisMapLayout axis_map_layout) {
  if (axis_map_layout == utils::AxisMapLayout::OPTIMIZED) {
    std::vector<int64_t> axis_map(sizes.size() + 1);
    std::iota(axis_map.begin(), axis_map.end() - 1, 0);

    std::stable_sort(
        axis_map.begin(), axis_map.end() - 1, [&sizes](size_t i1, size_t i2) {
          return sizes[i1] < sizes[i2];
        });

    assert(axis_map.size() > 0);
    // Find the index of the channel dimension
    for (size_t i = 0; i < axis_map.size() - 1; ++i) {
      assert(sizes.size() > axis_map[i]);
      if (sizes[axis_map[i]] == 2) {
        axis_map.back() = i;
        break;
      }
    }

    return axis_map;
  }
  // default
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

utils::ivec4 flip_and_unsqueeze_ivec4(
    const std::vector<int64_t>& tensor_metadata,
    const vTensor::Attribute metadata_type,
    const size_t numel) {
  VK_CHECK_COND(tensor_metadata.size() <= 4);
  std::vector<int32_t> flipped_metadata =
      flip_and_unsqueeze<int32_t>(tensor_metadata, metadata_type, numel);
  return {
      flipped_metadata.at(0),
      flipped_metadata.at(1),
      flipped_metadata.at(2),
      flipped_metadata.at(3),
  };
}

std::vector<int64_t> calculate_padded_sizes(
    const std::vector<int64_t>& sizes,
    const int32_t packed_dim) {
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
  const int64_t dim_offset = packed_dim + 1;
  const int64_t padded_dim_size = utils::val_at(-dim_offset, sizes);
  padded_sizes.at(ndim_up4 - dim_offset) = utils::align_up_4(padded_dim_size);

  return padded_sizes;
}

utils::uvec3 calculate_image_extents(
    const std::vector<int64_t>& padded_sizes,
    const std::vector<int64_t>& axis_map,
    const int32_t packed_dim) {
  utils::uvec3 extents({1, 1, 1});

  // For high dimensional tensors, buffer storage must be used. No need to
  // compute image extents in this case.
  if (padded_sizes.size() > 4) {
    return extents;
  }

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

  VK_CHECK_COND(extents[axis_map.at(packed_dim)] % 4 == 0);
  extents[axis_map.at(packed_dim)] /= 4;
  return extents;
}

/*
 * The physical image extents describe the size of an allocated texture resource
 * i.e. how many texels in the width, height and depth axis of the image.
 * However, the axis map allows a tensor logical dimension to map to a different
 * physical texture axis; in essence, it describes a permutation between the
 * logical width, height, channels, etc. dimensions of a tensor and the width,
 * height, depth axis of a texture.
 *
 * The "logical extents" is simply the physical image extents permuted by the
 * axis mapping. The logical extents is useful for constructing global work
 * group sizes, so that it is easier to convert the global thread ID to a
 * tensor index.
 */
utils::uvec3 calculate_logical_limits(
    const utils::uvec3& image_extents,
    const std::vector<int64_t>& axis_map) {
  return {
      image_extents[axis_map.at(0)],
      image_extents[axis_map.at(1)],
      image_extents[axis_map.at(2)],
  };
}

/*
 * Convenience overload of the above function to calculate logical limits
 * directly from tensor sizes.
 */
utils::uvec3 calculate_logical_limits(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& axis_map,
    const int32_t packed_dim) {
  return calculate_logical_limits(
      calculate_image_extents(
          calculate_padded_sizes(sizes, packed_dim), axis_map, packed_dim),
      axis_map);
}

int64_t calculate_gpu_buffer_numel(
    Context* const context,
    const std::vector<int64_t>& sizes,
    const utils::uvec3 image_extents,
    const utils::StorageType storage_type,
    const vkapi::ScalarType dtype) {
  // For texture backed tensors, simply multiply the total number of texels by 4
  if (storage_type != utils::kBuffer) {
    return image_extents[0] * image_extents[1] * image_extents[2] * 4;
  }
  const bool is_int8 = dtype == vkapi::kChar;
  const bool int8_supported =
      context->adapter_ptr()->has_full_int8_buffers_support();
  const size_t numel = utils::multiply_integers(sizes);
  // For int8 tensors, if the device does not support int8 buffers, then int32
  // is used instead to represent the buffer data. Therefore the number of
  // elements in the buffer is aligned to the next multiple of 4.
  if (is_int8 && int8_supported) {
    return utils::align_up_4(numel);
  }
  return numel;
}

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
int32_t pack_into_int32(const std::vector<T>& vec, const int32_t extra) {
  int32_t packed = static_cast<int32_t>(
      vec.at(0) + (vec.at(1) << 4) + (vec.at(2) << 8) + (vec.at(3) << 12) +
      (extra << 16));
  return packed;
}

int32_t create_hashed_layout(
    const std::vector<int64_t>& dim_order,
    const std::vector<int64_t>& axis_map,
    const int32_t packed_dim,
    const utils::StorageType storage_type) {
  if (storage_type == utils::kBuffer) {
    return pack_into_int32(
        flip_and_unsqueeze<int64_t>(dim_order, kTensorDimOrder, 0), 0);
  }
  return pack_into_int32(axis_map, packed_dim);
}

size_t calculate_max_ubo_nbytes(
    const size_t min_nbytes_per_ubo,
    const utils::StorageType storage_type) {
  size_t ivec4_ubo_nbytes = utils::align_up(size_t(16), min_nbytes_per_ubo);
  size_t uvec3_ubo_nbytes = utils::align_up(size_t(12), min_nbytes_per_ubo);
  size_t int32_ubo_nbytes = utils::align_up(size_t(4), min_nbytes_per_ubo);
  if (storage_type == utils::kBuffer) {
    // sizes, strides, dim order, numel
    return 3 * ivec4_ubo_nbytes + int32_ubo_nbytes;
  }
  // sizes, logical limits
  return ivec4_ubo_nbytes + uvec3_ubo_nbytes;
}

//
// vTensorStorage
//

utils::StorageType storage_type(const vkapi::VulkanImage& image) {
  const auto type = image.type();
  switch (type) {
    case VK_IMAGE_TYPE_3D:
      return utils::kTexture3D;
    case VK_IMAGE_TYPE_2D:
      return utils::kTexture2D;
    default:
      VK_THROW("Unsupported image type", type);
  }
}

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

    // TODO(ssjia): change to always check that the image extents do not exceed
    // physical limits. Adding the check now based on `maxImageDimension3D` will
    // cause some existing models to break. Anecdotally, on Adreno and
    // SwiftShader devices, using 3D textures that exceed `maxImageDimension3D`
    // appears to be ok. So we need to figure out if is it undefined behaviour
    // or if there's a better way to figure out what the limit is. For now, only
    // check during debug build so that we can detect when exceeding physical
    // limits could be a potential cause for model outputs to be wrong. In the
    // meantime, the threshold for using texture storage can be configured at
    // export time.
#ifdef VULKAN_DEBUG
  uint32_t max_extent = storage_type == utils::kTexture3D
      ? adapter_ptr->max_texture3d_dim()
      : adapter_ptr->max_texture2d_dim();

  VK_CHECK_COND(
      image_extents[0] <= max_extent && image_extents[1] <= max_extent &&
      image_extents[2] <= max_extent);
#endif

  VkSampler sampler = adapter_ptr->sampler_cache().retrieve(sampler_props);

  return adapter_ptr->vma().create_image(
      context_ptr->device(),
      vkapi::create_extent3d(image_extents),
      image_format,
      image_type,
      context_ptr->preferred_image_tiling(),
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

  VK_CHECK_COND(numel <= context_ptr->adapter_ptr()->max_buffer_numel());

  return adapter_ptr->vma().create_storage_buffer(
      element_size(dtype) * numel, allocate_memory);
}

vTensorStorage::vTensorStorage(
    Context* const context,
    const utils::StorageType storage_type,
    const std::vector<int64_t>& axis_map,
    const int32_t packed_dim,
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    const bool allocate_memory)
    : context_(context),
      storage_type_{storage_type},
      image_extents_(calculate_image_extents(
          calculate_padded_sizes(sizes, packed_dim),
          axis_map,
          packed_dim)),
      buffer_length_{calculate_gpu_buffer_numel(
          context_,
          sizes,
          image_extents_,
          storage_type,
          dtype)},
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
    Context* const context,
    const vkapi::VulkanImage& image)
    : context_(context),
      storage_type_{storage_type(image)},
      image_extents_(
          {image.extents().width,
           image.extents().height,
           image.extents().depth}),
      buffer_length_{0},
      buffer_offset_{0},
      image_(image),
      buffer_(vkapi::VulkanBuffer()),
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
  const bool cur_written = (cur_access & vkapi::MemoryAccessType::WRITE) != 0;

  VkImageLayout cur_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout new_layout = VK_IMAGE_LAYOUT_UNDEFINED;
  bool layout_changed = false;
  if (image_) {
    cur_layout = image_.layout();
    new_layout = vkapi::vk_layout(cur_stage, cur_access);

    layout_changed = cur_layout != new_layout;
  }

  // RAW: need to make sure current read sees previous writes
  // WAW: need to make sure the current write occurs after previous write so
  //      the final value is correct.
  // WAR: need to make sure previous read does not read the value from the
  //      current write.
  // RAR: no need for synchronization
  if (prev_written || cur_written || layout_changed) {
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

//
// vTensor
//

vTensor::vTensor(
    Context* const context,
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout memory_layout,
    const bool allocate_memory,
    const utils::AxisMapLayout axis_map_layout)
    : dtype_(dtype),
      // Calculate tensor metadata
      sizes_(sizes.begin(), sizes.end()),
      packed_dim_(utils::to_packed_dim<int32_t>(memory_layout)),
      dim_order_(calculate_dim_order(sizes_.size(), packed_dim_)),
      axis_map_(calculate_axis_map(sizes_, axis_map_layout)),
      strides_(calculate_strides(sizes, dim_order_)),
      numel_(utils::multiply_integers(sizes_)),
      hashed_layout_(create_hashed_layout(
          dim_order_,
          axis_map_,
          packed_dim_,
          storage_type)),
      // Related to tensor metadata UBOs
      min_nbytes_per_ubo_{context->adapter_ptr()->min_ubo_alignment()},
      max_ubo_nbytes_{
          calculate_max_ubo_nbytes(min_nbytes_per_ubo_, storage_type)},
      uniforms_(),
      buffer_meta_(),
      // Construct Tensor storage
      storage_(std::make_shared<vTensorStorage>(
          context,
          storage_type,
          axis_map_,
          packed_dim_,
          sizes,
          dtype_,
          allocate_memory)) {
  // uniform_data_ only valid for low dim tensors
  if (sizes.size() <= 4) {
    uniform_data_ = std::make_shared<UniformData>(UniformData{
        numel_,
        sizes_,
        dim_order_,
        strides_,
        calculate_logical_limits(storage_->image_extents_, axis_map_)});
  }

  VK_CHECK_COND(
      dim_order_is_valid(dim_order_), "computed dim order is invalid");
}

// NOLINTNEXTLINE
vTensor::vTensor(
    Context* context,
    const vkapi::VulkanImage& image,
    const utils::GPUMemoryLayout memory_layout,
    const utils::AxisMapLayout axis_map_layout)
    : dtype_(vkapi::element_scalartype(image.format())),
      // Calculate tensor metadata
      sizes_(calculate_sizes(image, memory_layout)),
      packed_dim_(utils::to_packed_dim<int32_t>(memory_layout)),
      dim_order_(),
      axis_map_(calculate_axis_map(sizes_, axis_map_layout)),
      strides_(),
      numel_(utils::multiply_integers(sizes_)),
      hashed_layout_(create_hashed_layout(
          dim_order_,
          axis_map_,
          packed_dim_,
          utils::kTexture3D)),
      // Related to tensor metadata UBOs
      min_nbytes_per_ubo_{context->adapter_ptr()->min_ubo_alignment()},
      max_ubo_nbytes_{
          calculate_max_ubo_nbytes(min_nbytes_per_ubo_, utils::kTexture3D)},
      uniforms_(),
      buffer_meta_(),
      // Construct Tensor storage
      storage_(std::make_shared<vTensorStorage>(context, image)) {
  uniform_data_ = std::make_shared<UniformData>(UniformData{
      numel_,
      sizes_,
      {0, 0, 0, 0},
      {0, 0, 0, 0},
      calculate_logical_limits(storage_->image_extents_, axis_map_)});
}

vTensor::vTensor(vTensor& other)
    : dtype_(other.dtype_),
      // Copy tensor size metadata
      sizes_(other.sizes_.begin(), other.sizes_.end()),
      packed_dim_{other.packed_dim_},
      dim_order_(other.dim_order_.begin(), other.dim_order_.end()),
      axis_map_(other.axis_map_.begin(), other.axis_map_.end()),
      strides_(other.strides_.begin(), other.strides_.end()),
      numel_(other.numel_),
      hashed_layout_(other.hashed_layout_),
      min_nbytes_per_ubo_{other.min_nbytes_per_ubo_},
      max_ubo_nbytes_{other.max_ubo_nbytes_},
      uniforms_(),
      buffer_meta_(),
      // Copy Tensor storage
      storage_(other.storage_) {
  uniform_data_ = std::make_shared<UniformData>(*other.get_uniform_data());
}

vTensor::vTensor(
    vTensor& other,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& dim_order)
    : dtype_(other.dtype_),
      // Copy tensor size metadata
      sizes_(sizes.begin(), sizes.end()),
      packed_dim_(other.packed_dim_),
      dim_order_(dim_order.begin(), dim_order.end()),
      axis_map_(calculate_axis_map(sizes_, utils::kDefaultAxisMap)),
      strides_(calculate_strides(sizes_, dim_order_)),
      numel_(other.numel_),
      hashed_layout_(create_hashed_layout(
          dim_order_,
          axis_map_,
          packed_dim_,
          other.storage_type())),
      min_nbytes_per_ubo_{other.min_nbytes_per_ubo_},
      max_ubo_nbytes_{other.max_ubo_nbytes_},
      uniforms_(),
      buffer_meta_(),
      // Copy Tensor storage
      storage_(other.storage_) {
  uniform_data_ = std::make_shared<UniformData>(UniformData{
      static_cast<size_t>(utils::multiply_integers(sizes_)),
      sizes_,
      dim_order_,
      strides_,
      other.logical_limits()});

  VK_CHECK_COND(
      dim_order_is_valid(dim_order_), "new dim order provided is invalid");
}

vTensor::UniformData::UniformData(
    const size_t numel_ll,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& dim_order,
    const std::vector<int64_t>& strides,
    const utils::uvec3& limits)
    : numel(utils::safe_downcast<int32_t>(numel_ll)),
      sizes_v(flip_and_unsqueeze_ivec4(sizes, kTensorSizes, numel_ll)),
      dim_order_v(
          flip_and_unsqueeze_ivec4(dim_order, kTensorDimOrder, numel_ll)),
      strides_v(flip_and_unsqueeze_ivec4(strides, kTensorStrides, numel_ll)),
      logical_limits(limits) {}

uint32_t vTensor::UniformData::write_attribute(
    void* dst,
    const uint32_t dst_offset,
    const uint32_t max_dst_size,
    const Attribute attr) {
#define WRITE_ATTRIBUTE_CASE(enum_name, member_name)                       \
  case vTensor::Attribute::enum_name: {                                    \
    VK_CHECK_COND(                                                         \
        (dst_offset + sizeof(member_name)) <= max_dst_size,                \
        "Attempting to write tensor attribute outside data boundary.");    \
    memcpy((uint8_t*)dst + dst_offset, &member_name, sizeof(member_name)); \
    return sizeof(member_name);                                            \
  }
  switch (attr) {
    WRITE_ATTRIBUTE_CASE(NUMEL, numel);
    WRITE_ATTRIBUTE_CASE(SIZES, sizes_v);
    WRITE_ATTRIBUTE_CASE(WHCN_DIM_ORDER, dim_order_v);
    WRITE_ATTRIBUTE_CASE(STRIDES, strides_v);
    WRITE_ATTRIBUTE_CASE(LOGICAL_LIMITS, logical_limits);
    default:
      VK_THROW("Invalid Attribute");
  }
#undef WRITE_ATTRIBUTE_CASE
  return 0;
}

vTensor::BufferMetadata::BufferMetadata(
    std::vector<int64_t>& src_sizes,
    std::vector<int64_t>& src_dim_order,
    std::vector<int64_t>& src_strides,
    size_t src_numel) {
  update(src_sizes, src_dim_order, src_strides, src_numel);
}

void vTensor::BufferMetadata::update(
    std::vector<int64_t>& src_sizes,
    std::vector<int64_t>& src_dim_order,
    std::vector<int64_t>& src_strides,
    size_t src_numel) {
  int32_t fixed_ndim = utils::safe_downcast<int32_t>(kTensorDimLimit);

  std::vector<uint32_t> fu_sizes = flip_and_unsqueeze<uint32_t>(
      src_sizes, kTensorSizes, src_numel, fixed_ndim);
  std::vector<uint32_t> fu_dim_order = flip_and_unsqueeze<uint32_t>(
      src_dim_order, kTensorDimOrder, src_numel, fixed_ndim);
  std::vector<uint32_t> fu_strides = flip_and_unsqueeze<uint32_t>(
      src_strides, kTensorStrides, src_numel, fixed_ndim);

  for (int i = 0; i < fixed_ndim; ++i) {
    sizes[i] = fu_sizes.at(i);
    dim_order[i] = fu_dim_order.at(i);
    strides[i] = fu_strides.at(i);
  }

  ndim = utils::safe_downcast<uint32_t>(src_sizes.size());
  numel = utils::safe_downcast<uint32_t>(src_numel);
}

vkapi::VulkanImage& vTensor::image(
    vkapi::PipelineBarrier& pipeline_barrier,
    const vkapi::PipelineStageFlags stage) & {
  storage_->transition(pipeline_barrier, stage, vkapi::MemoryAccessType::READ);
  return storage_->image_;
}

vkapi::VulkanImage& vTensor::image(
    vkapi::PipelineBarrier& pipeline_barrier,
    const vkapi::PipelineStageFlags stage,
    const vkapi::MemoryAccessFlags access) & {
  storage_->transition(pipeline_barrier, stage, access);
  return storage_->image_;
}

vkapi::VulkanBuffer& vTensor::buffer(
    vkapi::PipelineBarrier& pipeline_barrier,
    const vkapi::PipelineStageFlags stage) & {
  storage_->transition(pipeline_barrier, stage, vkapi::MemoryAccessType::READ);
  return storage_->buffer_;
}

vkapi::VulkanBuffer& vTensor::buffer(
    vkapi::PipelineBarrier& pipeline_barrier,
    const vkapi::PipelineStageFlags stage,
    const vkapi::MemoryAccessFlags access) & {
  storage_->transition(pipeline_barrier, stage, access);
  return storage_->buffer_;
}

utils::GPUMemoryLayout vTensor::estimate_memory_layout() const {
  switch (packed_dim_) {
    case WHCN::kWidthDim:
      return utils::kWidthPacked;
    case WHCN::kHeightDim:
      return utils::kHeightPacked;
    case WHCN::kChannelsDim:
      return utils::kChannelsPacked;
    default:
      VK_THROW("Invalid packed dim");
  }
}

bool vTensor::is_contiguous() const {
  if (storage_type() != utils::kBuffer) {
    return false;
  }
  for (size_t i = 0; i < dim_order_.size(); ++i) {
    if (dim_order_.at(i) != i) {
      return false;
    }
  }
  return true;
}

size_t vTensor::get_max_ubo_nbytes(const size_t nbytes_per_ubo) const {
  // For texture backed tensors, the metadata fields needed are:
  // sizes, logical limits
  size_t max_metadata_field_count = 2u;
  if (storage_type() == utils::kBuffer) {
    // sizes, strides, dim order, numel
    max_metadata_field_count = 4u;
  }
  return max_metadata_field_count * nbytes_per_ubo;
}

const vkapi::BufferBindInfo vTensor::sizes_ubo() {
  VK_CHECK_COND(sizes_.size() <= 4);
  return metadata_ubo_impl(&sizes_uniform_offset_, uniform_data_->sizes_v);
}

const vkapi::BufferBindInfo vTensor::dim_order_ubo() {
  VK_CHECK_COND(sizes_.size() <= 4);
  return metadata_ubo_impl(
      &dim_order_uniform_offset_, uniform_data_->dim_order_v);
}

const vkapi::BufferBindInfo vTensor::strides_ubo() {
  VK_CHECK_COND(sizes_.size() <= 4);
  return metadata_ubo_impl(&strides_uniform_offset, uniform_data_->strides_v);
}

const vkapi::BufferBindInfo vTensor::logical_limits_ubo() {
  VK_CHECK_COND(sizes_.size() <= 4);
  return metadata_ubo_impl(
      &logical_limits_uniform_offset_, uniform_data_->logical_limits);
}

const vkapi::BufferBindInfo vTensor::numel_ubo() {
  VK_CHECK_COND(sizes_.size() <= 4);
  return metadata_ubo_impl(&numel_uniform_offset_, uniform_data_->numel);
}

const vkapi::BufferBindInfo vTensor::buffer_meta_ubo() {
  size_t ubo_nbytes = sizeof(BufferMetadata);
  if (!buffer_meta_.buffer()) {
    BufferMetadata data(sizes_, dim_order_, strides_, numel_);
    buffer_meta_ = ParamsBuffer(storage_->context_, data);
  }
  return vkapi::BufferBindInfo(buffer_meta_.buffer(), 0, ubo_nbytes);
}

VkMemoryRequirements vTensor::get_memory_requirements() const {
  switch (storage_type()) {
    case utils::kBuffer:
      return storage_->buffer_.get_memory_requirements();
    case utils::kTexture2D:
    case utils::kTexture3D:
      return storage_->image_.get_memory_requirements();
  }
  return {};
}

bool vTensor::memory_is_bound() const {
  switch (storage_type()) {
    case utils::kBuffer:
      return storage_->buffer_.has_memory();
    case utils::kTexture2D:
    case utils::kTexture3D:
      return storage_->image_.has_memory();
  }
}

void vTensor::bind_allocation(const vkapi::Allocation& allocation) {
  switch (storage_type()) {
    case utils::kBuffer:
      storage_->buffer_.bind_allocation(allocation);
      break;
    case utils::kTexture2D:
    case utils::kTexture3D:
      storage_->image_.bind_allocation(allocation);
      break;
  }
}

void vTensor::acquire_allocation(vkapi::Allocation&& allocation) {
  switch (storage_type()) {
    case utils::kBuffer:
      storage_->buffer_.acquire_allocation(std::move(allocation));
      break;
    case utils::kTexture2D:
    case utils::kTexture3D:
      storage_->image_.acquire_allocation(std::move(allocation));
      break;
  }
}

void vTensor::update_metadata() {
  numel_ = utils::multiply_integers(sizes_);
  strides_ = calculate_strides(sizes_, dim_order_);

  // Update uniform data if it has been modified
  if (sizes_.size() <= 4) {
    uniform_data_->numel = utils::safe_downcast<int32_t>(numel_);
    uniform_data_->sizes_v =
        flip_and_unsqueeze_ivec4(sizes_, kTensorSizes, numel_);
    uniform_data_->dim_order_v =
        flip_and_unsqueeze_ivec4(dim_order_, kTensorDimOrder, numel_);
    uniform_data_->strides_v =
        flip_and_unsqueeze_ivec4(strides_, kTensorStrides, numel_);
    uniform_data_->logical_limits.limits =
        calculate_logical_limits(sizes_, axis_map_, packed_dim_);

    if (sizes_uniform_offset_ != kUniformOffsetUnset) {
      uniforms_.update(uniform_data_->sizes_v, sizes_uniform_offset_);
    }
    if (dim_order_uniform_offset_ != kUniformOffsetUnset) {
      uniforms_.update(uniform_data_->dim_order_v, dim_order_uniform_offset_);
    }
    if (strides_uniform_offset != kUniformOffsetUnset) {
      uniforms_.update(uniform_data_->strides_v, strides_uniform_offset);
    }
    if (numel_uniform_offset_ != kUniformOffsetUnset) {
      uniforms_.update(numel_, numel_uniform_offset_);
    }
    if (logical_limits_uniform_offset_ != kUniformOffsetUnset) {
      uniforms_.update(
          uniform_data_->logical_limits.limits, logical_limits_uniform_offset_);
    }
  }

  if (buffer_meta_.buffer()) {
    BufferMetadata data(sizes_, dim_order_, strides_, numel_);
    buffer_meta_.update(data);
  }
}

void vTensor::check_sizes(const std::vector<int64_t>& sizes) const {
  if (storage_type() != utils::kBuffer) {
    // For texture storage check that the current texture is large enough for
    // the new sizes of the tensor.
    utils::uvec3 virtual_extents = calculate_image_extents(
        calculate_padded_sizes(sizes_, packed_dim_), axis_map_, packed_dim_);

    bool valid_resize = virtual_extents[0] <= storage_->image_extents_[0];
    valid_resize =
        valid_resize && virtual_extents[1] <= storage_->image_extents_[1];
    valid_resize =
        valid_resize && virtual_extents[2] <= storage_->image_extents_[2];

    VK_CHECK_COND(
        valid_resize,
        "tensor sizes requires a larger texture than the current one.");
  } else {
    // For buffer storage check that the current buffer is large enough for the
    // new sizes of the tensor.
    int64_t numel = utils::multiply_integers(sizes);
    bool valid_resize =
        numel + storage_->buffer_offset_ <= storage_->buffer_length_;
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

  // Update the hashed layout because dim order is updated
  hashed_layout_ =
      create_hashed_layout(dim_order_, axis_map_, packed_dim_, storage_type());

  update_metadata();
}

void vTensor::virtual_clone(const vTensor& other) {
  VK_CHECK_COND(is_view_of(other));
  sizes_ = other.sizes_;
  dim_order_ = other.dim_order_;
  axis_map_ = other.axis_map_;
  packed_dim_ = other.packed_dim_;
  hashed_layout_ = other.hashed_layout_;

  *uniform_data_ = *other.get_uniform_data();
}

void vTensor::virtual_resize(const std::vector<int64_t>& new_sizes) {
  VK_CHECK_COND(
      new_sizes.size() == dim_order_.size(),
      "new sizes cannot modify the dimensionality of the tensor ");

  check_sizes(new_sizes);
  sizes_ = new_sizes;
  update_metadata();
}

/*
 * Transposing the dim order is a bit unintuitive. dim0 and dim1 have swapped
 * their "identities", so we need to swap the values of dim0 and dim1 wherever
 * they appear in the dim order vector. Compare this to just swapping the
 * elements at dim0 and dim1 in the `sizes` vectors.
 */
void transpose_dim_order_inplace(
    std::vector<int64_t>& dim_order,
    const int64_t dim0,
    const int64_t dim1) {
  for (int i = 0; i < dim_order.size(); ++i) {
    if (dim_order[i] == dim0) {
      dim_order[i] = dim1;
    } else if (dim_order[i] == dim1) {
      dim_order[i] = dim0;
    }
  }
}

void vTensor::virtual_transpose(const int64_t dim0, const int64_t dim1) {
  std::iter_swap(sizes_.begin() + dim0, sizes_.begin() + dim1);

  const int dim0_whcn = sizes_.size() - 1 - dim0;
  const int dim1_whcn = sizes_.size() - 1 - dim1;
  if (packed_dim_ == dim0_whcn) {
    packed_dim_ = dim1_whcn;
  } else if (packed_dim_ == dim1_whcn) {
    packed_dim_ = dim0_whcn;
  }

  if (storage_type() == utils::kBuffer) {
    transpose_dim_order_inplace(dim_order_, dim0, dim1);
  } else {
    // Cannot transpose batch dimension for texture storage
    VK_CHECK_COND(dim0_whcn < 3 && dim1_whcn < 3);
    std::iter_swap(
        axis_map_.begin() + dim0_whcn, axis_map_.begin() + dim1_whcn);
    // Update the "identity" of the concatted dimension
    if (axis_map_.at(3) == dim0_whcn) {
      axis_map_.at(3) = dim1_whcn;
    } else if (axis_map_.at(3) == dim1_whcn) {
      axis_map_.at(3) = dim0_whcn;
    }
  }

  // Update the hashed layout because dim order / axis mpa is updated
  hashed_layout_ =
      create_hashed_layout(dim_order_, axis_map_, packed_dim_, storage_type());

  update_metadata();
}

} // namespace api
} // namespace vkcompute
