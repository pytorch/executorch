/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/api/Context.h>

#include <executorch/backends/vulkan/runtime/api/containers/ParamsBuffer.h>

#include <executorch/backends/vulkan/runtime/utils/StorageUtils.h>

namespace vkcompute {
namespace api {

static constexpr size_t kTensorDimLimit = 8;

/*
 * Given a GPUMemoryLayout value, produce a dim order vector that matches the
 * given memory layout. The produced dim order vector will be in the NCHW
 * dimension order
 */
std::vector<int64_t> calculate_dim_order(
    const size_t ndim,
    const int32_t packed_dim);

/*
 * Given the sizes of a tensor and the dim order of the tensor (both in NCHW)
 * dimension order, calculate the strides of the tensor.
 */
std::vector<int64_t> calculate_strides(
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& dim_order);

/*
 * When stored on the GPU, tensor data is stored using texels (i.e. a vector of
 * 4 scalar values) in order to take advantage of the GPU's native vectorization
 * capabilities. Furthermore, tensor metadata is passed in to shaders as ivec4
 * types.
 *
 * To accommodate these vectorized types, the sizes of a tensor will be modified
 * for GPU storage in the following ways:
 *
 *   1. The dimensionality of the tensor will be padded to a multiple of 4.
 *   2. The size of the packed dimension will be padded to a multiple of 4.
 *
 * The "packed dimension" is determined based on the utils::GPUMemoryLayout
 * argument.
 */
std::vector<int64_t> calculate_padded_sizes(
    const std::vector<int64_t>& sizes,
    const int32_t packed_dim);

/*
 * Calculate the image extents required of a texture backed tensor.
 */
utils::uvec3 calculate_image_extents(
    const std::vector<int64_t>& padded_sizes,
    const std::vector<int64_t>& axis_map,
    const int32_t packed_dim);

struct LastAccess {
  vkapi::PipelineStageFlags stage;
  vkapi::MemoryAccessFlags access;

  LastAccess()
      : stage{vkapi::PipelineStage::NO_STAGE},
        access{vkapi::MemoryAccessType::NONE} {}

  LastAccess(
      vkapi::PipelineStageFlags stage_flags,
      vkapi::MemoryAccessFlags access_flags)
      : stage{stage_flags}, access{access_flags} {}
};

/*
 * Calculate the number of elements that a GPU buffer would require to store the
 * contents of a tensor. This will depend on the storage type and dtype of the
 * tensor, as well as the features available on the device.
 */
int64_t calculate_gpu_buffer_numel(
    Context* const context,
    const std::vector<int64_t>& sizes,
    const utils::uvec3 image_extents,
    const utils::StorageType storage_type,
    const vkapi::ScalarType dtype);

class vTensorStorage final {
 public:
  // Do not allow empty vTensorStorage construction
  vTensorStorage() = default;

  vTensorStorage(
      Context* context,
      const utils::StorageType storage_type,
      const utils::GPUMemoryLayout memory_layout,
      const std::vector<int64_t>& axis_map,
      const int32_t packed_dim,
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const bool allocate_memory = true);

  vTensorStorage(Context* const context, const vkapi::VulkanImage& image);

 public:
  vTensorStorage(vTensorStorage& other) = delete;
  vTensorStorage& operator=(const vTensorStorage& other) = delete;

  vTensorStorage(vTensorStorage&& other) = default;
  vTensorStorage& operator=(vTensorStorage&& other) = default;

  ~vTensorStorage();

  friend class vTensor;

 private:
  // Context
  Context* context_{};

  utils::StorageType storage_type_;

  // Resource sizings
  utils::uvec3 image_extents_{};
  int64_t buffer_length_{};
  int64_t buffer_offset_{};

  // GPU Storage
  mutable vkapi::VulkanImage image_;
  mutable vkapi::VulkanBuffer buffer_;

  // Last Access - used to insert memory barriers
  LastAccess last_access_;

 private:
  // Registers underlying memory for cleanup
  void flush();

  // Memory barrier insertion
  void transition(
      vkapi::PipelineBarrier&,
      const vkapi::PipelineStageFlags,
      const vkapi::MemoryAccessFlags);

  // Validation
  void verify() const;

 public:
  inline size_t buffer_len() const {
    return utils::safe_downcast<size_t>(buffer_length_);
  }

  inline VkFormat texture_format() {
    return image_.format();
  }
};

class vTensor final {
  struct TextureLimits {
    // Alignment is required to conform with Vulkan specification; a 3 or 4
    // component vector with components of size N must have base alignment of
    // 4N.
    alignas(16) utils::ivec3 limits;

    TextureLimits(const utils::uvec3& ulimits) : limits{ulimits} {}
  };

 public:
  explicit vTensor(
      Context* context,
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const utils::StorageType storage_type = utils::kTexture3D,
      const utils::GPUMemoryLayout memory_layout = utils::kChannelsPacked,
      const bool allocate_memory = true,
      const utils::AxisMapLayout axis_map_layout = utils::kDefaultAxisMap);

  vTensor(const vTensor& other) = delete;

  explicit vTensor(
      Context* context,
      const vkapi::VulkanImage& image,
      const utils::GPUMemoryLayout memory_layout = utils::kChannelsPacked,
      const utils::AxisMapLayout axis_map_layout = utils::kDefaultAxisMap);

  /*
   * This constructor allows for the creation of a vTensor that references the
   * same buffer resource of another vTensor, with the same sizes and strides
   * metadata. The created vTensor will not own the underlying resource. This is
   * only applicable for buffer backed tensors at the moment.
   *
   * Once created, the sizes and strides of the aliased vTensor can be changed
   * using the `virtual_reconfigure` member function.
   */
  vTensor(vTensor& other);

  /*
   * This constructor allows for the creation of a vTensor that references the
   * same buffer resource of another vTensor, but with different sizes and
   * strides metatdata. The created vTensor will not own the underlying
   * resource. This is only applicable for buffer backed tensors at the moment.
   *
   * Note that dim order is used as the source of truth regarding the strides,
   * and the new strides are computed from the new sizes and new dim order.
   * Thus only the dim order is provided as an argument to this function.
   *
   * The offset_numel argument allows the aliased tensor's memory region to
   * begin at an offset of N elements from the start of the original tensor's
   * buffer.
   */
  vTensor(
      vTensor& other,
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& dim_order);

  // To discourage making copies, the copy assignment operator is still deleted
  vTensor& operator=(const vTensor& other) = delete;

  vTensor(vTensor&& other) = default;
  vTensor& operator=(vTensor&& other) = default;

  ~vTensor() = default;

  enum class Attribute : uint8_t {
    SIZES,
    WHCN_DIM_ORDER,
    STRIDES,
    LOGICAL_LIMITS,
    NUMEL,
  };

  class UniformData {
    // Contains the number of elements in the tensor according to the canonical
    // sizes.
    int32_t numel;
    utils::ivec4 sizes_v;
    utils::ivec4 dim_order_v;
    utils::ivec4 strides_v;
    // See the comments documenting logical_limits() for more context.
    TextureLimits logical_limits;

    friend class vTensor;

    UniformData(
        const size_t numel_ll,
        const std::vector<int64_t>& sizes,
        const std::vector<int64_t>& dim_order,
        const std::vector<int64_t>& strides,
        const utils::uvec3& limits);

   public:
    /*
     * Write tensor's metadata into dst, at the given dst_offset. max_dst_size
     * is the size of dst and is used to avoid out of bounds writes.
     */
    uint32_t write_attribute(
        void* dst,
        const uint32_t dst_offset,
        const uint32_t max_dst_size,
        const Attribute attr);
  };

  struct BufferMetadata {
    uint32_t sizes[kTensorDimLimit];
    uint32_t dim_order[kTensorDimLimit];
    uint32_t strides[kTensorDimLimit];
    uint32_t ndim;
    uint32_t numel;

    BufferMetadata(
        std::vector<int64_t>& sizes,
        std::vector<int64_t>& dim_order,
        std::vector<int64_t>& strides,
        size_t numel);

    void update(
        std::vector<int64_t>& sizes,
        std::vector<int64_t>& dim_order,
        std::vector<int64_t>& strides,
        size_t numel);
  };

  struct TextureMetadata {
    int32_t sizes[4];
    int32_t logical_limits[4];
    int32_t axis_map[4];
    int32_t packed_dim;

    TextureMetadata(
        const std::vector<int64_t>& sizes,
        const TextureLimits& logical_limits,
        const std::vector<int64_t>& axis_map,
        const int32_t packed_dim);

    void update(
        const std::vector<int64_t>& sizes,
        const TextureLimits& logical_limits,
        const std::vector<int64_t>& axis_map,
        const int32_t packed_dim);
  };

 private:
  /*
   * "Core" tensor metadata. They are the minimum amount of information required
   * to construct a tensor.
   */

  // Whether the tensor has elements of type float, int, etc.
  vkapi::ScalarType dtype_;
  // sizes of the tensor in NCHW dimension order
  std::vector<int64_t> sizes_;
  // Describes which dimension is "tightly packed" using WHCN index (i.e. 0 for
  // width, 1 for height, etc.). For texture backed tensors, this describes
  // which dimension is packed along a texel. For buffer backed tensors, this
  // describes which dimension has a stride of 1 (i.e. is last in the dim
  // order).
  int32_t packed_dim_;

  /*
   * "Layout" metadata. These describe with further detail how tensor data is
   * laid out in memory. However, they are considered secondary to the "core"
   * metadata members above because defaults can be assumed based on a given
   * memory layout. When permuting the tensor without performing a copy, these
   * metadata members are the ones that will be changed. All other metadata is
   * derived from a combination of sizes, memory layout, and the below members.
   */

  // dim order of the tensor; dimension indices are in NCHW dimension order
  // i.e. 0 is N, 1 is C, 2 is H, 3 is W for a 4D tensor. The dims with larger
  // strides precede the dims with smaller strides in the dim order. The last
  // dim is always the fastest moving dim with a stride of 1.
  std::vector<int64_t> dim_order_;
  // Describes which axis of an image texture each dimension of the tensor maps
  // to. The axis mapping allows texture based tensors to be permuted and
  // transposed without modifying the underlying texture storage. For a more in
  // depth explanation of axis mapping, see the `default_axis_map()`
  // function.
  std::vector<int64_t> axis_map_;

  /*
   * The below can be consider "layout" metadata as well, but are derived from
   * the above data members.
   */

  // strides of the tensor in NCHW dimension order
  std::vector<int64_t> strides_;

  // number of elements based on the canonical sizes
  size_t numel_;

  // For texture backed tensors, this int32 contains the axis map data packed
  // into a single int32. For buffer backed tensors, this int32 contains the
  // wchn dim order data packed into a single int32.
  int32_t hashed_layout_;

  // Pre-compute these quantities to avoid frequent re-computation
  size_t min_nbytes_per_ubo_;
  size_t max_ubo_nbytes_;

  /*
   * Utility GPU buffer that can be passed to shaders in order to convey tensor
   * metadata. Uniform buffer will be initialized only the first time a ubo is
   * requested. Buffer offsets will be initialized the first time they are
   * accessed via the corresponding *_ubo() function. Uniform buffer's contents
   * will be updated whenever virtual_resize() is called.
   *
   * Refer to the comments for the corresponding *_ubo() functions for more
   * context about the data contained in each buffer.
   */
  ParamsBuffer uniforms_;

  /*
   * Used to store data for BufferMetadata to pass to shaders as buffer_meta_ubo
   */
  ParamsBuffer buffer_meta_;

  /*
   * Used to store data for TextureMetadata to pass to shaders as
   * texture_meta_ubo
   */
  ParamsBuffer texture_meta_;

  uint32_t uniforms_size_ = 0u;
  uint32_t sizes_uniform_offset_ = kUniformOffsetUnset;
  uint32_t dim_order_uniform_offset_ = kUniformOffsetUnset;
  uint32_t strides_uniform_offset = kUniformOffsetUnset;
  uint32_t numel_uniform_offset_ = kUniformOffsetUnset;
  uint32_t logical_limits_uniform_offset_ = kUniformOffsetUnset;

  // Initial value of uniform buffer offsets. 1 is selected as it is essentially
  // impossible for a ubo to have an offset of 1.
  constexpr static uint32_t kUniformOffsetUnset = 1;

  std::shared_ptr<vTensorStorage> storage_;

  std::shared_ptr<UniformData> uniform_data_;

 public:
  /*
   Texture Access
  */

  inline vkapi::VulkanImage& image() const& {
    return storage_->image_;
  }

  vkapi::VulkanImage& image(
      vkapi::PipelineBarrier&,
      const vkapi::PipelineStageFlags) &;

  vkapi::VulkanImage& image(
      vkapi::PipelineBarrier&,
      const vkapi::PipelineStageFlags,
      const vkapi::MemoryAccessFlags) &;

  inline vkapi::VulkanBuffer& buffer() const& {
    return storage_->buffer_;
  }

  vkapi::VulkanBuffer& buffer(
      vkapi::PipelineBarrier&,
      const vkapi::PipelineStageFlags) &;

  vkapi::VulkanBuffer& buffer(
      vkapi::PipelineBarrier&,
      const vkapi::PipelineStageFlags,
      const vkapi::MemoryAccessFlags) &;

  /*
    Metadata
  */

  inline utils::StorageType storage_type() const {
    return storage_->storage_type_;
  }

  inline bool has_buffer_storage() const {
    return storage_->storage_type_ == utils::kBuffer;
  }

 public:
  /*
   * The logical limits of the tensor are derived from the image extents of the
   * image texture used to store the tensor, but with two key differences.
   *
   * First, the image extents are permuted according to the axis map. This
   * makes it so that the first element of the logical limit is the limit of the
   * texture axis corresponding to the width dimension of the tensor, the next
   * element is the limit of the texture axis corresponding to the height
   * dimension and the last element is the limit of the texture axis that
   * corresponds to the channels dimension of the tensor.
   *
   * Second, the logical limits may use smaller extents than the actual image
   * extents of the image texture. This is due to dynamic shape; if the tensor's
   * `virtual_resize()` function is called, then the logical limits will reflect
   * the extents that would be needed to support a tensor with the updated sizes
   * instead of the original sizes.
   */
  inline const utils::ivec3& logical_limits() const {
    return uniform_data_->logical_limits.limits;
  }

  /*
   * Extract an `vkapi::ScalarType` from the TensorOptions member
   */
  inline vkapi::ScalarType dtype() const {
    return dtype_;
  }

  /*
   * Provide a "best guess" of a memory layout that can be used to construct a
   * tensor with similar layout metadata (i.e. strides, axis_map, etc.) as this
   * tensor. In some scenarios, the exact layout of the tensor may not be able
   * to be replicated due to calling `virtual_*()` functions after construction;
   * however, this function will provide a memory layout that will produce the
   * same `packed_dim_` as this tensor.
   */
  utils::GPUMemoryLayout estimate_memory_layout() const;

  inline int32_t packed_dim() const {
    return packed_dim_;
  }

  /*
   * Returns the WHCN index of the dimension that is used to concatenate batches
   * as an int32_t.
   */
  inline int32_t concat_dim() const {
    return utils::safe_downcast<int32_t>(axis_map_.at(3));
  }

  inline const std::vector<int64_t>& sizes() const {
    return sizes_;
  }

  inline const int64_t size(size_t dim) const {
    return sizes().at(dim);
  }

  inline const int64_t dim() const {
    return sizes_.size();
  }

  inline const std::vector<int64_t>& dim_order() const {
    return dim_order_;
  }

  inline const std::vector<int64_t>& strides() const {
    return strides_;
  }

  inline size_t numel() const {
    return numel_;
  }

  inline size_t nbytes() const {
    return element_size(dtype()) * numel();
  }

  inline const std::vector<int64_t>& axis_map() const {
    return axis_map_;
  }

  /*
   * For texture backed tensors, this function return a int32_t that contains
   * the axis map + packed dimension. Each element of the axis map occupies 4
   * bits of the int32.
   *
   * For buffer backed tensors, the int32_t contains the WHCN dim order, where
   * each element of the dim order array occupies 4 bits of the int32.
   *
   * This int32 is typically consumed as a specialization constant in compute
   * shaders where it is subsequently unpacked. The layout data of a vTensor
   * instance is typically static once created, which is why this method is
   * appropriate.
   */
  inline int32_t hashed_layout() const {
    return hashed_layout_;
  }

  /*
   * Return true if the tensor's axis map is {0, 1, 2, concat_dim}. This means
   * that the width dim is mapped to the width axis of the texture, the height
   * dim is mapped to the height axis of the texture, the channels dim is mapped
   * to the depth axis of the texture.
   */
  inline bool has_standard_axis_map() const {
    return axis_map_.at(0) == 0 && axis_map_.at(1) == 1 && axis_map_.at(2) == 2;
  }

  /*
   * Return true if a buffer backed tensor's dim order matches that of a
   * contiguous tensor, i.e. the dim order will be {0, 1, 2, ... }.
   * Returns false for texture backed tensors.
   */
  bool is_contiguous() const;

 private:
  inline size_t nbytes_per_ubo() const {
    return storage_->context_->adapter_ptr()->min_ubo_alignment();
  }

  size_t get_max_ubo_nbytes(const size_t nbytes_per_ubo) const;

  template <typename T>
  const vkapi::BufferBindInfo metadata_ubo_impl(
      uint32_t* param_buffer_offset,
      const T& data) {
    if (!uniforms_.buffer()) {
      uniforms_ = ParamsBuffer(storage_->context_, max_ubo_nbytes_, true);
    }
    size_t ubo_nbytes = utils::align_up(sizeof(data), min_nbytes_per_ubo_);
    if (*param_buffer_offset == kUniformOffsetUnset) {
      VK_CHECK_COND(
          (uniforms_size_ + ubo_nbytes) <= max_ubo_nbytes_,
          "Uniform data allocation has exceeded Tensor uniform buffer size");
      *param_buffer_offset = uniforms_size_;
      uniforms_size_ += ubo_nbytes;
      uniforms_.update(data, *param_buffer_offset);
    }
    return vkapi::BufferBindInfo(
        uniforms_.buffer(), *param_buffer_offset, ubo_nbytes);
  }

 public:
  /*
   * The functions below return the buffer binding info for a UBO that contains
   * some metadata of the tensor, which can be used to pass in tensor metadata
   * to a compute shader. The other method of passing in tensor metadata is via
   * push constants. The trade-off between each is that push constants may be
   * slightly more performant and memory efficient; however, to update the
   * values in a push constant due to i.e. a tensor resize between inferences,
   * the command buffer must be re-encoded. On the other hand, UBOs can update
   * their data by writing to their mapped memory without requiring a command
   * buffer re-encode.
   */

  const vkapi::BufferBindInfo sizes_ubo();

  const vkapi::BufferBindInfo dim_order_ubo();

  const vkapi::BufferBindInfo strides_ubo();

  const vkapi::BufferBindInfo logical_limits_ubo();

  const vkapi::BufferBindInfo numel_ubo();

  const vkapi::BufferBindInfo buffer_meta_ubo();

  const vkapi::BufferBindInfo texture_meta_ubo();

 public:
  inline size_t staging_buffer_numel() const {
    return storage_->buffer_len();
  }

  inline size_t staging_buffer_nbytes() const {
    return element_size(dtype()) * staging_buffer_numel();
  }

  /*
   * Return the VmaAllocationCreateInfo of the underlying resource
   */
  VmaAllocationCreateInfo get_allocation_create_info() const;

  /*
   * Checks if the tensor's underlying buffer or image resource is bound to a
   * memory allocation.
   */
  bool memory_is_bound() const;

  /*
   * Return the VkMemoryRequirements of the underlying resource
   */
  VkMemoryRequirements get_memory_requirements() const;

  /*
   * Binds the underlying resource to the given memory allocation
   */
  void bind_allocation(const vkapi::Allocation& allocation);

  /*
   * Binds and acquires a rvalue memory allocation
   */
  void acquire_allocation(vkapi::Allocation&& allocation);

 private:
  /*
   * Assuming sizes, dim order, or axis mapping was modified, recompute all
   * derived metadata and update metadata UBO with new values.
   */
  void update_metadata();

  /*
   * Check that tensor sizes are valid given the current storage resource's
   * limits.
   */
  void check_sizes(const std::vector<int64_t>& sizes) const;

 public:
  /*
   * Change how the tensor should be interpreted by compute shaders via updating
   * the size and dim order of the tensor. The new sizes and dim order may have
   * different dimensionality than the current dimensionality of the tensor.
   *
   * This function can only be used for buffer-backed tensors, since texture
   * backed buffers cannot change dimensionality or memory layout.
   *
   * TODO(ssjia): delete this API. prefer functions such as virtual_transpose
   * instead.
   */
  void virtual_reconfigure(
      const std::vector<int64_t>& new_sizes,
      const std::vector<int64_t>& new_dim_order);

  /*
   * Set all metadata of this tensor to match the metadata of another tensor.
   */
  void virtual_clone(const vTensor& other);

  /*
   * Perform a virtual resize of the vTensor by modifying the size metadata that
   * gets used in compute shaders. This allows the shader to treat the
   * underlying resource as if it were a different size. The new sizes cannot
   * modify the dimensionality of the tensor.
   */
  void virtual_resize(const std::vector<int64_t>& new_sizes);

  /*
   * Transpose the tensor in-place by updating its metadata.
   */
  void virtual_transpose(const int64_t dim0, const int64_t dim1);

  /*
   * Check if this vTensor instance is a view of another vTensor instance
   */
  inline bool is_view_of(const vTensor& other) const {
    return storage_.get() == other.storage_.get();
  }

  const std::shared_ptr<UniformData>& get_uniform_data() const {
    VK_CHECK_COND(sizes_.size() <= 4);
    return uniform_data_;
  }
};

static constexpr vTensor::Attribute kTensorSizes = vTensor::Attribute::SIZES;
static constexpr vTensor::Attribute kTensorDimOrder =
    vTensor::Attribute::WHCN_DIM_ORDER;
static constexpr vTensor::Attribute kTensorStrides =
    vTensor::Attribute::STRIDES;
static constexpr vTensor::Attribute kTensorLogicalLimits =
    vTensor::Attribute::LOGICAL_LIMITS;
static constexpr vTensor::Attribute kTensorNumel = vTensor::Attribute::NUMEL;

/*
 * Prepare tensor metadata vector for consumption on the GPU:
 * 1. Convert NCHW dim order and indexes to WCHN dim order and indexes
 * 2. Unsqueeze to the next multiple of 4 dims
 * 3. Convert to requested output dtype
 */
template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
std::vector<T> flip_and_unsqueeze(
    const std::vector<int64_t>& tensor_metadata,
    const vTensor::Attribute metadata_type,
    const size_t numel,
    const int32_t fixed_ndim = -1) {
  const size_t ndim = tensor_metadata.size();
  size_t ndim_up4 =
      std::max(utils::align_up_4(tensor_metadata.size()), size_t(4));

  if (fixed_ndim > 0) {
    VK_CHECK_COND(fixed_ndim >= ndim);
    ndim_up4 = static_cast<size_t>(fixed_ndim);
  }

  std::vector<T> flipped_metadata(ndim_up4);

  for (int flipped_i = 0; flipped_i < ndim; ++flipped_i) {
    T val_at_dim =
        utils::safe_downcast<T>(tensor_metadata.at(ndim - 1 - flipped_i));
    if (metadata_type == kTensorDimOrder) {
      val_at_dim = utils::safe_downcast<T>(ndim - 1 - val_at_dim);
    }
    flipped_metadata.at(flipped_i) = val_at_dim;
  }

  switch (metadata_type) {
    case kTensorStrides:
      for (int unsqueezed_i = ndim; unsqueezed_i < ndim_up4; ++unsqueezed_i) {
        flipped_metadata.at(unsqueezed_i) = utils::safe_downcast<T>(numel);
      }
      break;
    case kTensorDimOrder:
      for (int unsqueezed_i = ndim; unsqueezed_i < ndim_up4; ++unsqueezed_i) {
        flipped_metadata.at(unsqueezed_i) =
            utils::safe_downcast<T>(unsqueezed_i);
      }
      break;
    // Default: unsqueeze with ones
    default:
      for (int unsqueezed_i = ndim; unsqueezed_i < ndim_up4; ++unsqueezed_i) {
        flipped_metadata.at(unsqueezed_i) = utils::safe_downcast<T>(1);
      }
      break;
  }

  return flipped_metadata;
}

/*
 * Same as flip and unsqueeze, but returns the metadata as an `ivec4`.
 */
utils::ivec4 flip_and_unsqueezed_ivec4(
    const std::vector<int64_t>& tensor_metadata,
    const vTensor::Attribute metadata_type,
    const size_t numel);

} // namespace api
} // namespace vkcompute
