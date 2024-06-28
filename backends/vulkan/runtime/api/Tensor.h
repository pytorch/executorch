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
#include <executorch/backends/vulkan/runtime/api/Types.h>

namespace vkcompute {

/*
 * Given the sizes of a tensor and the GPU memory layout, calculate the strides
 * of the tensor in NCHW dimension order. The GPU memory layout will be used to
 * determine which dimension is packed along a texel; that dimension will be
 * used as the "fasted moving" dimension with a stride of 1.
 *
 * If texel_strides is true, then the strides will be calculated for a texel
 * buffer (i.e. the size of the packed dimension will be modified by the
 * div_up_4 function before being used in calculations). Otherwise, the strides
 * will be calculated assuming a contiguous scalar buffer.
 */
std::vector<int64_t> calculate_strides(
    const std::vector<int64_t>& sizes,
    const api::GPUMemoryLayout memory_layout,
    const bool texel_strides = true);

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
 * The "packed dimension" is determined based on the GPUMemoryLayout argument.
 */
std::vector<int64_t> calculate_padded_sizes(
    const std::vector<int64_t>& sizes,
    const api::GPUMemoryLayout memory_layout);

/*
 * Given the padded sizes of a tensor and the GPU memory layout, calculate the
 * 3D image extents required to store the tensor data as an image texture.
 */
api::utils::uvec3 calculate_image_extents(
    const std::vector<int64_t>& padded_sizes,
    const api::GPUMemoryLayout memory_layout);

struct LastAccess {
  api::PipelineStageFlags stage;
  api::MemoryAccessFlags access;

  LastAccess()
      : stage{api::PipelineStage::NO_STAGE},
        access{api::MemoryAccessType::NONE} {}

  LastAccess(
      api::PipelineStageFlags stage_flags,
      api::MemoryAccessFlags access_flags)
      : stage{stage_flags}, access{access_flags} {}
};

class vTensorStorage final {
 public:
  // Do not allow empty vTensorStorage construction
  vTensorStorage() = default;

  vTensorStorage(
      api::Context* context,
      const api::StorageType storage_type,
      const api::GPUMemoryLayout gpu_memory_layout,
      const std::vector<int64_t>& sizes,
      const api::ScalarType dtype,
      const bool allocate_memory = true);

  vTensorStorage(const vTensorStorage& other) = delete;
  vTensorStorage& operator=(const vTensorStorage& other) = delete;

  vTensorStorage(vTensorStorage&& other) = default;
  vTensorStorage& operator=(vTensorStorage&& other) = default;

  ~vTensorStorage();

  friend class vTensor;

 private:
  // Context
  api::Context* context_{};

  api::StorageType storage_type_;

  // Resource sizings
  api::utils::uvec3 image_extents_{};
  int64_t buffer_length_{};

  // GPU Storage
  mutable api::VulkanImage image_;
  mutable api::VulkanBuffer buffer_;

  // Last Access - used to insert memory barriers
  LastAccess last_access_;

 private:
  // Registers underlying memory for cleanup
  void flush();

  // Memory barrier insertion
  void transition(
      api::PipelineBarrier&,
      const api::PipelineStageFlags,
      const api::MemoryAccessFlags);

  // Validation
  void verify() const;

 public:
  inline VkFormat texture_format() {
    return image_.format();
  }

  void discard_and_reallocate(
      const std::vector<int64_t>& padded_sizes,
      const api::GPUMemoryLayout gpu_memory_layout,
      const api::ScalarType dtype);
};

class vTensor final {
  struct TextureLimits {
    // Alignment is required to conform with Vulkan specification; a 3 or 4
    // component vector with components of size N must have base alignment of
    // 4N.
    alignas(16) api::utils::ivec3 limits;
  };

 public:
  explicit vTensor(
      api::Context* context,
      const std::vector<int64_t>& sizes,
      const api::ScalarType dtype,
      const api::StorageType storage_type = api::kTexture3D,
      const api::GPUMemoryLayout memory_layout = api::kChannelsPacked,
      const bool allocate_memory = true);

  vTensor(const vTensor& other) = delete;
  vTensor& operator=(const vTensor& other) = delete;

  vTensor(vTensor&& other) = default;
  vTensor& operator=(vTensor&& other) = default;

 private:
  api::ScalarType dtype_;
  api::GPUMemoryLayout memory_layout_;

  // sizes of the tensor in NCHW dimension order
  std::vector<int64_t> sizes_;
  // padded sizes of the tensor in NCHW dimension order. See the
  // calculate_padded_sizes() function for more context.
  std::vector<int64_t> padded_sizes_;
  // Contains the "virtual" texture extents of the tensor. See the
  // texture_limits() function for more context.
  TextureLimits texture_limits_;

  /*
   * Utility GPU buffers that can be passed to shaders in order to convey tensor
   * metadata. These buffers will be initialized the first time they are
   * accessed via the corresponding *_ubo() function, and their contents will be
   * updated whenever virtual_resize() is called.
   *
   * Refer to the comments for the corresponding *_ubo() functions for more
   * context about the data contained in each buffer.
   */
  api::UniformParamsBuffer sizes_uniform_;
  api::UniformParamsBuffer texture_limits_uniform_;
  api::UniformParamsBuffer texel_strides_uniform_;
  api::UniformParamsBuffer ntexels_uniform_;

  vTensorStorage storage_;

 public:
  /*
   Texture Access
  */

  inline api::VulkanImage& image() const& {
    return storage_.image_;
  }

  api::VulkanImage& image(
      api::PipelineBarrier&,
      const api::PipelineStageFlags) &;

  api::VulkanImage& image(
      api::PipelineBarrier&,
      const api::PipelineStageFlags,
      const api::MemoryAccessFlags) &;

  inline api::VulkanBuffer& buffer() const& {
    return storage_.buffer_;
  }

  api::VulkanBuffer& buffer(
      api::PipelineBarrier&,
      const api::PipelineStageFlags) &;

  api::VulkanBuffer& buffer(
      api::PipelineBarrier&,
      const api::PipelineStageFlags,
      const api::MemoryAccessFlags) &;

  /*
    Metadata
  */

  inline api::StorageType storage_type() const {
    return storage_.storage_type_;
  }

  inline bool has_buffer_storage() const {
    return storage_.storage_type_ == api::kBuffer;
  }

  inline const api::utils::uvec3& image_extents() const {
    return storage_.image_extents_;
  }

  /*
   * Extract an `api::ScalarType` from the TensorOptions member
   */
  inline api::ScalarType dtype() const {
    return dtype_;
  }

  inline api::GPUMemoryLayout gpu_memory_layout() const {
    return memory_layout_;
  }

  inline int32_t packed_dim_whcn_idx() const {
    return static_cast<int32_t>(memory_layout_);
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

  /*
   * Returns a GPU buffer containing the sizes of the tensor in WHCN order.
   * Note that dimensions that are not present in the tensor's sizes are set to
   * a size of 1.
   */
  const api::BufferBindInfo sizes_ubo();

  /*
   * Returns a GPU buffer containing the virtual image extents of the tensor.
   * Since a tensor can be resized with the virtual_resize() function, this
   * GPU buffer contains the image extents of the tensor calculated using the
   * virtual_resize() function. This allows shaders to exit early if they are
   * working outside the limits of the texture.
   *
   * This buffer should only be used to
   */
  const api::BufferBindInfo texture_limits_ubo();

  /*
   * Returns the strides of the texel buffer used to store the tensor, as
   * calculated by calculate_strides().
   */
  const api::BufferBindInfo texel_strides_ubo();

  /*
   * Returns the number of texels in the texel buffer used to store the tensor.
   */
  const api::BufferBindInfo ntexels_ubo();

  inline const api::utils::ivec3 texture_limits() const {
    return texture_limits_.limits;
  }

  inline size_t numel() const {
    return api::utils::multiply_integers(sizes());
  }

  inline size_t nbytes() const {
    return api::element_size(dtype()) * numel();
  }

  /*
   * Returns numel but based on padded_sizes_ instead of sizes_
   */
  inline size_t gpu_numel() const {
    return api::utils::multiply_integers(padded_sizes_);
  }

  /*
   * Returns the number of texels in the image texture or texel buffer used to
   * store the tensor's data.
   */
  inline int32_t texel_numel() const {
    return api::utils::safe_downcast<int32_t>(gpu_numel() / 4);
  }

  /*
   * Return nbytes but based on padded_sizes_ instead of sizes_
   */
  inline VkDeviceSize gpu_nbytes() const {
    return api::element_size(dtype()) * gpu_numel();
  }

  /*
   * Return the VmaAllocationCreateInfo of the underlying resource
   */
  VmaAllocationCreateInfo get_allocation_create_info() const;

  /*
   * Return the VkMemoryRequirements of the underlying resource
   */
  VkMemoryRequirements get_memory_requirements() const;

  /*
   * Binds the underlying resource to the given memory allocation
   */
  void bind_allocation(const api::Allocation& allocation);

 private:
  /*
   * Update the size metadata of the vTensor to be new sizes. Should not be used
   * directly, reallocate() or virtual_resize() should be used instead.
   */
  void update_size_metadata(const std::vector<int64_t>& new_sizes);

 public:
  /*
   * Discard the underlying VkImage or VkBuffer and re-allocate based on new
   * tensor sizes
   */
  void reallocate(const std::vector<int64_t>& new_sizes);

  /*
   * Perform a virtual resize of the vTensor by modifying the size metadata that
   * gets used in compute shaders. This allows the shader to treat the
   * underlying resource as if it were a different size.
   */
  void virtual_resize(const std::vector<int64_t>& new_sizes);
};

} // namespace vkcompute
