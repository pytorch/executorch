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

  vTensorStorage(const vTensorStorage&) = delete;
  vTensorStorage& operator=(const vTensorStorage&) = delete;

  vTensorStorage(vTensorStorage&&) = default;
  vTensorStorage operator=(vTensorStorage&&) = delete;

  ~vTensorStorage();

  friend class vTensor;

 private:
  // Context
  api::Context* context_{};

  api::StorageType storage_type_;

  // Resource sizings
  api::utils::uvec3 extents_{};
  int64_t buffer_length_{};

  // Image Texture
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
      const std::vector<int64_t>& gpu_sizes,
      const api::GPUMemoryLayout gpu_memory_layout,
      const api::ScalarType dtype);
};

class vTensor final {
 public:
  // Do not allow empty vTensor construction
  vTensor() = default;

  // Default constructor
  vTensor(
      api::Context* context,
      const std::vector<int64_t>& sizes,
      const api::ScalarType dtype,
      const api::StorageType storage_type = api::StorageType::TEXTURE_3D,
      const api::GPUMemoryLayout memory_layout =
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
      const bool allocate_memory = true);

  // Default constructor for quantized vTensor
  vTensor(
      api::Context* const context,
      const std::vector<int64_t>& sizes,
      double q_scale,
      int64_t q_zero_point,
      const api::ScalarType dtype,
      const api::StorageType storage_type = api::StorageType::TEXTURE_3D,
      const api::GPUMemoryLayout memory_layout =
          api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);

  // Copy Constructor and Assignment; Ideally copying  would be disabled
  // (see the reasoning for move assignment below) but it is required for
  // compatibility with OpaqueTensorImpl
  vTensor(const vTensor& other) = default;
  vTensor& operator=(const vTensor& other) = default;

  // Move Constructor and assignment
  vTensor(vTensor&& other) = default;
  vTensor& operator=(vTensor&& other) = default;

 private:
  // Tensor Options
  api::ScalarType dtype_;

  // GPU specific memory layout qualifier
  api::GPUMemoryLayout memory_layout_;

  // Sizes and Strides
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;

  // Storage Dimensions. When stored on the GPU, one dimension will be aligned
  // to the next multiple of 4 in order to take advantage of vec4 data types.
  std::vector<int64_t> gpu_sizes_;
  std::vector<int64_t> gpu_strides_;

  // The extents that correspond to the tensor's size metadata. Note that this
  // may not be the same as the extents of the underlying image texture because
  // vTensor can be virtually resized via virtual_resize() which will cause it
  // to be interpreted as a tensor with a different size.
  api::utils::uvec3 virtual_extents_;

  // A Vulkan uniform buffer containing the tensor sizes that can be passed into
  // a shader.
  std::shared_ptr<api::UniformParamsBuffer> cpu_sizes_uniform_;

  // A Vulkan uniform buffer containing the GPU tensor sizes that can be passed
  // into a shader. GPU sizes refers to the sizes of the tensor after padding
  // has been applied to one dimension to align it to the next multiple of 4.
  std::shared_ptr<api::UniformParamsBuffer> gpu_sizes_uniform_;

  // A Vulkan uniform buffer containing the image extents of the underlying
  // image texture that can be passed into a shader.
  std::shared_ptr<api::UniformParamsBuffer> extents_uniform_;

  // Quantization params
  bool is_quantized_{false};
  double q_scale_{1.0f};
  int64_t q_zero_point_{0u};

  // Even at the cost of a heap allocation plus the resulting negative impact
  // on cache locality due to the subsequent pointer chasing, it is still
  // critical to share the view across vTensor implementations to minimize
  // programmer errors.  Ideally this class should have been only made movable,
  // and non-copyable - something we cannot do unfortunately due to the inner
  // workings of at::TensorImpl requiring copy semantics in
  // at::TensorImpl::release_resources() to function as expected.  Now that this
  // class is made copyable though, a new door to a whole new class of bugs is
  // opened, in that there now is a chance of two [shallow] copies, have their
  // StorageState objects go out of sync as a result of an operation being
  // performed on one shallow copy that is not reflected in the other.
  // Technically, if the programmer is very careful, it is possible to avoid
  // this trap and not pay the cost of indirection, but the resulting bugs of
  // missing memory barriers will be so frustrating to hunt down for those
  // unfamiliar with the internal mechanics of this class, that I decided to
  // take the performance penalty of this extra layer of indirection in favor
  // of making this class easier to use.
  std::shared_ptr<vTensorStorage> view_;

 public:
  /*
   Texture Access
  */

  inline api::StorageType storage_type() const {
    return view_->storage_type_;
  }

  inline api::VulkanImage& image() const& {
    return view_->image_;
  }

  api::VulkanImage& image(api::PipelineBarrier&, const api::PipelineStageFlags)
      const&;

  api::VulkanImage& image(
      api::PipelineBarrier&,
      const api::PipelineStageFlags,
      const api::MemoryAccessFlags) &;

  inline api::VulkanBuffer& buffer() const& {
    return view_->buffer_;
  }

  api::VulkanBuffer& buffer(
      api::PipelineBarrier&,
      const api::PipelineStageFlags) const&;

  api::VulkanBuffer& buffer(
      api::PipelineBarrier&,
      const api::PipelineStageFlags,
      const api::MemoryAccessFlags) &;

  /*
    Metadata
  */

  inline const api::utils::uvec3& extents() const {
    return view_->extents_;
  }

  /*
   * Extract an `api::ScalarType` from the TensorOptions member
   */
  inline api::ScalarType dtype() const {
    return dtype_;
  }

  /*
   * Get an `api::ScalarType` that corresponds to the image format of the
   * texture
   */
  inline api::ScalarType texture_dtype() const {
    return api::element_scalartype(view_->texture_format());
  }

  inline api::GPUMemoryLayout gpu_memory_layout() const {
    return memory_layout_;
  }

  inline uint32_t gpu_memory_layout_as_uint() const {
    return static_cast<uint32_t>(memory_layout_);
  }

  inline const std::vector<int64_t>& sizes() const {
    return sizes_;
  }

  inline const std::vector<int64_t>& strides() const {
    return strides_;
  }

  inline const std::vector<int64_t>& gpu_sizes() const {
    return gpu_sizes_;
  }

  inline const std::vector<int64_t>& gpu_strides() const {
    return gpu_strides_;
  }

  inline const api::utils::uvec3& virtual_extents() const {
    return virtual_extents_;
  }

  /*
   * Get a uniform buffer object containing the tensor sizes to use in a compute
   * shader. Note that the UBO will be created the first time this function is
   * called.
   */
  std::shared_ptr<api::UniformParamsBuffer> cpu_sizes_ubo();

  /*
   * Get a uniform buffer object containing the tensor GPU sizes to use in a
   * compute shader. Note that the UBO will be created the first time this
   * function is called.
   */
  std::shared_ptr<api::UniformParamsBuffer> gpu_sizes_ubo();

  /*
   * Get a uniform buffer object containing the image extents to use in a
   * compute shader. Note that the UBO will be created the first time this
   * function is called.
   */
  std::shared_ptr<api::UniformParamsBuffer> extents_ubo();

  inline void set_is_quantized() {
    is_quantized_ = true;
  }

  inline bool is_quantized() const {
    return is_quantized_;
  }

  inline void set_scale(const double q_scale) {
    q_scale_ = q_scale;
  }

  inline double get_scale() const {
    return q_scale_;
  }

  inline float get_scale_float() const {
    return api::utils::safe_downcast<float>(q_scale_);
  }

  inline void set_zero_point(const int64_t q_zero_point) {
    q_zero_point_ = q_zero_point;
  }

  inline int64_t get_zero_point() const {
    return q_zero_point_;
  }

  inline int32_t get_zero_point_int32() const {
    return api::utils::safe_downcast<int32_t>(q_zero_point_);
  }

  inline size_t numel() const {
    return api::utils::multiply_integers(sizes());
  }

  inline size_t nbytes() const {
    return api::element_size(dtype()) * numel();
  }

  /*
   * Returns numel but based on gpu_sizes_ instead of sizes_
   */
  inline size_t gpu_numel() const {
    return api::utils::multiply_integers(gpu_sizes_);
  }

  /*
   * Return nbytes but bnased on gpu_sizes_ instead of sizes_
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
  void bind_allocation(const api::MemoryAllocation& allocation);

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

void add_buffer_barrier(
    api::PipelineBarrier&,
    const api::VulkanBuffer&,
    const api::PipelineStageFlags,
    const api::MemoryAccessFlags,
    const api::PipelineStageFlags,
    const api::MemoryAccessFlags);

} // namespace vkcompute
