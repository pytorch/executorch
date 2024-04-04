/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/GraphConfig.h>

#include <executorch/backends/vulkan/runtime/graph/containers/SharedObject.h>
#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/PrepackNode.h>

namespace vkcompute {

// Define valid scalar types that the Value class can
// accept
template <typename T>
struct is_valid_scalar_type : std::false_type {};

template <>
struct is_valid_scalar_type<int64_t> : std::true_type {};

template <>
struct is_valid_scalar_type<double> : std::true_type {};

template <>
struct is_valid_scalar_type<bool> : std::true_type {};

/*
 * This is the core data structure used to execute Vulkan models in graph mode.
 * As opposed to ATen/eager mode where a command buffer is encoded every
 * inference (since ops are executed with the model), in graph mode the ops that
 * compose the model are intended to be parsed only once, upon which a command
 * buffer will be encoded. Model inference will then execute the cached command
 * buffer without needing to encode a new one.
 */
class ComputeGraph final {
 public:
  explicit ComputeGraph(GraphConfig config);

  ComputeGraph(ComputeGraph&&) = default;
  ComputeGraph& operator=(ComputeGraph&&) = default;

  ~ComputeGraph();

 private:
  GraphConfig config_;
  api::DescriptorPoolConfig prepack_descriptor_counts_;
  api::DescriptorPoolConfig execute_descriptor_counts_;

  std::unique_ptr<api::Context> context_;
  std::vector<SharedObject> shared_objects_;
  std::vector<Value> values_;

  std::vector<std::unique_ptr<PrepackNode>> prepack_nodes_;
  std::vector<std::unique_ptr<ExecuteNode>> execute_nodes_;

  std::vector<IOValueRef> inputs_;
  std::vector<IOValueRef> outputs_;

 public:
  //
  // Accessors
  //

  inline api::Context* context() {
    return context_.get();
  }

  inline std::vector<IOValueRef>& inputs() {
    return inputs_;
  }

  inline std::vector<IOValueRef>& outputs() {
    return outputs_;
  }

  void update_descriptor_counts(
      const api::ShaderInfo& shader_info,
      bool execute);

  /*
   * Returns the value at a particular index in the graph. If storing this
   * function's return value in a lvalue reference, it is imperative that no
   * values are added to the graph while the reference is in scope, otherwise
   * the underlying value may have been moved as part of a vector resize.
   */
  inline Value& get_val(ValueRef idx) {
    return values_.at(idx);
  }

  inline const std::vector<int64_t>& get_val_sizes(ValueRef idx) {
    Value& val = get_val(idx);
    if (val.isTensor()) {
      return val.toTensor().sizes();
    } else if (val.isTensorRef()) {
      return val.toTensorRef().sizes;
    }
    VK_THROW("Could not get sizes of value with type ", val.type());
  }

  inline api::ScalarType get_val_dtype(ValueRef idx) {
    Value& val = get_val(idx);
    if (val.isTensor()) {
      return val.toTensor().dtype();
    } else if (val.isTensorRef()) {
      return val.toTensorRef().dtype;
    }
    VK_THROW("Could not get dtype of value with type ", val.type());
  }

  inline std::vector<std::unique_ptr<PrepackNode>>& prepack_nodes() {
    return prepack_nodes_;
  }

  inline std::vector<std::unique_ptr<ExecuteNode>>& execute_nodes() {
    return execute_nodes_;
  }

  //
  // Utility functions
  //

  /*
   * Returns a suggested storage type (i.e. buffer or texture) that can be used
   * to construct `vTensor`s. The storage type is typically determined by the
   * GPU reported by the Vulkan context, unless a storage type override is
   * defined in the graph configuration. Some GPU architectures work better with
   * buffer storage, and others with texture storage. Current only texture
   * storage is supported.
   */
  api::StorageType suggested_storage_type();

  /*
   * Returns a suggested memory layout (i.e. channels, width, or height packed)
   * that can be used to construct `vTensor`s. The memory layout impacts which
   * dimension will be treated as the vectorized dimension. For texture storage,
   * elements along the vectorized dimension are packed into texels. The
   * suggested memory layout is determined based on the sizes of the tensor,
   * unless a memory layout override is defined in the graph configuration.
   */
  api::GPUMemoryLayout suggested_memory_layout(
      const std::vector<int64_t>& sizes);

  /*
   * Returns the memory layout of a Tensor value at the specified index.
   */
  inline api::GPUMemoryLayout memory_layout_of(ValueRef idx) {
    return get_val(idx).toTensor().gpu_memory_layout();
  }

  //
  // Graph Building
  //

  /*
   * Add a `vTensor` value to the graph with the specified properties. There are
   * various convenience overloads of this function that may be used instead.
   */
  ValueRef add_tensor(
      const std::vector<int64_t>& sizes,
      const api::ScalarType dtype,
      const api::StorageType storage_type,
      const api::GPUMemoryLayout memory_layout,
      const int64_t shared_object_idx = -1);

  /*
   * Add a `vTensor` value to the graph with the specified properties. The
   * suggested storage type will be used to construct the `vTensor`.
   */
  ValueRef add_tensor(
      const std::vector<int64_t>& sizes,
      const api::ScalarType dtype,
      const api::GPUMemoryLayout memory_layout,
      const int64_t shared_object_idx = -1);

  /*
   * Add a `vTensor` value to the graph with the specified properties. The
   * suggested storage type and memory layout will be used to construct the
   * `vTensor`.
   */
  ValueRef add_tensor(
      const std::vector<int64_t>& sizes,
      const api::ScalarType dtype,
      const int64_t shared_object_idx = -1);

  /*
   * Add a `vTensor` value to the graph with the properties of `vref`.
   */
  ValueRef add_tensor_like(
      const ValueRef vref,
      const api::StorageType storage_type,
      const api::GPUMemoryLayout memory_layout);

  /*
   * Add a `vTensor` value to the graph with the properties of `vref`. The
   * suggested storage type will be used to construct the `vTensor`.
   */
  ValueRef add_tensor_like(
      const ValueRef vref,
      const api::GPUMemoryLayout memory_layout);

  /*
   * Add a `TensorRef` value to the graph with the specific properties. A
   * `TensorRef` is a reference to a `vTensor` whose data is stored in an
   * external CPU buffer.
   */
  ValueRef add_tensorref(
      const std::vector<int64_t>& sizes,
      const api::ScalarType dtype,
      const void* const data);

  /*
   * Add a staging buffer to the graph. Staging buffers are data buffers that
   * use memory that is visible to both the CPU and GPU, and therefore is used
   * as a intermediary when transferring data between the CPU and GPU.
   */
  ValueRef add_staging(const api::ScalarType dtype, const size_t numel);

  ValueRef add_none();

  template <typename T>
  typename std::enable_if<is_valid_scalar_type<T>::value, ValueRef>::type
  add_scalar(T value);

  template <typename T>
  typename std::enable_if<is_valid_scalar_type<T>::value, ValueRef>::type
  add_scalar_list(std::vector<T>&& value);

  ValueRef add_value_list(std::vector<ValueRef>&& value);

  ValueRef add_string(std::string&& str);

  ValueRef set_input_tensor(const ValueRef idx, const bool use_staging = true);
  ValueRef set_output_tensor(const ValueRef idx, const bool use_staging = true);

  template <typename Block>
  inline std::shared_ptr<api::UniformParamsBuffer> create_params_buffer(
      const Block& data) {
    return std::make_shared<api::UniformParamsBuffer>(context_.get(), data);
  }

  /*
   * Convenience function to add an input tensor along with its staging buffer
   */
  inline IOValueRef add_input_tensor(
      const std::vector<int64_t>& sizes,
      const api::ScalarType dtype,
      const int64_t shared_object_idx = -1) {
    ValueRef t = add_tensor(sizes, dtype, shared_object_idx);
    ValueRef staging = set_input_tensor(t);
    return {t, staging};
  }

  /*
   * Convenience function to add an input tensor with a specific memory layout
   * along with its staging buffer
   */
  inline IOValueRef add_input_tensor(
      const std::vector<int64_t>& sizes,
      const api::ScalarType dtype,
      const api::GPUMemoryLayout memory_layout,
      const int64_t shared_object_idx = -1) {
    ValueRef t = add_tensor(sizes, dtype, memory_layout, shared_object_idx);
    ValueRef staging = set_input_tensor(t);
    return {t, staging};
  }

  SharedObject& get_shared_object(const int64_t idx);

  //
  // Graph Preparation
  //

  void prepare();

  //
  // Input/Output
  //

  void
  copy_into_staging(const ValueRef idx, const void* data, const size_t numel);
  void copy_from_staging(const ValueRef idx, void* data, const size_t numel);

  //
  // Graph Prepacking
  //

  void encode_prepack();
  void prepack() const;

  //
  // Graph Execution
  //

  void encode_execute();
  void execute() const;

  //
  // Dynamic Shape support
  //

  void resize_input(const int64_t idx, const std::vector<int64_t>& new_sizes);
  void propagate_resize();
};

template <typename T>
inline typename std::enable_if<is_valid_scalar_type<T>::value, ValueRef>::type
ComputeGraph::add_scalar(T value) {
  ValueRef idx(static_cast<int>(values_.size()));
  values_.emplace_back(value);
  return idx;
}

template <typename T>
inline typename std::enable_if<is_valid_scalar_type<T>::value, ValueRef>::type
ComputeGraph::add_scalar_list(std::vector<T>&& value) {
  ValueRef idx(static_cast<int>(values_.size()));
  values_.emplace_back(std::move(value));
  return idx;
}

} // namespace vkcompute
