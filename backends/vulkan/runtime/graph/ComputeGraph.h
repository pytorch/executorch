/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <optional>

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

//
// Guarded Pointer Classes
//

class ComputeGraph;

#define DECL_VALUE_PTR_CLASS(classname, ctype)                         \
  class classname final {                                              \
    ComputeGraph* const graph_;                                        \
    ctype* ptr_;                                                       \
                                                                       \
   public:                                                             \
    explicit classname(ComputeGraph* const graph, const ValueRef idx); \
    ctype* operator->() const;                                         \
    ctype& operator*() const;                                          \
    ~classname();                                                      \
  };

DECL_VALUE_PTR_CLASS(vTensorPtr, api::vTensor)
DECL_VALUE_PTR_CLASS(TensorRefPtr, TensorRef)
DECL_VALUE_PTR_CLASS(StagingPtr, api::StorageBuffer)
DECL_VALUE_PTR_CLASS(IntListPtr, std::vector<int64_t>)
DECL_VALUE_PTR_CLASS(DoubleListPtr, std::vector<double>)
DECL_VALUE_PTR_CLASS(BoolListPtr, std::vector<bool>)
DECL_VALUE_PTR_CLASS(ValueListPtr, std::vector<ValueRef>)

#undef DECL_VALUE_PTR_CLASS

//
// ComputeGraph
//

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
  vkapi::DescriptorPoolConfig prepack_descriptor_counts_;
  vkapi::DescriptorPoolConfig execute_descriptor_counts_;

  std::unique_ptr<api::Context> context_;
  std::vector<SharedObject> shared_objects_;
  std::vector<Value> values_;
  std::vector<api::ParamsBuffer> param_ubos_;

  std::vector<std::unique_ptr<PrepackNode>> prepack_nodes_;
  std::vector<std::unique_ptr<ExecuteNode>> execute_nodes_;

  std::vector<IOValueRef> inputs_;
  std::vector<IOValueRef> outputs_;

 protected:
  size_t values_in_use_ = 0;

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

  inline std::vector<std::unique_ptr<PrepackNode>>& prepack_nodes() {
    return prepack_nodes_;
  }

  inline std::vector<std::unique_ptr<ExecuteNode>>& execute_nodes() {
    return execute_nodes_;
  }

  inline GraphConfig& graphconfig() {
    return config_;
  }

  //
  // Value Extraction
  //

#define GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(ptr_type, short_name, type_name) \
  inline ptr_type get_##short_name(const ValueRef idx) {                   \
    return ptr_type(this, idx);                                            \
  }                                                                        \
  inline bool val_is_##short_name(const ValueRef idx) {                    \
    return values_.at(idx).is##type_name();                                \
  }

  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(vTensorPtr, tensor, Tensor)
  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(TensorRefPtr, tref, TensorRef)
  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(StagingPtr, staging, Staging)
  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(IntListPtr, int_list, IntList)
  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(DoubleListPtr, double_list, DoubleList)
  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(BoolListPtr, bool_list, BoolList)
  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(ValueListPtr, value_list, ValueList)

#undef GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS

#define GET_AND_CHECK_VAL_AS_TYPE_FNS(ctype, short_name, type_name) \
  inline ctype get_##short_name(const ValueRef idx) {               \
    return values_.at(idx).to##type_name();                         \
  }                                                                 \
  inline bool val_is_##short_name(const ValueRef idx) {             \
    return values_.at(idx).is##type_name();                         \
  }

  GET_AND_CHECK_VAL_AS_TYPE_FNS(int64_t, int, Int)
  GET_AND_CHECK_VAL_AS_TYPE_FNS(double, double, Double)
  GET_AND_CHECK_VAL_AS_TYPE_FNS(bool, bool, Bool)
  GET_AND_CHECK_VAL_AS_TYPE_FNS(std::string, string, String)

#undef GET_AND_CHECK_VAL_AS_TYPE_FNS

  inline bool val_is_none(const ValueRef idx) {
    return values_.at(idx).isNone();
  }

  inline TypeTag get_val_type(const ValueRef idx) {
    return values_.at(idx).type();
  }

  // Get Tensor Property

  std::vector<int64_t> sizes_of(const ValueRef idx) const;

  vkapi::ScalarType dtype_of(const ValueRef idx) const;

  inline utils::uvec3 image_extents_of(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().image_extents();
  }

  inline int32_t texel_numel_of(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().texel_numel();
  }

  inline utils::StorageType storage_type_of(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().storage_type();
  }

  inline bool is_buffer_storage(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().has_buffer_storage();
  }

  inline utils::GPUMemoryLayout memory_layout_of(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().gpu_memory_layout();
  }

  inline int32_t packed_dim_whcn_idx_of(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().packed_dim_whcn_idx();
  }

  inline vkapi::BufferBindInfo sizes_ubo(const ValueRef idx) {
    return values_.at(idx).toTensor().sizes_ubo();
  }

  inline vkapi::BufferBindInfo texture_limits_ubo(const ValueRef idx) {
    return values_.at(idx).toTensor().texture_limits_ubo();
  }

  inline vkapi::BufferBindInfo texel_strides_ubo(const ValueRef idx) {
    return values_.at(idx).toTensor().texel_strides_ubo();
  }

  inline vkapi::BufferBindInfo ntexels_ubo(const ValueRef idx) {
    return values_.at(idx).toTensor().ntexels_ubo();
  }

  // Scalar Value Extraction

  template <typename T>
  T extract_scalar(const ValueRef idx) {
    Value& value = values_.at(idx);
    if (value.isInt()) {
      return static_cast<T>(value.toInt());
    }
    if (value.isDouble()) {
      return static_cast<T>(value.toDouble());
    }
    if (value.isBool()) {
      return static_cast<T>(value.toBool());
    }
    VK_THROW("Cannot extract scalar from Value with type ", value.type());
  }

  template <typename T>
  std::optional<T> extract_optional_scalar(const ValueRef idx) {
    if (val_is_none(idx)) {
      return ::std::nullopt;
    } else {
      return extract_scalar<T>(idx);
    }
  }

  std::string extract_string(const ValueRef idx) {
    return values_.at(idx).toString();
  }

  //
  // Utility functions
  //

  /*
   * Returns a suggested storage type (i.e. buffer or texture) that can be used
   * to construct `api::vTensor`s. The storage type is typically determined by
   * the GPU reported by the Vulkan context, unless a storage type override is
   * defined in the graph configuration. Some GPU architectures work better with
   * buffer storage, and others with texture storage. Current only texture
   * storage is supported.
   */
  utils::StorageType suggested_storage_type();

  /*
   * Returns a suggested memory layout (i.e. channels, width, or height packed)
   * that can be used to construct `api::vTensor`s. The memory layout impacts
   * which dimension will be treated as the vectorized dimension. For texture
   * storage, elements along the vectorized dimension are packed into texels.
   * The suggested memory layout is determined based on the sizes of the tensor,
   * unless a memory layout override is defined in the graph configuration.
   */
  utils::GPUMemoryLayout suggested_memory_layout(
      const std::vector<int64_t>& sizes);

  //
  // Graph Building
  //

 private:
  void check_no_active_value_ptrs();

 public:
  /*
   * Add a `api::vTensor` value to the graph with the specified properties.
   * There are various convenience overloads of this function that may be used
   * instead.
   */
  ValueRef add_tensor(
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const utils::StorageType storage_type,
      const utils::GPUMemoryLayout memory_layout,
      const int64_t shared_object_idx = -1);

  /*
   * Add a `api::vTensor` value to the graph with the specified properties. The
   * suggested memory layout will be used to construct the `api::vTensor`.
   */
  ValueRef add_tensor(
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const utils::StorageType storage_type,
      const int64_t shared_object_idx = -1);

  /*
   * Add a `api::vTensor` value to the graph with the specified properties. The
   * suggested storage type will be used to construct the `api::vTensor`.
   */
  ValueRef add_tensor(
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const utils::GPUMemoryLayout memory_layout,
      const int64_t shared_object_idx = -1);

  /*
   * Add a `api::vTensor` value to the graph with the specified properties. The
   * suggested storage type and memory layout will be used to construct the
   * `api::vTensor`.
   */
  ValueRef add_tensor(
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const int64_t shared_object_idx = -1);

  /*
   * Add a `api::vTensor` value to the graph with the properties of `vref`.
   */
  ValueRef add_tensor_like(
      const ValueRef vref,
      const utils::StorageType storage_type,
      const utils::GPUMemoryLayout memory_layout);

  /*
   * Add a `api::vTensor` value to the graph with the properties of `vref`. The
   * suggested storage type will be used to construct the `api::vTensor`.
   */
  ValueRef add_tensor_like(
      const ValueRef vref,
      const utils::GPUMemoryLayout memory_layout);

  /*
   * Add a `TensorRef` value to the graph with the specific properties. A
   * `TensorRef` is a reference to a `api::vTensor` whose data is stored in an
   * external CPU buffer.
   */
  ValueRef add_tensorref(
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const void* const data);

  /*
   * Add a staging buffer to the graph. Staging buffers are data buffers that
   * use memory that is visible to both the CPU and GPU, and therefore is used
   * as a intermediary when transferring data between the CPU and GPU.
   */
  ValueRef add_staging(const vkapi::ScalarType dtype, const size_t numel);

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
  const vkapi::BufferBindInfo create_params_buffer(const Block& data) {
    param_ubos_.emplace_back(api::ParamsBuffer(context_.get(), data));
    return vkapi::BufferBindInfo(param_ubos_.back().buffer());
  }

  /*
   * Convenience function to add an input tensor along with its staging buffer
   */
  inline IOValueRef add_input_tensor(
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
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
      const vkapi::ScalarType dtype,
      const utils::GPUMemoryLayout memory_layout,
      const int64_t shared_object_idx = -1) {
    ValueRef t = add_tensor(sizes, dtype, memory_layout, shared_object_idx);
    ValueRef staging = set_input_tensor(t);
    return {t, staging};
  }

  /*
   * Convenience function to add an input tensor with a specific storage type
   * along with its staging buffer
   */
  inline IOValueRef add_input_tensor(
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const utils::StorageType storage_type,
      const int64_t shared_object_idx = -1) {
    ValueRef t = add_tensor(sizes, dtype, storage_type, shared_object_idx);
    ValueRef staging = set_input_tensor(t);
    return {t, staging};
  }

  SharedObject& get_shared_object(const int64_t idx);

  //
  // Graph Preparation
  //

  void update_descriptor_counts(
      const vkapi::ShaderInfo& shader_info,
      bool execute);

  void prepare();

  //
  // Dispatch Utilities
  //

  /*
   * Create a global workgroup size for a given `api::vTensor` value assuming
   * that every shader invocation calculates one texel element of the output
   * tensor.
   *
   * For tensors that use texture storage, the image extents of the
   * `api::vTensor` will be used to set the global workgroup size.
   *
   * For tensor that use buffer storage, the number of texels in the texel
   * buffer will be used to set the x component of the global workgroup size.
   * All other components will be set to 1 (i.e. {ntexels, 1, 1} will be
   * returned).
   */
  utils::uvec3 create_global_wg_size(const ValueRef idx);

  /*
   * Suggest a local workgroup size for a given `api::vTensor` value, assuming
   * that every shader invocation calculates one texel element of the output
   * tensor.
   *
   * The local workgroup size will be formed to try and minimize the number of
   * inactive invocations.
   *
   * Currently, the local workgroup size is hard-coded to contain a total of 64
   * shader invocations. In the future, this value can be configured.
   */
  utils::uvec3 create_local_wg_size(const ValueRef idx);

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

  //
  // Debug support (implemented in Logging.cpp)
  //

  void print_readable();

  //
  // Friend classes
  //

  friend class vTensorPtr;
  friend class TensorRefPtr;
  friend class StagingPtr;
  friend class IntListPtr;
  friend class DoubleListPtr;
  friend class BoolListPtr;
  friend class ValueListPtr;
};

template <typename T>
inline typename std::enable_if<is_valid_scalar_type<T>::value, ValueRef>::type
ComputeGraph::add_scalar(T value) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(value);
  return idx;
}

template <typename T>
inline typename std::enable_if<is_valid_scalar_type<T>::value, ValueRef>::type
ComputeGraph::add_scalar_list(std::vector<T>&& value) {
  ValueRef idx(static_cast<int>(values_.size()));
  check_no_active_value_ptrs();
  values_.emplace_back(std::move(value));
  return idx;
}

} // namespace vkcompute
