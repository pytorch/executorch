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
#include <stack>

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/GraphConfig.h>

#include <executorch/backends/vulkan/runtime/graph/containers/SharedObject.h>
#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/PrepackNode.h>

#ifdef ET_EVENT_TRACER_ENABLED
std::string& set_and_get_current_operator_json(const std::string& json);
size_t get_current_operator_count(const bool increment = false);
#endif

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
DECL_VALUE_PTR_CLASS(StagingPtr, api::StagingBuffer)
DECL_VALUE_PTR_CLASS(IntListPtr, std::vector<int64_t>)
DECL_VALUE_PTR_CLASS(DoubleListPtr, std::vector<double>)
DECL_VALUE_PTR_CLASS(BoolListPtr, std::vector<bool>)
DECL_VALUE_PTR_CLASS(ValueListPtr, std::vector<ValueRef>)
DECL_VALUE_PTR_CLASS(SymIntPtr, SymInt);

#undef DECL_VALUE_PTR_CLASS

//
// TmpTensor
//

/*
 * This struct is used to recycle the memory of temporary tensors that are
 * created during the execution of a node. Upon construction, this struct will
 * check the `tmp_shared_object_idxs_` of the provided `ComputeGraph` instance
 * if any shared objects are available; if not, then a new one is created. A
 * tensor value is then added to the `ComputeGraph` instance with the requested
 * specifications. Upon destruction, the shared object index of the temporary
 * tensor is returned to `tmp_shared_object_idxs_`.
 *
 * Note that instances of this struct can be used as if they were `ValueRef` due
 * to implementation of a custom casting operator.
 *
 * This class should only be used to create tensors whose lifetimes exist only
 * in a well defined scope (i.e. within a function).
 */
struct TmpTensor {
  ComputeGraph* graph_p;
  int64_t sobj_idx;
  ValueRef vref;

  //
  // Match all available overloads of `add_tensor`
  //

  TmpTensor(
      ComputeGraph* const graph_ptr,
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const utils::StorageType storage_type,
      const utils::GPUMemoryLayout memory_layout);

  TmpTensor(
      ComputeGraph* const graph_ptr,
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const utils::StorageType storage_type);

  TmpTensor(
      ComputeGraph* const graph_ptr,
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const utils::GPUMemoryLayout memory_layout);

  TmpTensor(
      ComputeGraph* const graph_ptr,
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype);

  // No copy construction or assignment
  TmpTensor(TmpTensor& other) = delete;
  TmpTensor& operator=(TmpTensor& other) = delete;

  // No move construction or assignment
  TmpTensor(TmpTensor&& other) = delete;
  TmpTensor& operator=(TmpTensor&& other) = delete;

  // Custom cast to ValueRef
  operator ValueRef() const {
    return vref;
  };

  ~TmpTensor();

 private:
  // Helper function to get first available shared object index or request a new
  // one to be created.
  int64_t get_sobj_idx();
};

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
  // This stack is used by `TmpTensor` instances to recycle shared objects
  // for temporary tensors. See the comments of `TmpTensor` for more details
  std::stack<int64_t> tmp_shared_object_idxs_;

  std::vector<Value> values_;
  std::vector<api::ParamsBuffer> param_ubos_;

  std::vector<std::unique_ptr<PrepackNode>> prepack_nodes_;
  std::vector<std::unique_ptr<ExecuteNode>> execute_nodes_;

  std::vector<IOValueRef> inputs_;
  std::vector<IOValueRef> outputs_;

  std::unordered_set<
      vkapi::ComputePipelineCache::Key,
      vkapi::ComputePipelineCache::Hasher>
      pipeline_descriptors_;

  // Utility constexpr to express byte quantities
  constexpr static size_t MB = 1024 * 1024;

  // List of command buffers deferred for submission
  std::vector<vkapi::CommandBuffer> deferred_cmd_list_;

  // Set to track which ValueRefs were updated during inference
  std::unordered_set<ValueRef> updated_values_;

  // Flag to indicate if re-encoding is required
  bool requires_reencode_ = false;

 protected:
  size_t values_in_use_ = 0;
  size_t execute_count_ = 0;

  // Total number of bytes needed to store model weights
  size_t total_constant_nbytes_ = 0;

  // Represents the amount of staging buffer data that will be copied if the
  // current Context's command buffer is submitted now.
  size_t staging_nbytes_in_cmd_ = 0;

  // Represents the nodes to wait before submitting commands.
  // If command buffers created with config.execute_threshold_node_count exceeds
  // config.execute_max_cmds, then execute_threshold_node_count will be
  // increased to fit command buffers within the limit. Otherwise,
  // execute_threshold_node_count will be set to
  // config.execute_threshold_node_count.
  size_t execute_threshold_node_count_ = 0;

  // Whether the underlying GPU support accelerated integer dot product
  // extensions
  bool can_use_int8_dot_product_ = false;

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

  // Check if the ComputeGraph has a value at the specified index
  bool is_valid_value_idx(const ValueRef idx) const noexcept;

  //
  // Value Extraction
  //

#define GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(ptr_type, short_name, type_name) \
  inline ptr_type get_##short_name(const ValueRef idx) {                   \
    return ptr_type(this, idx);                                            \
  }                                                                        \
  inline bool val_is_##short_name(const ValueRef idx) const {              \
    return values_.at(idx).is##type_name();                                \
  }

 protected:
  inline vTensorPtr get_tensor(const ValueRef idx) {
    return vTensorPtr(this, idx);
  }

 public:
  inline bool val_is_tensor(const ValueRef idx) const {
    return values_.at(idx).isTensor();
  }

  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(TensorRefPtr, tref, TensorRef)
  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(StagingPtr, staging, Staging)
  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(IntListPtr, int_list, IntList)
  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(DoubleListPtr, double_list, DoubleList)
  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(BoolListPtr, bool_list, BoolList)
  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(ValueListPtr, value_list, ValueList)
  GET_AND_CHECK_VAL_AS_PTR_TYPE_FNS(SymIntPtr, symint, SymInt);

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
    return idx == kDummyValueRef ? true : values_.at(idx).isNone();
  }

  inline bool val_is_not_none(const ValueRef idx) {
    return !val_is_none(idx);
  }

  inline TypeTag get_val_type(const ValueRef idx) {
    return values_.at(idx).type();
  }

  //
  // Tensor Properties Accessors
  //

  std::vector<int64_t> sizes_of(const ValueRef idx) const;

  /*
   * Returns the size of the tensor at `idx` along the specified dimension.
   * Negative indexing is allowed.
   */
  template <typename T>
  T size_at(const int64_t dim, const ValueRef idx) const {
    const Value& val = values_.at(idx);
    if (val.isTensor()) {
      return static_cast<T>(utils::val_at(dim, val.toConstTensor().sizes()));
    } else if (val.isTensorRef()) {
      return static_cast<T>(utils::val_at(dim, val.toConstTensorRef().sizes));
    }
    VK_THROW("Could not get sizes of value with type ", val.type());
  }

  int64_t dim_of(const ValueRef idx) const;

  std::vector<int64_t> dim_order_of(const ValueRef idx) const;

  std::vector<int64_t> strides_of(const ValueRef idx) const;

  vkapi::ScalarType dtype_of(const ValueRef idx) const;

  inline const utils::ivec3& logical_limits_of(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().logical_limits();
  }

  inline int32_t numel_of(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().numel();
  }

  inline size_t staging_buffer_numel_of(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().staging_buffer_numel();
  }

  inline utils::StorageType storage_type_of(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().storage_type();
  }

  inline bool is_buffer_storage(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().has_buffer_storage();
  }

  inline bool is_texture_storage(const ValueRef idx) const {
    return !is_buffer_storage(idx);
  }

  /*
   * Checks that the following is true:
   * 1. The value at `idx` is a tensor
   * 2. The tensor at `idx` has buffer storage
   * 3. The buffer backed tensor at `idx` has a contiguous memory layout
   */
  bool is_contiguous_buffer_tensor(const ValueRef idx) const;

  /*
   * Checks that the following is true:
   * 1. The value at `idx` is a tensor
   * 2. The tensor at `idx` has texture storage
   * 3. The texture backed tensor at `idx` has a standard axis mapping
   * 4. The texture backed tensor at `idx` is width packed
   */
  bool is_contiguous_texture_tensor(const ValueRef idx) const;

  /*
   * Checks that the following is true:
   * 1. The value at `idx` is a tensor
   * 2. The tensor at `idx` has texture storage
   * 3. The texture backed tensor at `idx` has a standard axis mapping
   * 4. The texture backed tensor at `idx` is channels packed
   */
  bool is_standard_channels_packed_texture_tensor(const ValueRef idx) const;

  /*
   * Checks that the value at `idx` is either a 2D tensor, or if the tensor has
   * more than 2 dims, the outermost dims have size of 1, i.e. can be squeezed
   * to be a 2D tensor.
   */
  bool is_2d_matrix(const ValueRef idx) const;

  /*
   * Same as the above, but also requires that the tensor is a contiguous
   * buffer with a width divisible by 4 or a standard width packed texture.
   */
  bool is_vectorizable_contiguous_2d_matrix(const ValueRef idx) const;

  /*
   * Checks that the following is true:
   * 1. The value at `idx` is a tensor
   * 2. The tensor at `idx` is width packed
   * 3. The tensor at `idx` has a standard axis mapping or is a contiguous
   * buffer
   */
  bool is_vectorizable_width_packed_tensor(const ValueRef idx) const;

  inline bool val_is_view_of(const ValueRef maybe_view, const ValueRef base)
      const {
    return values_.at(maybe_view)
        .toConstTensor()
        .is_view_of(values_.at(base).toConstTensor());
  }

  inline utils::GPUMemoryLayout estimate_memory_layout_of(
      const ValueRef idx) const {
    return values_.at(idx).toConstTensor().estimate_memory_layout();
  }

  inline int32_t hashed_layout_of(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().hashed_layout();
  }

  inline int32_t packed_dim_of(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().packed_dim();
  }

  inline const api::PackedDimInfo& packed_dim_info_of(
      const ValueRef idx) const {
    return values_.at(idx).toConstTensor().packed_dim_info();
  }

  inline int32_t concat_dim_of(const ValueRef idx) const {
    return values_.at(idx).toConstTensor().concat_dim();
  }

  inline vkapi::BufferBindInfo sizes_ubo(const ValueRef idx) {
    return values_.at(idx).toTensor().sizes_ubo();
  }

  inline vkapi::BufferBindInfo buffer_meta_ubo(const ValueRef idx) {
    return values_.at(idx).toTensor().buffer_meta_ubo();
  }

  inline vkapi::BufferBindInfo texture_meta_ubo(const ValueRef idx) {
    return values_.at(idx).toTensor().texture_meta_ubo();
  }

  inline vkapi::BufferBindInfo meta_ubo(const ValueRef idx) {
    if (is_buffer_storage(idx)) {
      return buffer_meta_ubo(idx);
    } else {
      return texture_meta_ubo(idx);
    }
  }

  inline vkapi::BufferBindInfo strides_ubo(const ValueRef idx) {
    return values_.at(idx).toTensor().strides_ubo();
  }

  inline vkapi::BufferBindInfo dim_order_ubo(const ValueRef idx) {
    return values_.at(idx).toTensor().dim_order_ubo();
  }

  inline vkapi::BufferBindInfo numel_ubo(const ValueRef idx) {
    return values_.at(idx).toTensor().numel_ubo();
  }

  inline bool has_standard_axis_map(const ValueRef idx) const {
    return values_.at(idx).toTensor().has_standard_axis_map();
  }

  inline bool is_contiguous(const ValueRef idx) const {
    return values_.at(idx).toTensor().is_contiguous();
  }

  inline vkapi::BufferBindInfo logical_limits_ubo(const ValueRef idx) {
    return values_.at(idx).toTensor().logical_limits_ubo();
  }

  inline PushConstantDataInfo sizes_pc_of(const ValueRef idx) const {
    PushConstantDataInfo pc_data = PushConstantDataInfo(
        values_.at(idx).toConstTensor().get_uniform_data(), api::kTensorSizes);
    pc_data.set_value(idx);
    return pc_data;
  }

  inline PushConstantDataInfo dim_order_pc_of(const ValueRef idx) const {
    PushConstantDataInfo pc_data = PushConstantDataInfo(
        values_.at(idx).toConstTensor().get_uniform_data(),
        api::kTensorDimOrder);
    pc_data.set_value(idx);
    return pc_data;
  }

  inline PushConstantDataInfo strides_pc_of(const ValueRef idx) const {
    PushConstantDataInfo pc_data = PushConstantDataInfo(
        values_.at(idx).toConstTensor().get_uniform_data(),
        api::kTensorStrides);
    pc_data.set_value(idx);
    return pc_data;
  }

  inline PushConstantDataInfo logical_limits_pc_of(const ValueRef idx) const {
    PushConstantDataInfo pc_data = PushConstantDataInfo(
        values_.at(idx).toConstTensor().get_uniform_data(),
        api::kTensorLogicalLimits);
    pc_data.set_value(idx);
    return pc_data;
  }

  inline PushConstantDataInfo numel_pc_of(const ValueRef idx) const {
    PushConstantDataInfo pc_data = PushConstantDataInfo(
        values_.at(idx).toConstTensor().get_uniform_data(), api::kTensorNumel);
    pc_data.set_value(idx);
    return pc_data;
  }

  //
  // Scalar Value Extraction
  //

  bool is_scalar_or_none(const ValueRef idx) const {
    const Value& value = values_.at(idx);
    return value.isInt() || value.isDouble() || value.isBool() ||
        value.isNone();
  }

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
  T extract_scalar_or(const ValueRef idx, const T default_value) {
    Value& value = values_.at(idx);
    if (value.isNone()) {
      return default_value;
    }
    return extract_scalar<T>(idx);
  }

  template <typename T>
  std::optional<T> extract_optional_scalar(const ValueRef idx) {
    if (val_is_none(idx)) {
      return ::std::nullopt;
    } else if (val_is_symint(idx)) {
      return utils::safe_downcast<T>(read_symint(idx));
    } else {
      return extract_scalar<T>(idx);
    }
  }

  template <typename T>
  T extract_optional_scalar(const ValueRef idx, const T default_val) {
    if (val_is_none(idx)) {
      return default_val;
    } else if (val_is_symint(idx)) {
      return utils::safe_downcast<T>(read_symint(idx));
    } else {
      return extract_scalar<T>(idx);
    }
  }

  std::string extract_string(const ValueRef idx) {
    return values_.at(idx).toString();
  }

  /*
   * Utility function to extract a list of integers from a ValueRef.
   * If the ValueRef is an IntList, returns a copy of the list.
   * If the ValueRef is a ValueList, extracts each element as an Int or SymInt
   * and returns the resulting list.
   * Throws an error if the ValueRef is neither an IntList nor a ValueList.
   */
  std::vector<int64_t> extract_int_or_symint_list(const ValueRef idx);

  template <
      typename T,
      typename std::enable_if<
          std::is_integral<T>::value && std::is_signed<T>::value,
          int>::type = 0>
  T extract_whcn_dim(const ValueRef idx, const int64_t ndim) {
    T dim = extract_scalar<T>(idx);
    // Normalize dim to account for negative indexing
    dim = (dim % ndim + ndim) % ndim;
    // Assume original value is NCHW ordering, obtain the WHCN ordering
    return ndim - 1 - dim;
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

  inline bool device_is_adreno() {
    return context_->adapter_ptr()->device_type() == vkapi::DeviceType::ADRENO;
  }
  const std::string& device_name() {
    return context()->adapter_ptr()->device_name();
  }

  bool device_name_contains(const char* substr);

  int64_t max_buffer_numel() {
    return static_cast<int64_t>(context_->adapter_ptr()->max_buffer_numel());
  }

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
      const int64_t shared_object_idx = -1,
      const utils::AxisMapLayout axis_map_layout = utils::kDefaultAxisMap);

  /*
   * Add a `api::vTensor` value to the graph with the specified properties. The
   * suggested memory layout will be used to construct the `api::vTensor`.
   */
  ValueRef add_tensor(
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const utils::StorageType storage_type,
      const int64_t shared_object_idx = -1,
      const utils::AxisMapLayout axis_map_layout = utils::kDefaultAxisMap);

  /*
   * Add a `api::vTensor` value to the graph with the specified properties. The
   * suggested storage type will be used to construct the `api::vTensor`.
   */
  ValueRef add_tensor(
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const utils::GPUMemoryLayout memory_layout,
      const int64_t shared_object_idx = -1,
      const utils::AxisMapLayout axis_map_layout = utils::kDefaultAxisMap);

  /*
   * Add a `api::vTensor` value to the graph with the specified properties. The
   * suggested storage type and memory layout will be used to construct the
   * `api::vTensor`.
   */
  ValueRef add_tensor(
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const int64_t shared_object_idx = -1,
      const utils::AxisMapLayout axis_map_layout = utils::kDefaultAxisMap);

  /*
   * Add a `api::vTensor` value to the graph with the specified image.
   */
  ValueRef add_tensor(const vkapi::VulkanImage& image);

  /*
   * Add a `api::vTensor` value to the graph with the properties of `vref`.
   */
  ValueRef add_tensor_like(
      const ValueRef vref,
      const utils::StorageType storage_type,
      const utils::GPUMemoryLayout memory_layout,
      const utils::AxisMapLayout axis_map_layout = utils::kDefaultAxisMap);

  /*
   * Add a `api::vTensor` value to the graph with the properties of `vref`. The
   * suggested storage type will be used to construct the `api::vTensor`.
   */
  ValueRef add_tensor_like(
      const ValueRef vref,
      const utils::GPUMemoryLayout memory_layout,
      const utils::AxisMapLayout axis_map_layout = utils::kDefaultAxisMap);

  /*
   * Use the copy constructor of `api::vTensor` to create a "view" of the
   * `vTensor` value at `vref`. See the copy constructor of `api::vTensor` for
   * more details.
   */
  ValueRef add_tensor_view(const ValueRef vref);

  /*
   * Use the copy constructor of `api::vTensor` to create a "view" of the
   * `vTensor` value at `vref` with different sizes and dim order. See the copy
   * constructor of `api::vTensor` for more details.
   */
  ValueRef add_tensor_view(
      const ValueRef vref,
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& dim_order);

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
   * Add a `TensorRef` value to the graph with the specific properties. A
   * `TensorRef` is a reference to a `api::vTensor` whose data is stored in a
   * FreeableBuffer. The TensorRef will take ownership of the FreeableBuffer.
   */
  ValueRef add_tensorref(
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      executorch::runtime::FreeableBuffer&& buffer);

  /*
   * Add a staging buffer to the graph. Staging buffers are data buffers that
   * use memory that is visible to both the CPU and GPU, and therefore is used
   * as a intermediary when transferring data between the CPU and GPU.
   */
  ValueRef add_staging(
      const vkapi::ScalarType dtype,
      const size_t numel,
      const vkapi::CopyDirection direction);

  ValueRef add_none();

  template <typename T>
  typename std::enable_if<is_valid_scalar_type<T>::value, ValueRef>::type
  add_scalar(T value);

  template <typename T>
  typename std::enable_if<is_valid_scalar_type<T>::value, ValueRef>::type
  add_scalar_list(std::vector<T>&& value);

  ValueRef add_value_list(std::vector<ValueRef>&& value);

  ValueRef add_string(std::string&& str);

  ValueRef add_symint(const int32_t val);

  /*
   * Searches the graph's value list for a Int value with the specified value.
   * If one is found, returns the index of the value. Otherwise, add a new value
   * and return the index of the new value.
   */
  ValueRef get_or_add_value_for_int(const int64_t val);

  ValueRef set_input_tensor(
      const ValueRef idx,
      vkapi::ScalarType staging_dtype);

  ValueRef set_input_tensor(const ValueRef idx, const bool use_staging = true);

  ValueRef set_output_tensor(
      const ValueRef idx,
      vkapi::ScalarType staging_dtype);

  ValueRef set_output_tensor(const ValueRef idx, const bool use_staging = true);

  ValueRef set_output_value(const ValueRef idx);

  template <typename Block>
  vkapi::BufferBindInfo create_params_buffer(const Block& data) {
    param_ubos_.emplace_back(api::ParamsBuffer(context_.get(), data));
    return vkapi::BufferBindInfo(param_ubos_.back().buffer());
  }

  /*
   * Given a ValueRef, do the following depending on the type of the Value:
   * - If it is a SymInt, return the BufferBindInfo of the ParamsBuffer object
   *   backing the SymInt.
   * - If it is a regular Int, create a new ParamsBuffer using the integer value
   *   and return the BufferBindInfo of the created ParamsBuffer.
   */
  vkapi::BufferBindInfo get_or_create_int_param_buffer(const ValueRef idx);

  vkapi::BufferBindInfo get_or_create_int_param_buffer(
      const ValueRef idx,
      const int32_t default_value);

  void set_symint(const ValueRef idx, const int32_t val);

  int32_t read_symint(const ValueRef idx);

  inline void set_val_as_input(const ValueRef idx) {
    inputs_.push_back({idx, kDummyValueRef});
  }

  ValueRef staging_of(const ValueRef idx);

  inline void set_val_as_output(const ValueRef idx) {
    outputs_.push_back({idx, kDummyValueRef});
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

  /*
   * Add an input tensor with the specified properties along with its staging
   * buffer.
   */
  inline IOValueRef add_input_tensor(
      const std::vector<int64_t>& sizes,
      const vkapi::ScalarType dtype,
      const utils::StorageType storage_type,
      const utils::GPUMemoryLayout memory_layout,
      const int64_t shared_object_idx = -1) {
    ValueRef t = add_tensor(
        sizes, dtype, storage_type, memory_layout, shared_object_idx);
    ValueRef staging = set_input_tensor(t);
    return {t, staging};
  }

  SharedObject& get_shared_object(const int64_t idx);

  /*
   * Creates a dedicated memory allocation for a vTensor value, and have the
   * tensor acquire the allocation object. If the tensor is already bound to a
   * memory allocation, this function will be a no-op.
   */
  void create_dedicated_allocation_for(const ValueRef idx);

  //
  // Graph Preparation
  //

  void update_descriptor_counts(
      const vkapi::ShaderInfo& shader_info,
      bool execute);

  void register_pipeline_to_create(
      const vkapi::ShaderInfo& shader_info,
      const utils::WorkgroupSize& local_workgroup_size,
      const vkapi::SpecVarList& spec_vars,
      const std::vector<PushConstantDataInfo>& push_constants);

  void prepare();

  void prepare_pipelines();

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
   * Suggest a local workgroup size for a given global workgroup size.
   *
   * The local workgroup size will be formed to try and minimize the number of
   * inactive invocations.
   *
   * Currently, the local workgroup size is hard-coded to contain a total of 64
   * shader invocations. In the future, this value can be configured.
   */
  utils::uvec3 create_local_wg_size(const utils::uvec3 global_wg_size);

  /*
   * Convenience function to suggest a local workgroup size for a given
   * `api::vTensor` value, assuming that every shader invocation calculates one
   * texel element of the output tensor.
   */
  utils::uvec3 create_local_wg_size(const ValueRef idx);

  void bind_tensor_to_descriptor_set(
      const ValueRef ref,
      vkapi::PipelineBarrier& pipeline_barrier,
      const vkapi::MemoryAccessFlags accessType,
      vkapi::DescriptorSet& descriptor_set,
      const uint32_t idx);

  void bind_value_to_descriptor_set(
      const ValueRef ref,
      vkapi::PipelineBarrier& pipeline_barrier,
      const vkapi::MemoryAccessFlags access_type,
      vkapi::DescriptorSet& descriptor_set,
      const uint32_t idx);

  //
  // Input/Output
  //

  void
  copy_into_staging(const ValueRef idx, const void* data, const size_t numel);

  void maybe_cast_and_copy_into_staging(
      const ValueRef idx,
      const void* data,
      const size_t numel,
      const vkapi::ScalarType src_data_dtype);

  void copy_from_staging(const ValueRef idx, void* data, const size_t numel);

  void maybe_cast_and_copy_from_staging(
      const ValueRef idx,
      void* data,
      const size_t numel,
      const vkapi::ScalarType dst_data_dtype);

 protected:
  // Command Buffer Management

  /*
   * Submits the current command buffer in the Context to the GPU for execution.
   */
  void submit_current_cmd(const bool final_use = false);

  /*
   * Submits the current command buffer in the Context to the GPU for execution,
   * and wait for it to complete before returning.
   */
  void submit_current_cmd_and_wait(const bool final_use = false);

  /*
   * Submit one command buffer to the GPU.
   */
  void submit_cmd(vkapi::CommandBuffer& cmd_buf, VkFence fence);

  /*
   * Submits all the commands gathered in deferred_cmd_bufs_ to the GPU.
   */
  void submit_deferred_cmds_and_wait();

  /*
   * Ends and invalidates all deferred commands.
   */
  void clear_deferred_cmds();

 public:
  //
  // Graph Prepacking
  //

  inline void update_staging_nbytes_in_cmd(const size_t staging_bytes) {
    staging_nbytes_in_cmd_ += staging_bytes;
  }

  /*
   * Executes prepacking operations to transfer model weight data from the CPU
   * to GPU.
   */
  void prepack();

  //
  // Optional Graph Execution
  //

  void optional_warmup_execute();

  //
  // Graph Execution
  //

  void execute();

  //
  // Tensor View
  //

  void virtual_clone(const ValueRef dst, const ValueRef src);

  void virtual_transpose(
      const ValueRef tensor,
      const int64_t dim0,
      const int64_t dim1);

  //
  // Dynamic Shape support
  //

  void resize_input(const int64_t idx, const std::vector<int64_t>& new_sizes);

  void virtual_resize(
      const ValueRef idx,
      const std::vector<int64_t>& new_sizes);

  void propagate_resize();

  // Check if a specific ValueRef (or ValueList) was updated, with recursive
  // handling
  bool was_value_updated(const ValueRef idx) const noexcept;

  // Set the flag to indicate that re-encoding is required
  inline void set_requires_reencode() noexcept {
    requires_reencode_ = true;
  }

  //
  // Miscellaneous Utilities
  //

  inline bool int16_shader_types_enabled() const {
    return context_->adapter_ptr()->supports_int16_shader_types();
  }

  inline size_t execute_count() const {
    return execute_count_;
  }

  inline bool can_use_int8_dot_product() const {
    return can_use_int8_dot_product_;
  }

  inline void set_has_data_dependent_shapes() {
    config_.has_data_dependent_shapes = true;
  }

  inline bool has_data_dependent_shapes() const {
    return config_.has_data_dependent_shapes;
  }

  /*
   * Check whether the GPU supports 8 bit buffers.
   */
  inline bool int8_buffers_enabled() const {
    return context_->adapter_ptr()->has_full_int8_buffers_support();
  }

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
  friend class SymIntPtr;

  friend struct TmpTensor;
  friend struct SharedObject;
  friend class BlitNode;
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
