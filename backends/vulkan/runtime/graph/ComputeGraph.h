/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/GraphConfig.h>

#include <executorch/backends/vulkan/runtime/graph/containers/SharedObject.h>
#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/PrepackNode.h>

namespace at {
namespace native {
namespace vulkan {

// Define valid scalar types that the Value class can accept
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
   * Returns the value at a particular reference
   */
  inline Value& get_val(ValueRef idx) {
    return values_[idx];
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
  // Graph Building
  //

  ValueRef add_tensor(
      const std::vector<int64_t>& sizes,
      const api::ScalarType dtype = api::ScalarType::Float,
      const int64_t shared_object_idx = -1);
  ValueRef add_tensorref(
      const std::vector<int64_t>& sizes,
      const api::ScalarType dtype,
      const void* const data);
  ValueRef add_staging(const api::ScalarType dtype, const size_t numel);

  template <typename T>
  typename std::enable_if<is_valid_scalar_type<T>::value, ValueRef>::type
  add_scalar_list(std::vector<T>&& values);

  template <typename T>
  typename std::enable_if<is_valid_scalar_type<T>::value, ValueRef>::type
  add_scalar(T value);

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
ComputeGraph::add_scalar_list(std::vector<T>&& values) {
  ValueRef idx(static_cast<int>(values_.size()));
  values_.emplace_back(std::move(values));
  return idx;
}

template <typename T>
inline typename std::enable_if<is_valid_scalar_type<T>::value, ValueRef>::type
ComputeGraph::add_scalar(T value) {
  ValueRef idx(static_cast<int>(values_.size()));
  values_.emplace_back(value);
  return idx;
}

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
