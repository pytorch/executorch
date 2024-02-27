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

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Tensor.h>
#include <ATen/native/vulkan/api/Types.h>

#include <executorch/backends/vulkan/runtime/graph/Config.h>
#include <executorch/backends/vulkan/runtime/graph/Value.h>

namespace at {
namespace native {
namespace vulkan {

using ValueRef = int32_t;

struct IOValueRef {
  ValueRef value;
  ValueRef staging;
};

class ComputeGraph;

/*
 * Represents a single prepacking op in a ML model. In graph mode, ops will be
 * implemented in a derived class that implements encode, which will implement
 * encoding of shaders transferring necessary data (such as weights and biases)
 * to the GPU.
 */
class PrepackNode {
  friend class ComputeGraph;

 public:
  PrepackNode(ValueRef tref, ValueRef packed) : tref_{tref}, packed_{packed} {}

  virtual ~PrepackNode() = default;

 protected:
  ValueRef tref_;
  ValueRef packed_;

 public:
  virtual void encode(ComputeGraph* graph) const = 0;
};

/*
 * Represents a single execution op in a ML model. In graph mode, ops will be
 * implemented in a derived class that implements encode, which will implement
 * encoding of the shader corresponding to the op into the command buffer of a
 * ComputeGraph.
 */
class ExecuteNode {
  friend class ComputeGraph;

 public:
  ExecuteNode(ValueRef input, ValueRef output)
      : inputs_{input}, outputs_{output} {}
  ExecuteNode(
      const std::vector<ValueRef>& inputs,
      const std::vector<ValueRef>& outputs)
      : inputs_(inputs), outputs_(outputs) {}

  virtual ~ExecuteNode() = default;

 protected:
  std::vector<ValueRef> inputs_;
  std::vector<ValueRef> outputs_;

 public:
  virtual void encode(ComputeGraph* graph) const = 0;
};

struct SharedObject {
  friend class ComputeGraph;

  explicit SharedObject() = default;

  VkMemoryRequirements aggregate_memory_requirements;
  VmaAllocationCreateInfo aggregate_create_info;
  std::vector<ValueRef> users;
  api::MemoryAllocation allocation;

  void add_user(ComputeGraph* const graph, const ValueRef idx);
  void allocate(ComputeGraph* const graph);
  void bind_users(ComputeGraph* const graph);
};

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
  std::unique_ptr<api::Context> context_;
  std::vector<SharedObject> shared_objects_;
  std::vector<Value> values_;

  std::vector<std::unique_ptr<PrepackNode>> prepack_nodes_;
  std::vector<std::unique_ptr<ExecuteNode>> execute_nodes_;

  std::vector<ValueRef> inputs_;
  std::vector<ValueRef> outputs_;

 public:
  //
  // Accessors
  //

  inline api::Context* context() {
    return context_.get();
  }

  inline std::vector<ValueRef>& inputs() {
    return inputs_;
  }

  inline std::vector<ValueRef>& outputs() {
    return outputs_;
  }

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

  ValueRef set_input_tensor(const ValueRef idx, const bool use_staging = true);
  ValueRef set_output_tensor(const ValueRef idx, const bool use_staging = true);

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
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
