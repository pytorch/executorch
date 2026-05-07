/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <webgpu/webgpu.h>

#include <cstdint>
#include <string>
#include <vector>

namespace executorch {
namespace backends {
namespace webgpu {

struct WebGPUTensor {
  WGPUBuffer buffer = nullptr;
  std::vector<int64_t> dims;
  size_t nbytes = 0;
};

struct WebGPUDispatch {
  WGPUComputePipeline pipeline = nullptr;
  WGPUBindGroup bind_group = nullptr;
  uint32_t workgroup_count_x = 1;
};

struct WebGPUMemoryStats {
  size_t tensor_buffer_bytes = 0;
  size_t shared_buffer_bytes = 0;
  int num_shared_objects = 0;
  size_t unshared_tensor_buffer_bytes = 0;
  size_t staging_buffer_bytes = 0;
  size_t uniform_buffer_bytes = 0;
  int num_tensors = 0;
  int num_dispatches = 0;

  size_t total_bytes() const {
    return tensor_buffer_bytes + staging_buffer_bytes + uniform_buffer_bytes;
  }
};

class WebGPUGraph {
 public:
  WebGPUGraph();
  ~WebGPUGraph();

  // Build the graph from a deserialized VkGraph flatbuffer and constant data.
  // The flatbuffer_data pointer must remain valid during build().
  void build(const void* flatbuffer_data, const uint8_t* constant_data);

  // Copy input tensor data from host pointers into GPU buffers.
  void copy_inputs(const std::vector<std::pair<const void*, size_t>>& inputs);

  // Execute all recorded dispatches.
  void execute();

  // Copy output tensor data from GPU buffers back to host pointers.
  // Uses mapAsync + ASYNCIFY in Wasm.
  void copy_outputs(std::vector<std::pair<void*, size_t>>& outputs);

  const std::vector<int>& input_ids() const {
    return input_ids_;
  }
  const std::vector<int>& output_ids() const {
    return output_ids_;
  }

  // Access tensors by value ID (used by op implementations).
  WebGPUTensor& get_tensor(int id) {
    return tensors_[id];
  }
  const WebGPUTensor& get_tensor(int id) const {
    return tensors_[id];
  }

  // Access scalar values stored during graph build.
  double get_double(int id) const {
    return doubles_[id];
  }
  int64_t get_int(int id) const {
    return ints_[id];
  }

  WGPUDevice device() const {
    return device_;
  }
  WGPUQueue queue() const {
    return queue_;
  }

  void add_dispatch(WebGPUDispatch dispatch) {
    dispatches_.push_back(dispatch);
  }

  void add_uniform_buffer_bytes(size_t bytes) {
    uniform_buffer_bytes_ += bytes;
  }

  void set_instance(WGPUInstance instance) {
    instance_ = instance;
  }
  void set_device(WGPUDevice device) {
    device_ = device;
  }

  WebGPUMemoryStats memory_stats() const;

  int num_values() const {
    return static_cast<int>(value_types_.size());
  }

  enum class ValueType { Tensor, Int, Double, Bool, Null, String };

  ValueType get_value_type(int id) const {
    return value_types_[id];
  }

 private:
  WGPUInstance instance_ = nullptr;
  WGPUDevice device_ = nullptr;
  WGPUQueue queue_ = nullptr;

  // Flat arrays indexed by value ID. Only the relevant one is populated
  // per ID based on value_types_.
  std::vector<ValueType> value_types_;
  std::vector<WebGPUTensor> tensors_;
  std::vector<int64_t> ints_;
  std::vector<double> doubles_;
  std::vector<bool> bools_;

  std::vector<int> input_ids_;
  std::vector<int> output_ids_;

  // Memory aliasing: tensors with the same mem_obj_id share a WGPUBuffer.
  std::vector<int> tensor_mem_obj_ids_;
  std::vector<WGPUBuffer> shared_buffers_;
  std::vector<size_t> shared_buffer_sizes_;

  // Staging buffers for reading back outputs (MapRead | CopyDst).
  std::vector<WGPUBuffer> output_staging_buffers_;

  std::vector<WebGPUDispatch> dispatches_;

  size_t uniform_buffer_bytes_ = 0;
};

} // namespace webgpu
} // namespace backends
} // namespace executorch
