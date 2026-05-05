/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/serialization/schema_generated.h>

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <webgpu/wgpu.h>

#include <executorch/runtime/platform/assert.h>

#include <cstring>
#include <stdexcept>

namespace executorch {
namespace backends {
namespace webgpu {

// vkgraph namespace is declared at global scope in the generated FlatBuffer
// header

namespace {

size_t vk_datatype_size(vkgraph::VkDataType dtype) {
  switch (dtype) {
    case vkgraph::VkDataType::BOOL:
    case vkgraph::VkDataType::UINT8:
    case vkgraph::VkDataType::INT8:
      return 1;
    case vkgraph::VkDataType::FLOAT16:
      return 2;
    case vkgraph::VkDataType::INT32:
    case vkgraph::VkDataType::FLOAT32:
      return 4;
    case vkgraph::VkDataType::INT64:
    case vkgraph::VkDataType::FLOAT64:
      return 8;
    default:
      return 0;
  }
}

} // namespace

WebGPUGraph::WebGPUGraph() = default;

WebGPUGraph::~WebGPUGraph() {
  for (size_t i = 0; i < tensors_.size(); i++) {
    if (tensors_[i].buffer &&
        (i >= tensor_mem_obj_ids_.size() || tensor_mem_obj_ids_[i] < 0)) {
      wgpuBufferRelease(tensors_[i].buffer);
    }
  }
  for (auto& buf : shared_buffers_) {
    if (buf) {
      wgpuBufferRelease(buf);
    }
  }
  for (auto& buf : output_staging_buffers_) {
    if (buf) {
      wgpuBufferRelease(buf);
    }
  }
  for (auto& d : dispatches_) {
    if (d.pipeline) {
      wgpuComputePipelineRelease(d.pipeline);
    }
    if (d.bind_group) {
      wgpuBindGroupRelease(d.bind_group);
    }
  }
}

void WebGPUGraph::build(
    const void* flatbuffer_data,
    const uint8_t* constant_data) {
  if (!device_) {
    auto* ctx = get_default_webgpu_context();
    if (ctx) {
      device_ = ctx->device;
      instance_ = ctx->instance;
    }
  }
  if (!device_) {
    throw std::runtime_error(
        "WebGPU device not available. "
        "Call set_default_webgpu_context() before loading.");
  }
  queue_ = wgpuDeviceGetQueue(device_);

  const auto* graph = vkgraph::GetVkGraph(flatbuffer_data);

  // Phase 1: Create all values
  const auto* values = graph->values();
  const int num_vals = values ? values->size() : 0;
  value_types_.resize(num_vals, ValueType::Null);
  tensors_.resize(num_vals);
  tensor_mem_obj_ids_.resize(num_vals, -1);
  ints_.resize(num_vals, 0);
  doubles_.resize(num_vals, 0.0);
  bools_.resize(num_vals, false);

  for (int i = 0; i < num_vals; i++) {
    const auto* val = values->Get(i);
    if (!val || val->value_type() == vkgraph::GraphTypes::NONE) {
      value_types_[i] = ValueType::Null;
      continue;
    }

    switch (val->value_type()) {
      case vkgraph::GraphTypes::VkTensor: {
        value_types_[i] = ValueType::Tensor;
        const auto* vk_tensor = val->value_as_VkTensor();
        auto& tensor = tensors_[i];

        const auto* dims = vk_tensor->dims();
        size_t numel = 1;
        if (dims) {
          for (unsigned j = 0; j < dims->size(); j++) {
            tensor.dims.push_back(static_cast<int64_t>(dims->Get(j)));
            numel *= dims->Get(j);
          }
        }
        tensor.nbytes = numel * vk_datatype_size(vk_tensor->datatype());

        int constant_id = vk_tensor->constant_id();
        int mem_obj_id = vk_tensor->mem_obj_id();
        tensor_mem_obj_ids_[i] = mem_obj_id;

        if (constant_id >= 0 || mem_obj_id < 0) {
          // Dedicated buffer: constants or tensors that don't share memory
          WGPUBufferDescriptor buf_desc = {};
          ET_CHECK_MSG(tensor.nbytes > 0, "Tensor has zero bytes");
          buf_desc.size = tensor.nbytes;
          buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
              WGPUBufferUsage_CopySrc;
          buf_desc.mappedAtCreation = false;
          tensor.buffer = wgpuDeviceCreateBuffer(device_, &buf_desc);

          if (constant_id >= 0 && constant_data) {
            const auto* constants = graph->constants();
            if (constants &&
                constant_id < static_cast<int>(constants->size())) {
              const auto* vk_bytes = constants->Get(constant_id);
              if (vk_bytes->offset() != UINT64_MAX) {
                const uint8_t* src = constant_data + vk_bytes->offset();
                wgpuQueueWriteBuffer(
                    queue_, tensor.buffer, 0, src, tensor.nbytes);
              }
            }
          }
        } else {
          // Shared buffer: track required size, defer allocation to pass 2
          size_t id = static_cast<size_t>(mem_obj_id);
          if (id >= shared_buffer_sizes_.size()) {
            shared_buffer_sizes_.resize(id + 1, 0);
          }
          shared_buffer_sizes_[id] =
              std::max(shared_buffer_sizes_[id], tensor.nbytes);
        }
        break;
      }
      case vkgraph::GraphTypes::Int: {
        value_types_[i] = ValueType::Int;
        ints_[i] = val->value_as_Int()->int_val();
        break;
      }
      case vkgraph::GraphTypes::Double: {
        value_types_[i] = ValueType::Double;
        doubles_[i] = val->value_as_Double()->double_val();
        break;
      }
      case vkgraph::GraphTypes::Bool: {
        value_types_[i] = ValueType::Bool;
        bools_[i] = val->value_as_Bool()->bool_val();
        break;
      }
      default:
        value_types_[i] = ValueType::Null;
        break;
    }
  }

  // Allocate shared buffers and assign to tensors
  shared_buffers_.resize(shared_buffer_sizes_.size(), nullptr);
  for (size_t id = 0; id < shared_buffer_sizes_.size(); id++) {
    WGPUBufferDescriptor buf_desc = {};
    ET_CHECK_MSG(shared_buffer_sizes_[id] > 0, "Shared buffer has zero bytes");
    buf_desc.size = shared_buffer_sizes_[id];
    buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
        WGPUBufferUsage_CopySrc;
    buf_desc.mappedAtCreation = false;
    shared_buffers_[id] = wgpuDeviceCreateBuffer(device_, &buf_desc);
  }
  for (int i = 0; i < num_vals; i++) {
    int mid = tensor_mem_obj_ids_[i];
    if (mid >= 0) {
      tensors_[i].buffer = shared_buffers_[mid];
    }
  }

  // Phase 2: Record input and output IDs
  const auto* fb_input_ids = graph->input_ids();
  if (fb_input_ids) {
    for (unsigned i = 0; i < fb_input_ids->size(); i++) {
      input_ids_.push_back(static_cast<int>(fb_input_ids->Get(i)));
    }
  }
  const auto* fb_output_ids = graph->output_ids();
  if (fb_output_ids) {
    for (unsigned i = 0; i < fb_output_ids->size(); i++) {
      int oid = static_cast<int>(fb_output_ids->Get(i));
      output_ids_.push_back(oid);

      // Create staging buffer for output readback
      WGPUBufferDescriptor staging_desc = {};
      ET_CHECK_MSG(tensors_[oid].nbytes > 0, "Output tensor has zero bytes");
      staging_desc.size = tensors_[oid].nbytes;
      staging_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
      staging_desc.mappedAtCreation = false;
      output_staging_buffers_.push_back(
          wgpuDeviceCreateBuffer(device_, &staging_desc));
    }
  }

  // Phase 3: Build operator dispatch chain
  const auto* chain = graph->chain();
  if (chain) {
    for (unsigned i = 0; i < chain->size(); i++) {
      const auto* op_call = chain->Get(i);
      std::string op_name = op_call->name()->str();

      if (!webgpu_operator_registry().has_op(op_name)) {
        throw std::runtime_error("WebGPU backend: unsupported op: " + op_name);
      }

      const auto* fb_args = op_call->args();
      std::vector<int> args;
      if (fb_args) {
        for (unsigned j = 0; j < fb_args->size(); j++) {
          args.push_back(static_cast<int>(fb_args->Get(j)));
        }
      }

      webgpu_operator_registry().get_op_fn(op_name)(*this, args);
    }
  }
}

void WebGPUGraph::copy_inputs(
    const std::vector<std::pair<const void*, size_t>>& inputs) {
  for (size_t i = 0; i < inputs.size() && i < input_ids_.size(); i++) {
    int tid = input_ids_[i];
    const auto& tensor = tensors_[tid];
    wgpuQueueWriteBuffer(
        queue_, tensor.buffer, 0, inputs[i].first, inputs[i].second);
  }
}

void WebGPUGraph::execute() {
  WGPUCommandEncoderDescriptor enc_desc = {};
  WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(device_, &enc_desc);

  WGPUComputePassDescriptor pass_desc = {};
  WGPUComputePassEncoder pass =
      wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);

  for (const auto& dispatch : dispatches_) {
    wgpuComputePassEncoderSetPipeline(pass, dispatch.pipeline);
    wgpuComputePassEncoderSetBindGroup(
        pass, 0, dispatch.bind_group, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(
        pass, dispatch.workgroup_count_x, 1, 1);
  }

  wgpuComputePassEncoderEnd(pass);
  wgpuComputePassEncoderRelease(pass);

  // Copy outputs to staging buffers
  for (size_t i = 0; i < output_ids_.size(); i++) {
    int oid = output_ids_[i];
    wgpuCommandEncoderCopyBufferToBuffer(
        encoder,
        tensors_[oid].buffer,
        0,
        output_staging_buffers_[i],
        0,
        tensors_[oid].nbytes);
  }

  WGPUCommandBufferDescriptor cmd_desc = {};
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
  wgpuQueueSubmit(queue_, 1, &cmd);

  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(encoder);
}

namespace {

struct MapCallbackData {
  bool done = false;
  WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
};

void buffer_map_callback(
    WGPUMapAsyncStatus status,
    WGPUStringView /*message*/,
    void* userdata1,
    void* /*userdata2*/) {
  auto* data = static_cast<MapCallbackData*>(userdata1);
  data->status = status;
  data->done = true;
}

} // namespace

void WebGPUGraph::copy_outputs(std::vector<std::pair<void*, size_t>>& outputs) {
  for (size_t i = 0; i < outputs.size() && i < output_staging_buffers_.size();
       i++) {
    MapCallbackData cb_data;
    WGPUBufferMapCallbackInfo cb_info = {};
    cb_info.mode = WGPUCallbackMode_AllowSpontaneous;
    cb_info.callback = buffer_map_callback;
    cb_info.userdata1 = &cb_data;
    wgpuBufferMapAsync(
        output_staging_buffers_[i],
        WGPUMapMode_Read,
        0,
        outputs[i].second,
        cb_info);

    // Poll until the map callback fires.
    wgpuDevicePoll(device_, true, nullptr);

    if (cb_data.status == WGPUMapAsyncStatus_Success) {
      const void* mapped = wgpuBufferGetConstMappedRange(
          output_staging_buffers_[i], 0, outputs[i].second);
      std::memcpy(outputs[i].first, mapped, outputs[i].second);
      wgpuBufferUnmap(output_staging_buffers_[i]);
    } else {
      throw std::runtime_error("WebGPU buffer map failed for output");
    }
  }
}

WebGPUMemoryStats WebGPUGraph::memory_stats() const {
  WebGPUMemoryStats stats;
  for (size_t i = 0; i < value_types_.size(); i++) {
    if (value_types_[i] == ValueType::Tensor && tensors_[i].nbytes > 0) {
      stats.num_tensors++;
      // Shared tensors are tracked via shared_buffer_sizes_
      bool is_shared =
          i < tensor_mem_obj_ids_.size() && tensor_mem_obj_ids_[i] >= 0;
      if (!is_shared) {
        stats.unshared_tensor_buffer_bytes += tensors_[i].nbytes;
      }
    }
  }
  for (size_t s : shared_buffer_sizes_) {
    stats.shared_buffer_bytes += s;
  }
  stats.num_shared_objects = static_cast<int>(shared_buffers_.size());
  stats.tensor_buffer_bytes =
      stats.shared_buffer_bytes + stats.unshared_tensor_buffer_bytes;
  for (size_t i = 0; i < output_ids_.size(); i++) {
    stats.staging_buffer_bytes += tensors_[output_ids_[i]].nbytes;
  }
  stats.uniform_buffer_bytes = uniform_buffer_bytes_;
  stats.num_dispatches = static_cast<int>(dispatches_.size());
  return stats;
}

} // namespace webgpu
} // namespace backends
} // namespace executorch
