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
#include <executorch/runtime/core/named_data_map.h>

#include <executorch/backends/webgpu/runtime/WebGPUCompat.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>

#include <cstring>
#include <stdexcept>

namespace executorch::backends::webgpu {

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

WGPUBuffer WebGPUGraph::create_scratch_buffer(size_t nbytes) {
  WGPUBufferDescriptor buf_desc = {};
  buf_desc.size = nbytes > 0 ? nbytes : 4;
  buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
      WGPUBufferUsage_CopySrc;
  buf_desc.mappedAtCreation = false;
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device_, &buf_desc);
  scratch_buffers_.push_back(buffer);
  return buffer;
}

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
  for (auto& buf : scratch_buffers_) {
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
  for (auto& [_, shader] : shader_cache_) {
    if (shader) {
      wgpuShaderModuleRelease(shader);
    }
  }
  for (auto& [_, pipeline] : pipeline_cache_) {
    if (pipeline) {
      wgpuComputePipelineRelease(pipeline);
    }
  }
  for (auto& [_, bgl] : bgl_cache_) {
    if (bgl) {
      wgpuBindGroupLayoutRelease(bgl);
    }
  }
}

void WebGPUGraph::build(
    const void* flatbuffer_data,
    const uint8_t* constant_data,
    const executorch::runtime::NamedDataMap* named_data_map) {
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

        // Constants always get dedicated buffers regardless of mem_obj_id
        if (constant_id >= 0 || mem_obj_id < 0) {
          tensor_mem_obj_ids_[i] = -1;
          WGPUBufferDescriptor buf_desc = {};
          buf_desc.size = std::max(tensor.nbytes, size_t(4));
          buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
              WGPUBufferUsage_CopySrc;
          buf_desc.mappedAtCreation = false;
          tensor.buffer = wgpuDeviceCreateBuffer(device_, &buf_desc);

          if (constant_id >= 0 && constant_data && tensor.nbytes > 0) {
            const auto* constants = graph->constants();
            if (constants &&
                constant_id < static_cast<int>(constants->size())) {
              const auto* vk_bytes = constants->Get(constant_id);
              if (vk_bytes->offset() != UINT64_MAX) {
                const uint8_t* src = constant_data + vk_bytes->offset();
                wgpuQueueWriteBuffer(
                    queue_, tensor.buffer, 0, src, tensor.nbytes);
              } else if (
                  vk_bytes->named_key() != nullptr &&
                  named_data_map != nullptr) {
                // Constant stored in the PTE named-data map.
                auto buf =
                    named_data_map->get_data(vk_bytes->named_key()->c_str());
                if (!buf.ok()) {
                  throw std::runtime_error(
                      std::string("WebGPU: named constant '") +
                      vk_bytes->named_key()->c_str() +
                      "' not found in NamedDataMap");
                }
                if (buf->size() < tensor.nbytes) {
                  throw std::runtime_error(
                      std::string("WebGPU: named constant '") +
                      vk_bytes->named_key()->c_str() + "' undersized: have " +
                      std::to_string(buf->size()) + " bytes, need " +
                      std::to_string(tensor.nbytes));
                }
                wgpuQueueWriteBuffer(
                    queue_, tensor.buffer, 0, buf->data(), tensor.nbytes);
                buf->Free();
              } else {
                throw std::runtime_error(
                    "WebGPU: constant has no inline offset and no named-data key");
              }
            }
          }
        } else {
          // Shared buffer: track required size, defer allocation to pass 2
          tensor_mem_obj_ids_[i] = mem_obj_id;
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
    buf_desc.size = std::max(shared_buffer_sizes_[id], size_t(4));
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
      staging_desc.size = std::max(tensors_[oid].nbytes, size_t(4));
      staging_desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
      staging_desc.mappedAtCreation = false;
      output_staging_buffers_.push_back(
          wgpuDeviceCreateBuffer(device_, &staging_desc));
    }
  }

  for (size_t i = 0; i < output_ids_.size(); i++) {
    int oid = output_ids_[i];
    output_copies_.push_back(
        {tensors_[oid].buffer,
         output_staging_buffers_[i],
         tensors_[oid].nbytes});
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

WGPUShaderModule WebGPUGraph::get_or_create_shader(
    const std::string& key,
    const char* wgsl_source) {
  auto it = shader_cache_.find(key);
  if (it != shader_cache_.end()) {
    return it->second;
  }

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {wgsl_source, WGPU_STRLEN};

  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device_, &shader_desc);

  shader_cache_[key] = shader;
  return shader;
}

WGPUComputePipeline WebGPUGraph::get_or_create_pipeline(
    const std::string& key,
    WGPUShaderModule shader,
    WGPUPipelineLayout layout) {
  auto it = pipeline_cache_.find(key);
  if (it != pipeline_cache_.end()) {
    return it->second;
  }

  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device_, &pipeline_desc);

  pipeline_cache_[key] = pipeline;
  return pipeline;
}

WGPUBindGroupLayout WebGPUGraph::get_or_create_bgl(
    const std::string& key,
    const WGPUBindGroupLayoutEntry* entries,
    uint32_t count) {
  auto it = bgl_cache_.find(key);
  if (it != bgl_cache_.end()) {
    return it->second;
  }

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = count;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device_, &bgl_desc);

  bgl_cache_[key] = bgl;
  return bgl;
}

void WebGPUGraph::copy_inputs(
    const std::vector<std::pair<const void*, size_t>>& inputs) {
  for (size_t i = 0; i < inputs.size() && i < input_ids_.size(); i++) {
    if (inputs[i].second == 0) {
      continue;
    }
    int tid = input_ids_[i];
    const auto& tensor = tensors_[tid];
    wgpuQueueWriteBuffer(
        queue_, tensor.buffer, 0, inputs[i].first, inputs[i].second);
  }
}

void WebGPUGraph::execute() {
  const size_t n = dispatches_.size();
  const size_t chunk = execute_config_.chunk_size;

  if (chunk == 0 || n <= chunk) {
    WGPUCommandEncoderDescriptor enc_desc = {};
    WGPUCommandEncoder encoder =
        wgpuDeviceCreateCommandEncoder(device_, &enc_desc);

    // One pass per dispatch: enforces storage RAW ordering across deps.
    for (const auto& dispatch : dispatches_) {
      WGPUComputePassDescriptor pass_desc = {};
      WGPUComputePassEncoder pass =
          wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
      wgpuComputePassEncoderSetPipeline(pass, dispatch.pipeline);
      wgpuComputePassEncoderSetBindGroup(
          pass, 0, dispatch.bind_group, 0, nullptr);
      wgpuComputePassEncoderDispatchWorkgroups(
          pass, dispatch.workgroup_count_x, 1, 1);
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass);
    }

    for (const auto& copy : output_copies_) {
      wgpuCommandEncoderCopyBufferToBuffer(
          encoder, copy.src_buffer, 0, copy.staging_buffer, 0, copy.nbytes);
    }

    WGPUCommandBufferDescriptor cmd_desc = {};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(queue_, 1, &cmd);

    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
    return;
  }

  const size_t first_chunk = execute_config_.initial_chunk_size > 0
      ? execute_config_.initial_chunk_size
      : chunk;

  size_t start = 0;
  size_t current_chunk = first_chunk;

  while (start < n) {
    size_t end = std::min(start + current_chunk, n);

    WGPUCommandEncoderDescriptor enc_desc = {};
    WGPUCommandEncoder encoder =
        wgpuDeviceCreateCommandEncoder(device_, &enc_desc);

    for (size_t i = start; i < end; i++) {
      WGPUComputePassDescriptor pass_desc = {};
      WGPUComputePassEncoder pass =
          wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
      wgpuComputePassEncoderSetPipeline(pass, dispatches_[i].pipeline);
      wgpuComputePassEncoderSetBindGroup(
          pass, 0, dispatches_[i].bind_group, 0, nullptr);
      wgpuComputePassEncoderDispatchWorkgroups(
          pass, dispatches_[i].workgroup_count_x, 1, 1);
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass);
    }

    if (end == n) {
      for (const auto& copy : output_copies_) {
        wgpuCommandEncoderCopyBufferToBuffer(
            encoder, copy.src_buffer, 0, copy.staging_buffer, 0, copy.nbytes);
      }
    }

    WGPUCommandBufferDescriptor cmd_desc = {};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(queue_, 1, &cmd);

    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

    start = end;
    current_chunk = chunk;
  }
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
  const size_t count = std::min(outputs.size(), output_staging_buffers_.size());

  std::vector<MapCallbackData> cb_data(count);

  for (size_t i = 0; i < count; i++) {
    if (outputs[i].second == 0) {
      cb_data[i].done = true;
      cb_data[i].status = WGPUMapAsyncStatus_Success;
      continue;
    }
    WGPUBufferMapCallbackInfo cb_info = {};
    cb_info.mode = WGPUCallbackMode_AllowSpontaneous;
    cb_info.callback = buffer_map_callback;
    cb_info.userdata1 = &cb_data[i];
    wgpuBufferMapAsync(
        output_staging_buffers_[i],
        WGPUMapMode_Read,
        0,
        outputs[i].second,
        cb_info);
  }

  // WaitAny returns pre-queue on CI Dawn+SwiftShader here; pump until mapped.
  bool all_mapped = false;
  while (!all_mapped) {
    webgpu_poll(instance_);
    all_mapped = true;
    for (size_t i = 0; i < count; i++) {
      if (outputs[i].second != 0 && !cb_data[i].done) {
        all_mapped = false;
        break;
      }
    }
  }

  for (size_t i = 0; i < count; i++) {
    if (outputs[i].second == 0) {
      continue;
    }
    if (cb_data[i].status == WGPUMapAsyncStatus_Success) {
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
  stats.num_cached_pipelines = static_cast<int>(pipeline_cache_.size());
  stats.num_cached_shaders = static_cast<int>(shader_cache_.size());
  return stats;
}

} // namespace executorch::backends::webgpu
