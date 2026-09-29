/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUShaderRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/serialization/schema_generated.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/portable_type/half.h>

#include <executorch/backends/webgpu/runtime/WebGPUCompat.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/passes/QkvBk64.h>
#include <executorch/backends/webgpu/runtime/passes/SwiGLU.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

namespace executorch::backends::webgpu {

// vkgraph namespace is declared at global scope in the generated FlatBuffer
// header

namespace {

const uint8_t* checked_inline_constant(
    const uint8_t* data,
    size_t data_size,
    uint64_t offset,
    size_t required_size,
    const char* error_message) {
  if (data == nullptr || offset > data_size ||
      required_size > data_size - static_cast<size_t>(offset)) {
    throw std::runtime_error(error_message);
  }
  return data + static_cast<size_t>(offset);
}

class ScopedBindGroupLayout final {
 public:
  explicit ScopedBindGroupLayout(WGPUBindGroupLayout handle)
      : handle_(handle) {}
  ~ScopedBindGroupLayout() {
    if (handle_ != nullptr) {
      wgpuBindGroupLayoutRelease(handle_);
    }
  }
  ScopedBindGroupLayout(const ScopedBindGroupLayout&) = delete;
  ScopedBindGroupLayout& operator=(const ScopedBindGroupLayout&) = delete;

  WGPUBindGroupLayout get() const {
    return handle_;
  }

 private:
  WGPUBindGroupLayout handle_;
};

class ScopedBindGroup final {
 public:
  explicit ScopedBindGroup(WGPUBindGroup handle) : handle_(handle) {}
  ~ScopedBindGroup() {
    if (handle_ != nullptr) {
      wgpuBindGroupRelease(handle_);
    }
  }
  ScopedBindGroup(const ScopedBindGroup&) = delete;
  ScopedBindGroup& operator=(const ScopedBindGroup&) = delete;

  WGPUBindGroup get() const {
    return handle_;
  }

  WGPUBindGroup release() {
    WGPUBindGroup handle = handle_;
    handle_ = nullptr;
    return handle;
  }

 private:
  WGPUBindGroup handle_;
};

class ScopedComputePipeline final {
 public:
  explicit ScopedComputePipeline(WGPUComputePipeline handle)
      : handle_(handle) {}
  ~ScopedComputePipeline() {
    if (handle_ != nullptr) {
      wgpuComputePipelineRelease(handle_);
    }
  }
  ScopedComputePipeline(const ScopedComputePipeline&) = delete;
  ScopedComputePipeline& operator=(const ScopedComputePipeline&) = delete;

  WGPUComputePipeline release() {
    WGPUComputePipeline handle = handle_;
    handle_ = nullptr;
    return handle;
  }

 private:
  WGPUComputePipeline handle_;
};

class ScopedComputePipelineRef final {
 public:
  explicit ScopedComputePipelineRef(WGPUComputePipeline handle)
      : handle_(handle) {
    wgpuComputePipelineAddRef(handle_);
  }
  ~ScopedComputePipelineRef() {
    if (handle_ != nullptr) {
      wgpuComputePipelineRelease(handle_);
    }
  }
  ScopedComputePipelineRef(const ScopedComputePipelineRef&) = delete;
  ScopedComputePipelineRef& operator=(const ScopedComputePipelineRef&) = delete;

  WGPUComputePipeline release() {
    WGPUComputePipeline handle = handle_;
    handle_ = nullptr;
    return handle;
  }

 private:
  WGPUComputePipeline handle_;
};

void append_key_component(std::string& key, const std::string& value) {
  const uint64_t size = value.size();
  key.append(reinterpret_cast<const char*>(&size), sizeof(size));
  key.append(value);
}

std::vector<WebGPUSpecializationConstant> canonical_constants(
    const WebGPUComputeDispatchDescriptor& descriptor) {
  std::vector<WebGPUSpecializationConstant> constants = descriptor.constants;
  std::sort(
      constants.begin(),
      constants.end(),
      [](const auto& left, const auto& right) {
        return left.name < right.name;
      });
  for (size_t i = 0; i < constants.size(); ++i) {
    if (constants[i].name.empty()) {
      throw std::runtime_error(
          "WebGPU compute dispatch: empty specialization constant name");
    }
    if (!std::isfinite(constants[i].value)) {
      throw std::runtime_error(
          "WebGPU compute dispatch: non-finite specialization constant");
    }
    if (i > 0 && constants[i - 1].name == constants[i].name) {
      throw std::runtime_error(
          "WebGPU compute dispatch: duplicate specialization constant");
    }
    if (constants[i].value == 0.0) {
      constants[i].value = 0.0;
    }
  }
  return constants;
}

// Op name the AOT exporter emits for a prepacked constant (must match the
// serialized schema); compared in the prepack pre-scan below.
constexpr const char* kPrepackOpName = "et_vk.prepack.default";
constexpr const char* kQ4gswLinearOpName = "et_vk.linear_q4gsw.default";
constexpr size_t kQ4gswOutputArg = 5;

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

bool vk_datatype_is_int(vkgraph::VkDataType dtype) {
  switch (dtype) {
    case vkgraph::VkDataType::BOOL:
    case vkgraph::VkDataType::UINT8:
    case vkgraph::VkDataType::INT8:
    case vkgraph::VkDataType::INT32:
    case vkgraph::VkDataType::INT64:
      return true;
    default:
      return false;
  }
}

size_t storage_buffer_size(size_t nbytes) {
  const size_t at_least_four = std::max(nbytes, size_t(4));
  if (at_least_four > std::numeric_limits<size_t>::max() - 3u) {
    throw std::runtime_error("WebGPU: storage buffer size overflows alignment");
  }
  return (at_least_four + 3u) & ~size_t(3);
}

void write_storage_buffer(
    WGPUQueue queue,
    WGPUBuffer buffer,
    const void* data,
    size_t nbytes) {
  if (nbytes == 0u) {
    return;
  }
  if (nbytes % 4u == 0u) {
    wgpuQueueWriteBuffer(queue, buffer, 0, data, nbytes);
    return;
  }
  std::vector<uint8_t> padded(storage_buffer_size(nbytes), 0u);
  std::memcpy(padded.data(), data, nbytes);
  wgpuQueueWriteBuffer(queue, buffer, 0, padded.data(), padded.size());
}

// Normalize a possibly-negative dim against rank; throws (fail-loud) if OOR.
int normalize_dim(int dim, int rank, const char* op) {
  if (dim < 0) {
    dim += rank;
  }
  if (dim < 0 || dim >= rank) {
    throw std::runtime_error(
        std::string("WebGPU ") + op + ": dim out of range");
  }
  return dim;
}

} // namespace

std::string make_compute_pipeline_key(
    const WebGPUComputeDispatchDescriptor& descriptor) {
  if (descriptor.shader_name.empty()) {
    throw std::runtime_error("WebGPU compute dispatch: empty shader name");
  }
  if (descriptor.entry_point.empty()) {
    throw std::runtime_error("WebGPU compute dispatch: empty entry point");
  }

  std::string key;
  append_key_component(key, descriptor.shader_name);
  append_key_component(key, descriptor.entry_point);
  for (const auto& constant : canonical_constants(descriptor)) {
    append_key_component(key, constant.name);
    uint64_t value_bits = 0;
    static_assert(sizeof(value_bits) == sizeof(constant.value));
    std::memcpy(&value_bits, &constant.value, sizeof(value_bits));
    key.append(reinterpret_cast<const char*>(&value_bits), sizeof(value_bits));
  }
  return key;
}

void validate_compute_dispatch_descriptor(
    const WebGPUComputeDispatchDescriptor& descriptor) {
  (void)make_compute_pipeline_key(descriptor);
  if (descriptor.bindings.empty()) {
    throw std::runtime_error("WebGPU compute dispatch: no buffer bindings");
  }
  for (const auto& binding : descriptor.bindings) {
    if (binding.buffer == nullptr) {
      throw std::runtime_error("WebGPU compute dispatch: null buffer binding");
    }
    if (binding.size == 0) {
      throw std::runtime_error("WebGPU compute dispatch: zero-size binding");
    }
    if (binding.offset > UINT64_MAX - binding.size) {
      throw std::runtime_error(
          "WebGPU compute dispatch: binding range overflow");
    }
    if (binding.offset + binding.size > wgpuBufferGetSize(binding.buffer)) {
      throw std::runtime_error(
          "WebGPU compute dispatch: binding range exceeds buffer");
    }
  }
}

WebGPUGraph::WebGPUGraph() = default;

WGPUBuffer WebGPUGraph::create_scratch_buffer(size_t nbytes) {
  WGPUBufferDescriptor buf_desc = {};
  buf_desc.size = storage_buffer_size(nbytes);
  buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
      WGPUBufferUsage_CopySrc;
  buf_desc.mappedAtCreation = false;
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device_, &buf_desc);
  scratch_buffers_.push_back(buffer);
  return buffer;
}

WGPUBuffer WebGPUGraph::acquire_scratch(size_t nbytes) {
  nbytes = storage_buffer_size(nbytes);
  // Best-fit reuse: smallest free slot with size in [nbytes, 2*nbytes] -- the
  // 2x cap stops a large Cmax-sized buffer from backing a tiny request. Never
  // reuse an in_use slot (co-live safety).
  ScratchSlot* best = nullptr;
  for (auto& s : scratch_pool_) {
    // s.size - nbytes (safe: s.size >= nbytes) avoids overflowing 2 * nbytes.
    if (!s.in_use && s.size >= nbytes && s.size - nbytes <= nbytes) {
      if (best == nullptr || s.size < best->size) {
        best = &s;
      }
    }
  }
  if (best != nullptr) {
    best->in_use = true;
    return best->buffer;
  }
  // None reusable -> create a new slot (freed in the dtor, like
  // scratch_buffers_).
  WGPUBufferDescriptor buf_desc = {};
  buf_desc.size = nbytes;
  buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
      WGPUBufferUsage_CopySrc;
  buf_desc.mappedAtCreation = false;
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device_, &buf_desc);
  scratch_pool_.push_back({buffer, nbytes, true});
  return buffer;
}

void WebGPUGraph::release_scratch(WGPUBuffer buffer) {
  if (!buffer) {
    return;
  }
  for (auto& s : scratch_pool_) {
    if (s.buffer == buffer) {
      s.in_use = false;
      return;
    }
  }
  // Not a pooled buffer -> no-op; the dtor frees it via scratch_buffers_.
}

WGPUBuffer WebGPUGraph::make_uniform_buffer(const void* data, size_t size) {
  if (data == nullptr || size == 0u) {
    throw std::runtime_error("WebGPU: invalid uniform buffer data");
  }
  WGPUBufferDescriptor desc = {};
  desc.size = size;
  desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  desc.mappedAtCreation = true;
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device_, &desc);
  if (buffer == nullptr) {
    throw std::runtime_error("WebGPU: failed to create uniform buffer");
  }
  void* mapped = wgpuBufferGetMappedRange(buffer, 0, size);
  if (mapped == nullptr) {
    wgpuBufferRelease(buffer);
    throw std::runtime_error("WebGPU: failed to map uniform buffer");
  }
  std::memcpy(mapped, data, size);
  wgpuBufferUnmap(buffer);
  uniform_buffer_bytes_ += size;
  return buffer;
}

size_t WebGPUGraph::add_compute_dispatch(
    const WebGPUComputeDispatchDescriptor& descriptor) {
  validate_compute_dispatch_descriptor(descriptor);
  const WebGPUShaderInfo& shader_info =
      get_webgpu_shader_info(descriptor.shader_name);
  WGPUShaderModule shader =
      get_or_create_shader(descriptor.shader_name, shader_info.source);

  const std::string pipeline_key = make_compute_pipeline_key(descriptor);
  WGPUComputePipeline pipeline = nullptr;
  auto pipeline_it = pipeline_cache_.find(pipeline_key);
  if (pipeline_it != pipeline_cache_.end()) {
    pipeline = pipeline_it->second;
  } else {
    const auto constants = canonical_constants(descriptor);
    std::vector<WGPUConstantEntry> entries(constants.size());
    for (size_t i = 0; i < constants.size(); ++i) {
      entries[i].key = {constants[i].name.data(), constants[i].name.size()};
      entries[i].value = constants[i].value;
    }

    WGPUComputePipelineDescriptor pipeline_desc = {};
    pipeline_desc.layout = nullptr;
    pipeline_desc.compute.module = shader;
    pipeline_desc.compute.entryPoint = {
        descriptor.entry_point.data(), descriptor.entry_point.size()};
    pipeline_desc.compute.constantCount = entries.size();
    pipeline_desc.compute.constants = entries.data();
    ScopedComputePipeline created_pipeline(
        wgpuDeviceCreateComputePipeline(device_, &pipeline_desc));
    pipeline = created_pipeline.release();
    if (pipeline == nullptr) {
      throw std::runtime_error("WebGPU: failed to create compute pipeline");
    }
    ScopedComputePipeline pipeline_owner(pipeline);
    pipeline_cache_.emplace(pipeline_key, pipeline);
    pipeline_owner.release();
  }

  ScopedBindGroupLayout layout(
      wgpuComputePipelineGetBindGroupLayout(pipeline, 0));
  if (layout.get() == nullptr) {
    throw std::runtime_error("WebGPU: failed to get bind-group layout");
  }

  std::vector<WGPUBindGroupEntry> entries(descriptor.bindings.size());
  for (size_t i = 0; i < descriptor.bindings.size(); ++i) {
    entries[i].binding = i;
    entries[i].buffer = descriptor.bindings[i].buffer;
    entries[i].offset = descriptor.bindings[i].offset;
    entries[i].size = descriptor.bindings[i].size;
  }
  WGPUBindGroupDescriptor bind_group_desc = {};
  bind_group_desc.layout = layout.get();
  bind_group_desc.entryCount = entries.size();
  bind_group_desc.entries = entries.data();
  ScopedBindGroup bind_group(
      wgpuDeviceCreateBindGroup(device_, &bind_group_desc));
  if (bind_group.get() == nullptr) {
    throw std::runtime_error("WebGPU: failed to create bind group");
  }

  ScopedComputePipelineRef dispatch_pipeline(pipeline);
  const size_t dispatch_index = add_dispatch(
      {pipeline,
       bind_group.get(),
       descriptor.grid.x,
       descriptor.kernel_name.empty() ? descriptor.shader_name
                                      : descriptor.kernel_name,
       descriptor.grid.y});
  bind_group.release();
  dispatch_pipeline.release();
  return dispatch_index;
}

size_t WebGPUGraph::add_dynamic_compute_dispatch_impl(
    const WebGPUComputeDispatchDescriptor& descriptor,
    int trigger_tensor_id,
    std::function<WebGPUDispatchGrid(const WebGPUGraph&)> pick_grid) {
  if (trigger_tensor_id < 0 || trigger_tensor_id >= num_values() ||
      get_value_type(trigger_tensor_id) != ValueType::Tensor) {
    throw std::runtime_error(
        "WebGPU dynamic dispatch: trigger must be a Tensor");
  }
  if (!pick_grid) {
    throw std::runtime_error("WebGPU dynamic dispatch: null grid picker");
  }

  const WebGPUDispatchGrid initial_grid = pick_grid(*this);
  if (initial_grid.x == 0 || initial_grid.y == 0) {
    throw std::runtime_error("WebGPU dynamic dispatch: zero grid");
  }

  WebGPUComputeDispatchDescriptor initial_descriptor = descriptor;
  initial_descriptor.grid = initial_grid;

  // Reserve both vectors before creating GPU objects, then stage the sidecar.
  // If GPU-object creation fails, removing the sidecar restores the graph; no
  // operation that can fail remains after add_compute_dispatch succeeds.
  const size_t new_size = dynamic_dispatch_grids_.size() + 1;
  dynamic_dispatch_grids_.reserve(new_size);
  pending_dynamic_dispatch_grids_.reserve(new_size);

  const size_t expected_index = dispatches_.size();
  dynamic_dispatch_grids_.push_back(
      {expected_index, trigger_tensor_id, std::move(pick_grid)});
  try {
    add_compute_dispatch(initial_descriptor);
  } catch (...) {
    dynamic_dispatch_grids_.pop_back();
    throw;
  }
  return expected_index;
}

void WebGPUGraph::validate_dynamic_dispatch_route_ranges(
    const std::vector<utils::DispatchRange>& ranges) const {
  for (const auto& dynamic_grid : dynamic_dispatch_grids_) {
    for (const auto& range : ranges) {
      if (range.begin <= dynamic_grid.dispatch_index &&
          dynamic_grid.dispatch_index < range.end) {
        throw std::runtime_error(
            "WebGPU dispatch cannot have both dynamic-grid and route ownership");
      }
    }
  }
}

void WebGPUGraph::update_symints_from_inputs(
    const std::vector<InputData>& inputs) {
  for (const auto& src : symint_sources_) {
    int pos = -1;
    for (size_t i = 0; i < input_ids_.size(); i++) {
      if (input_ids_[i] == src.input_tensor_id) {
        pos = static_cast<int>(i);
        break;
      }
    }
    if (pos < 0 || pos >= static_cast<int>(inputs.size())) {
      throw std::runtime_error(
          "select_as_symint: source tensor is not a graph input");
    }
    // Live cur_dims: the source may be a dynamic-shape input.
    const auto& dims = tensors_[src.input_tensor_id].cur_dims;
    int dim = normalize_dim(
        src.dim, static_cast<int>(dims.size()), "select_as_symint");
    int index = src.index;
    if (index < 0) {
      index += static_cast<int>(dims[dim]);
    }
    if (index < 0 || index >= static_cast<int>(dims[dim])) {
      throw std::runtime_error("select_as_symint: index out of range");
    }
    int64_t numel = 1;
    for (int64_t d : dims) {
      numel *= d;
    }
    if (numel <= 0) {
      throw std::runtime_error("select_as_symint: empty input tensor");
    }
    int64_t stride = 1;
    for (size_t i = static_cast<size_t>(dim) + 1; i < dims.size(); i++) {
      stride *= dims[i];
    }
    // Reads the [0,..,index,..,0] element; symint sources are scalar-ish.
    const int64_t offset = static_cast<int64_t>(index) * stride;
    const void* host = inputs[pos].data;
    // Interpret the HOST buffer by its scalar type, not the tensor's serialized
    // elem_size: copy_inputs narrows an int64 host input to an int32 buffer, so
    // elem_size (buffer-derived) would misread int64 host data as int32.
    int32_t val;
    if (inputs[pos].host_is_int64) {
      const int64_t raw = static_cast<const int64_t*>(host)[offset];
      if (raw < std::numeric_limits<int32_t>::min() ||
          raw > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error(
            "select_as_symint: selected value is outside int32 range");
      }
      val = static_cast<int32_t>(raw);
    } else {
      val = static_cast<const int32_t*>(host)[offset];
    }
    set_symint(src.symint_id, val);
  }
  // sym_size.int: SymInt = a tensor's live dim (cur_dims). Usually unused (ops
  // read cur_dims directly); for an intermediate source cur_dims is the build
  // max here (hooks run later in propagate_resize), which is fine while unused.
  for (const auto& s : symint_dim_sources_) {
    const auto& d = tensors_[s.tensor_id].cur_dims;
    int dim = normalize_dim(s.dim, static_cast<int>(d.size()), "sym_size");
    set_symint(s.symint_id, static_cast<int32_t>(d[dim]));
  }
}

void WebGPUGraph::set_symint(int id, int32_t val) {
  auto it = symints_.find(id);
  if (it == symints_.end()) {
    throw std::runtime_error("WebGPUGraph::set_symint: id is not a SymInt");
  }
  if (it->second.value != val) {
    it->second.value = val;
    wgpuQueueWriteBuffer(
        queue_, it->second.buffer, 0, &it->second.value, sizeof(int32_t));
    dirty_symints_.insert(id);
  }
}

void WebGPUGraph::set_cur_dims(
    int value_id,
    const std::vector<int64_t>& new_dims) {
  auto& t = tensors_[value_id];
  if (new_dims.size() != t.dims.size()) {
    throw std::runtime_error("WebGPU resize: tensor rank changed");
  }
  size_t numel = 1;
  for (size_t d = 0; d < new_dims.size(); d++) {
    // 0-sized dims unsupported: live shapes are always in [1, max] per dim.
    if (new_dims[d] <= 0) {
      throw std::runtime_error("WebGPU resize: new dim must be positive");
    }
    if (new_dims[d] > t.dims[d]) {
      throw std::runtime_error(
          "WebGPU resize: new dim exceeds the max (serialized) allocation");
    }
    numel *= static_cast<size_t>(new_dims[d]);
  }
  const size_t new_nbytes = numel * t.elem_size;
  if (t.cur_dims != new_dims) {
    t.cur_dims = new_dims;
    t.cur_nbytes = new_nbytes;
    dirty_tensors_.insert(value_id);
  }
}

void WebGPUGraph::resize_input(
    int value_id,
    const std::vector<int64_t>& new_dims) {
  if (std::find(input_ids_.begin(), input_ids_.end(), value_id) ==
      input_ids_.end()) {
    throw std::runtime_error(
        "WebGPUGraph::resize_input: value_id is not a graph input");
  }
  set_cur_dims(value_id, new_dims);
}

void WebGPUGraph::propagate_resize() {
  if (dirty_symints_.empty() && dirty_tensors_.empty()) {
    return;
  }
  // Hooks fire in registration (topological) order: operands update first.
  for (auto& hook : resize_hooks_) {
    if (dirty_symints_.count(hook.symint_id) != 0) {
      hook.fn(*this);
    }
  }
  dirty_symints_.clear();
  // Tensor hooks: bounded fixpoint. A hook may dirty its output (cascading to a
  // consumer); each pass handles the currently-dirty set. A forward DAG
  // converges in <= depth passes (set_cur_dims re-dirties only on a change).
  for (size_t pass = 0;
       !dirty_tensors_.empty() && pass <= tensor_resize_hooks_.size();
       pass++) {
    std::unordered_set<int> processing;
    processing.swap(dirty_tensors_);
    pending_dynamic_dispatch_grids_.clear();
    try {
      for (auto& hook : tensor_resize_hooks_) {
        if (processing.count(hook.trigger_tensor_id) != 0) {
          hook.fn(*this);
        }
      }

      // A hook or picker may fail, so compute and validate every affected grid
      // before changing any dispatch. The graph-owned staging vector has
      // capacity for every registered dynamic grid and is reused on execute.
      for (const auto& dynamic_grid : dynamic_dispatch_grids_) {
        if (processing.count(dynamic_grid.trigger_tensor_id) == 0) {
          continue;
        }
        const WebGPUDispatchGrid grid = dynamic_grid.pick_grid(*this);
        if (grid.x == 0 || grid.y == 0) {
          throw std::runtime_error("WebGPU dynamic dispatch: zero grid");
        }
        pending_dynamic_dispatch_grids_.push_back(
            {dynamic_grid.dispatch_index, grid});
      }
    } catch (...) {
      pending_dynamic_dispatch_grids_.clear();
      // Keep both the current triggers and any cascaded outputs dirty so the
      // caller can fix the hook or picker and retry without rebuilding.
      dirty_tensors_.insert(processing.begin(), processing.end());
      throw;
    }
    for (const auto& pending : pending_dynamic_dispatch_grids_) {
      auto& dispatch = dispatches_[pending.dispatch_index];
      dispatch.workgroup_count_x = pending.grid.x;
      dispatch.workgroup_count_y = pending.grid.y;
    }
    pending_dynamic_dispatch_grids_.clear();
  }
  if (!dirty_tensors_.empty()) {
    throw std::runtime_error(
        "WebGPU resize: tensor resize hooks did not converge");
  }
  // Tensor hooks must not set_symint (dirty_symints_ already drained above).
  if (!dirty_symints_.empty()) {
    throw std::runtime_error(
        "WebGPU resize: a tensor resize hook set a SymInt; not supported");
  }
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
  for (auto& s : scratch_pool_) {
    if (s.buffer) {
      wgpuBufferRelease(s.buffer);
    }
  }
  for (auto& buf : owned_uniform_buffers_) {
    if (buf) {
      wgpuBufferRelease(buf);
    }
  }
  for (auto& kv : symints_) {
    if (kv.second.buffer) {
      wgpuBufferRelease(kv.second.buffer);
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
    size_t constant_data_size,
    const executorch::runtime::NamedDataMap* named_data_map,
    WebGPUGraphConfig config) {
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

  // .pte byte sources for prepack-time constant materialization (build-only).
  constant_data_ = constant_data;
  constant_data_size_ = constant_data_size;
  named_data_map_ = named_data_map;

  // f16 KV cache (runtime opt-in): store K/V caches as f16 iff the opt-in is
  // set AND the device negotiated shader-f16 (fail-closed).
  config_ = config;
  const WebGPUContext* kv_ctx = get_default_webgpu_context();
  kv_f16_ = config_.f16_kv_cache &&
      (kv_ctx != nullptr && kv_ctx->shader_f16_supported);

  // Phase 1: Create all values
  const auto* values = graph->values();
  const int num_vals = values ? values->size() : 0;
  value_types_.resize(num_vals, ValueType::Null);
  tensors_.resize(num_vals);
  tensor_mem_obj_ids_.resize(num_vals, -1);
  ints_.resize(num_vals, 0);
  int_lists_.resize(num_vals);
  value_lists_.resize(num_vals);
  doubles_.resize(num_vals, 0.0);
  bools_.resize(num_vals, false);
  strings_.resize(num_vals);

  // Pre-scan the op chain: a constant may be DEFERRED (no eager GPU buffer; the
  // prepack node materializes it once) only if it is a prepack source AND never
  // a direct arg of a non-prepack op. ValueList args are expanded so a constant
  // reached through a list still counts as a direct use.
  std::unordered_set<int> prepack_src_ids;
  std::unordered_set<int> direct_use_ids;
  const auto* chain_prescan = graph->chain();
  if (chain_prescan) {
    for (unsigned ci = 0; ci < chain_prescan->size(); ci++) {
      const auto* oc = chain_prescan->Get(ci);
      const bool is_prepack = oc->name()->str() == kPrepackOpName;
      const auto* a = oc->args();
      if (!a) {
        continue;
      }
      if (oc->name()->str() == "sym_size.int" && a->size() >= 3 && values) {
        const auto* out = values->Get(a->Get(2));
        if (out && out->value_type() == vkgraph::GraphTypes::SymInt) {
          dynamic_tensor_ids_.insert(static_cast<int>(a->Get(0)));
        }
      }
      // f16 KV: tag sdpa K/V cache values (args[3],[4]) for half-size alloc.
      // Inert unless kv_f16_ (runtime opt-in) is set.
      if (kv_f16_ && a->size() > 4 &&
          oc->name()->str() == "sdpa_with_kv_cache.default") {
        kv_cache_ids_.insert(static_cast<int>(a->Get(3)));
        kv_cache_ids_.insert(static_cast<int>(a->Get(4)));
      }
      for (unsigned j = 0; j < a->size(); j++) {
        int id = static_cast<int>(a->Get(j));
        if (is_prepack && j == 0) {
          prepack_src_ids.insert(id);
        } else if (!is_prepack) {
          direct_use_ids.insert(id);
          const auto* v = values ? values->Get(id) : nullptr;
          if (v && v->value_type() == vkgraph::GraphTypes::ValueList) {
            const auto* items = v->value_as_ValueList()->items();
            if (items) {
              for (unsigned k = 0; k < items->size(); k++) {
                direct_use_ids.insert(static_cast<int>(items->Get(k)));
              }
            }
          }
        }
      }
    }
  }

  // f16 KV defensive guard: fail loud if a non-sdpa op reads an f16 cache.
  // Inert unless kv_f16_ (runtime opt-in) is set.
  if (kv_f16_ && !kv_cache_ids_.empty() && chain_prescan) {
    for (unsigned ci = 0; ci < chain_prescan->size(); ci++) {
      const auto* oc = chain_prescan->Get(ci);
      const std::string nm = oc->name()->str();
      if (nm == "sdpa_with_kv_cache.default" || nm == kPrepackOpName) {
        continue;
      }
      const auto* a = oc->args();
      if (!a) {
        continue;
      }
      for (unsigned j = 0; j < a->size(); j++) {
        const int id = static_cast<int>(a->Get(j));
        if (kv_cache_ids_.count(id) != 0) {
          throw std::runtime_error(
              "WebGPU f16 KV: cache tensor consumed by non-sdpa op '" + nm +
              "' would misread the f16 buffer");
        }
        const auto* value = values ? values->Get(id) : nullptr;
        if (value && value->value_type() == vkgraph::GraphTypes::ValueList) {
          const auto* items = value->value_as_ValueList()->items();
          if (!items) {
            continue;
          }
          for (unsigned k = 0; k < items->size(); k++) {
            if (kv_cache_ids_.count(static_cast<int>(items->Get(k))) != 0) {
              throw std::runtime_error(
                  "WebGPU f16 KV: cache tensor consumed through a ValueList "
                  "by non-sdpa op '" +
                  nm + "' would misread the f16 buffer");
            }
          }
        }
      }
    }
  }

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
            const uint32_t dim = dims->Get(j);
            tensor.dims.push_back(static_cast<int64_t>(dim));
            if (dim != 0 && numel > std::numeric_limits<size_t>::max() / dim) {
              throw std::runtime_error(
                  "WebGPU: tensor element count overflows");
            }
            numel *= dim;
          }
        }
        tensor.elem_size = vk_datatype_size(vk_tensor->datatype());
        if (tensor.elem_size != 0 &&
            numel > std::numeric_limits<size_t>::max() / tensor.elem_size) {
          throw std::runtime_error("WebGPU: tensor byte size overflows");
        }
        tensor.is_int = vk_datatype_is_int(vk_tensor->datatype());
        tensor.is_bool = vk_tensor->datatype() == vkgraph::VkDataType::BOOL;
        tensor.is_int8 = vk_tensor->datatype() == vkgraph::VkDataType::INT8;
        tensor.nbytes = numel * tensor.elem_size;
        // Live dims start == max (serialized upper bound); resize_input shrinks
        // them per call. Static graphs keep cur == max forever.
        tensor.cur_dims = tensor.dims;
        tensor.cur_nbytes = tensor.nbytes;

        // f16 KV cache: dedicated half-size array<f16> buffer. WebGPU
        // zero-initializes freshly-created buffers, so no explicit clear is
        // needed. Inert unless kv_f16_ (runtime opt-in) is set.
        if (kv_f16_ && kv_cache_ids_.count(i) != 0) {
          if (tensor.is_int || tensor.elem_size != sizeof(float) ||
              tensor.nbytes != numel * sizeof(float)) {
            throw std::runtime_error(
                "WebGPU f16 KV: serialized cache tensor must be fp32");
          }
          tensor.elem_size = 2;
          tensor.nbytes = numel * 2;
          tensor.cur_nbytes = tensor.nbytes;
          tensor_mem_obj_ids_[i] = -1;
          WGPUBufferDescriptor buf_desc = {};
          buf_desc.size = storage_buffer_size(tensor.nbytes);
          buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
              WGPUBufferUsage_CopySrc;
          buf_desc.mappedAtCreation = false;
          tensor.buffer = wgpuDeviceCreateBuffer(device_, &buf_desc);

          // Mutable caches normally start empty. If the serialized graph owns
          // an initialized cache constant, preserve it while changing storage
          // representation instead of silently replacing it with zeros.
          const int cache_constant_id = vk_tensor->constant_id();
          if (cache_constant_id >= 0) {
            const auto* constants = graph->constants();
            if (!constants ||
                cache_constant_id >= static_cast<int>(constants->size())) {
              throw std::runtime_error(
                  "WebGPU f16 KV: cache constant id is out of range");
            }
            const auto* bytes = constants->Get(cache_constant_id);
            auto write_fp16_cache = [&](const uint8_t* src) {
              std::vector<executorch::runtime::etensor::Half> converted(numel);
              for (size_t e = 0; e < numel; e++) {
                float value = 0.0f;
                std::memcpy(&value, src + e * sizeof(float), sizeof(float));
                converted[e] = executorch::runtime::etensor::Half(value);
              }
              write_storage_buffer(
                  queue_,
                  tensor.buffer,
                  converted.data(),
                  converted.size() * sizeof(converted[0]));
            };
            if (bytes->offset() != UINT64_MAX) {
              write_fp16_cache(checked_inline_constant(
                  constant_data_,
                  constant_data_size_,
                  bytes->offset(),
                  numel * sizeof(float),
                  "WebGPU f16 KV: inline cache constant exceeds constant "
                  "data"));
            } else if (
                bytes->named_key() != nullptr && named_data_map_ != nullptr) {
              const std::string key = bytes->named_key()->str();
              auto data = named_data_map_->get_data(key.c_str());
              if (!data.ok()) {
                throw std::runtime_error(
                    "WebGPU f16 KV: named cache constant '" + key +
                    "' not found");
              }
              if (data->size() < numel * sizeof(float)) {
                data->Free();
                throw std::runtime_error(
                    "WebGPU f16 KV: named cache constant '" + key +
                    "' is undersized");
              }
              write_fp16_cache(static_cast<const uint8_t*>(data->data()));
              data->Free();
            } else {
              throw std::runtime_error(
                  "WebGPU f16 KV: cache constant has no readable source");
            }
          }
          break;
        }

        int constant_id = vk_tensor->constant_id();
        int mem_obj_id = vk_tensor->mem_obj_id();

        // Constants are dedicated. Every constant is recorded as a
        // ConstantSource and materialized via materialize_constant (one
        // CPU->GPU write); a constant consumed ONLY via prepack is deferred
        // (no eager buffer -- its prepack node performs that one write).
        if (constant_id >= 0 || mem_obj_id < 0) {
          tensor_mem_obj_ids_[i] = -1;

          if (constant_id >= 0) {
            const auto* constants = graph->constants();
            if (!constants ||
                constant_id >= static_cast<int>(constants->size())) {
              throw std::runtime_error(
                  "WebGPU: constant_id set but the constants table is missing "
                  "or the id is out of range");
            }
            const auto* vk_bytes = constants->Get(constant_id);
            ConstantSource cs;
            cs.nbytes = tensor.nbytes;
            if (vk_bytes->offset() != UINT64_MAX) {
              cs.inline_offset = vk_bytes->offset();
            } else if (vk_bytes->named_key() != nullptr) {
              cs.named_key = vk_bytes->named_key()->str();
            } else {
              throw std::runtime_error(
                  "WebGPU: constant has no inline offset and no named-data key");
            }
            constant_sources_[i] = std::move(cs);
          }

          // Defer constants consumed solely via prepack: skip the eager buffer.
          const bool defer = constant_id >= 0 &&
              prepack_src_ids.count(i) != 0 && direct_use_ids.count(i) == 0;
          if (!defer) {
            WGPUBufferDescriptor buf_desc = {};
            buf_desc.size = storage_buffer_size(tensor.nbytes);
            buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                WGPUBufferUsage_CopySrc;
            buf_desc.mappedAtCreation = false;
            tensor.buffer = wgpuDeviceCreateBuffer(device_, &buf_desc);

            // Same single CPU->GPU write the prepack node uses (no
            // duplication).
            if (constant_id >= 0) {
              materialize_constant(i, tensor.buffer);
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
      case vkgraph::GraphTypes::IntList: {
        value_types_[i] = ValueType::IntList;
        const auto* items = val->value_as_IntList()->items();
        if (items) {
          int_lists_[i].assign(items->cbegin(), items->cend());
        }
        break;
      }
      case vkgraph::GraphTypes::ValueList: {
        value_types_[i] = ValueType::ValueList;
        const auto* items = val->value_as_ValueList()->items();
        if (items) {
          value_lists_[i].reserve(items->size());
          for (unsigned j = 0; j < items->size(); j++) {
            value_lists_[i].push_back(static_cast<int>(items->Get(j)));
          }
        }
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
      case vkgraph::GraphTypes::String: {
        value_types_[i] = ValueType::String;
        const auto* sv = val->value_as_String()->string_val();
        if (sv) {
          strings_[i] = sv->str();
        }
        break;
      }
      case vkgraph::GraphTypes::SymInt: {
        // Live scalar: small Uniform buffer the CPU rewrites per execute.
        value_types_[i] = ValueType::SymInt;
        SymIntSlot slot;
        slot.value = static_cast<int32_t>(val->value_as_SymInt()->value());
        // 16B matches the backend uniform-struct alignment; int32 in first 4.
        constexpr size_t kSymIntUniformBytes = 16;
        WGPUBufferDescriptor d = {};
        d.size = kSymIntUniformBytes;
        d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        d.mappedAtCreation = true;
        slot.buffer = wgpuDeviceCreateBuffer(device_, &d);
        void* mapped =
            wgpuBufferGetMappedRange(slot.buffer, 0, kSymIntUniformBytes);
        std::memset(mapped, 0, kSymIntUniformBytes);
        std::memcpy(mapped, &slot.value, sizeof(int32_t));
        wgpuBufferUnmap(slot.buffer);
        symints_[i] = slot;
        add_uniform_buffer_bytes(kSymIntUniformBytes);
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
    buf_desc.size = storage_buffer_size(shared_buffer_sizes_[id]);
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
      staging_desc.size = storage_buffer_size(tensors_[oid].nbytes);
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

  std::vector<passes::SwiGluFusion> swiglu_fusions;
  std::unordered_map<unsigned, size_t> swiglu_gate_producers;
  std::unordered_map<unsigned, size_t> swiglu_anchors;
  std::unordered_set<unsigned> swiglu_skipped_ops;
  std::unordered_set<unsigned> claimed_fusion_ops;

  std::vector<passes::QkvBk64Fusion> qkv_fusions;
  std::unordered_map<unsigned, size_t> qkv_first_ops;
  std::unordered_map<unsigned, size_t> qkv_last_ops;
  std::unordered_map<unsigned, size_t> qkv_member_ops;

  const auto* chain = graph->chain();
  passes::detect_qkv_bk64_fusions(
      *this,
      graph,
      num_vals,
      qkv_fusions,
      qkv_first_ops,
      qkv_last_ops,
      qkv_member_ops);
  passes::detect_swiglu_fusions(
      *this,
      graph,
      num_vals,
      swiglu_fusions,
      swiglu_gate_producers,
      swiglu_anchors,
      swiglu_skipped_ops,
      claimed_fusion_ops);

  // SwiGLU keeps precedence when the exact QKV geometry is also formed by a
  // q projection plus gate/up projections. QKV detection runs first because it
  // validates constant geometry, but it has no side effects until Phase 3; now
  // discard candidates claimed by the completed SwiGLU pass and rebuild the
  // index maps for the retained groups.
  passes::retain_unclaimed_qkv_fusions(
      qkv_fusions,
      qkv_first_ops,
      qkv_last_ops,
      qkv_member_ops,
      claimed_fusion_ops);

  // Phase 3: Build operator dispatch chain
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

      const auto gate_it = swiglu_gate_producers.find(i);
      if (gate_it != swiglu_gate_producers.end()) {
        const int gate_id = swiglu_fusions[gate_it->second].gate_id;
        tensors_[gate_id].buffer = acquire_scratch(tensors_[gate_id].nbytes);
      }
      const auto anchor_it = swiglu_anchors.find(i);
      if (anchor_it != swiglu_anchors.end()) {
        const passes::SwiGluFusion& fusion = swiglu_fusions[anchor_it->second];
        passes::add_silu_mul_fused_dispatch(
            *this,
            fusion.common_input_id,
            fusion.gate_id,
            fusion.up_id,
            fusion.out_id);
        release_scratch(tensors_[fusion.gate_id].buffer);
        continue;
      }
      if (swiglu_skipped_ops.count(i) != 0) {
        continue;
      }

      const auto qkv_first = qkv_first_ops.find(i);
      if (qkv_first != qkv_first_ops.end()) {
        passes::QkvBk64Fusion& fusion = qkv_fusions[qkv_first->second];
        for (int output_id : fusion.output_ids) {
          tensors_[output_id].buffer =
              create_scratch_buffer(tensors_[output_id].nbytes);
        }
      }

      const size_t dispatch_begin = dispatches_.size();
      webgpu_operator_registry().get_op_fn(op_name)(*this, args);
      const size_t dispatch_end = dispatches_.size();

      const auto qkv_member = qkv_member_ops.find(i);
      if (qkv_member != qkv_member_ops.end()) {
        passes::QkvBk64Fusion& fusion = qkv_fusions[qkv_member->second];
        size_t member = 0;
        while (member < 3 && fusion.op_indices[member] != i) {
          member++;
        }
        if (member == 3 || dispatch_end <= dispatch_begin) {
          throw std::runtime_error(
              "linear_q4gsw_bk64_qkv: malformed member dispatch range");
        }
        fusion.separate_begin[member] = dispatch_begin;
        fusion.separate_end[member] = dispatch_end;
        if (member == 0) {
          passes::add_qkv_bk64_dispatch(*this, fusion);
        }
      }
      const auto qkv_last = qkv_last_ops.find(i);
      if (qkv_last != qkv_last_ops.end()) {
        passes::add_qkv_bk64_resize_hook(*this, qkv_fusions[qkv_last->second]);
      }

      if (i + 1 == chain->size() && op_name == kQ4gswLinearOpName &&
          args.size() > kQ4gswOutputArg && dispatch_end > dispatch_begin) {
        const int output_id = args[kQ4gswOutputArg];
        const auto output_it =
            std::find(output_ids_.begin(), output_ids_.end(), output_id);
        if (output_it != output_ids_.end() &&
            std::count(output_ids_.begin(), output_ids_.end(), output_id) ==
                1) {
          suppressible_outputs_.push_back(
              {output_id,
               static_cast<size_t>(output_it - output_ids_.begin()),
               dispatch_begin,
               dispatch_end});
        }
      }
    }
  }

  // Prepack nodes (Phase 3) materialized their constants directly into the
  // consumer buffers via materialize_constant; no separate copy pass needed.
  // The .pte bytes are freed right after build() returns (WebGPUBackend
  // processed->Free()), so clear the build-only source pointers.
  constant_data_ = nullptr;
  constant_data_size_ = 0;
  named_data_map_ = nullptr;
}

void WebGPUGraph::materialize_constant(int const_value_id, WGPUBuffer dst) {
  auto it = constant_sources_.find(const_value_id);
  if (it == constant_sources_.end()) {
    throw std::runtime_error(
        "WebGPU: no source recorded for constant id " +
        std::to_string(const_value_id));
  }
  const ConstantSource& cs = it->second;
  if (cs.inline_offset != UINT64_MAX) {
    const uint8_t* data = checked_inline_constant(
        constant_data_,
        constant_data_size_,
        cs.inline_offset,
        cs.nbytes,
        "WebGPU: inline constant exceeds constant data");
    if (cs.nbytes != 0) {
      write_storage_buffer(queue_, dst, data, cs.nbytes);
    }
  } else if (cs.nbytes == 0) {
    return;
  } else if (!cs.named_key.empty() && named_data_map_ != nullptr) {
    auto buf = named_data_map_->get_data(cs.named_key.c_str());
    if (!buf.ok()) {
      throw std::runtime_error(
          "WebGPU: named constant '" + cs.named_key + "' not found");
    }
    if (buf->size() < cs.nbytes) {
      throw std::runtime_error(
          "WebGPU: named constant '" + cs.named_key + "' undersized");
    }
    write_storage_buffer(queue_, dst, buf->data(), cs.nbytes);
    buf->Free();
  } else {
    throw std::runtime_error("WebGPU: constant has no source");
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
  if (shader == nullptr) {
    throw std::runtime_error("WebGPU: failed to create shader module");
  }

  try {
    shader_cache_.emplace(key, shader);
  } catch (...) {
    wgpuShaderModuleRelease(shader);
    throw;
  }
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

void WebGPUGraph::copy_inputs(const std::vector<InputData>& inputs) {
  for (size_t i = 0; i < inputs.size() && i < input_ids_.size(); i++) {
    const InputData& in = inputs[i];
    if (in.nbytes == 0) {
      continue;
    }
    int tid = input_ids_[i];
    const auto& tensor = tensors_[tid];
    // Upload only the live (cur) bytes, not the max allocation; cur_nbytes ==
    // nbytes on a static graph, so this is byte-identical there.
    const size_t live_nbytes = tensor.cur_nbytes;
    const bool buffer_is_fp16 = !tensor.is_int && tensor.elem_size == 2;
    if (buffer_is_fp16 && !in.host_is_fp32) {
      throw std::runtime_error(
          "WebGPU: fp16 device input requires an fp32 host tensor");
    }

    // Fast path: host and GPU element types match byte-for-byte.
    if (in.nbytes == live_nbytes) {
      write_storage_buffer(queue_, tensor.buffer, in.data, live_nbytes);
      continue;
    }

    // Narrow int64 host indices into the int32 buffer (mirrors Vulkan).
    const bool buffer_is_int32 = tensor.is_int && tensor.elem_size == 4;
    if (in.host_is_int64 && buffer_is_int32 && in.nbytes == live_nbytes * 2) {
      const size_t numel = live_nbytes / 4;
      const int64_t* src = static_cast<const int64_t*>(in.data);
      std::vector<int32_t> narrowed(numel);
      for (size_t e = 0; e < numel; e++) {
#ifndef NDEBUG
        // Index tensors (tokens/positions) are far below int32 range in
        // practice; assert in debug that the narrowing is lossless.
        if (static_cast<int32_t>(src[e]) != src[e]) {
          throw std::runtime_error("WebGPU: int64 index overflows int32");
        }
#endif
        narrowed[e] = static_cast<int32_t>(src[e]);
      }
      write_storage_buffer(queue_, tensor.buffer, narrowed.data(), live_nbytes);
      continue;
    }

    // Require an explicit fp32 host dtype, not merely "not int64": inferring
    // the narrow from the 2:1 byte ratio alone would silently reinterpret a
    // same-sized non-fp32 host buffer (e.g. a stale int32) as fp32.
    if (in.host_is_fp32 && buffer_is_fp16 && in.nbytes == live_nbytes * 2) {
      const size_t numel = live_nbytes / sizeof(uint16_t);
      const float* src = static_cast<const float*>(in.data);
      std::vector<executorch::runtime::etensor::Half> narrowed(numel);
      for (size_t e = 0; e < numel; e++) {
        narrowed[e] = executorch::runtime::etensor::Half(src[e]);
      }
      write_storage_buffer(queue_, tensor.buffer, narrowed.data(), live_nbytes);
      continue;
    }

    throw std::runtime_error(
        "WebGPU: unsupported input copy for input " + std::to_string(i) +
        " (host " + std::to_string(in.nbytes) + " bytes" +
        (in.host_is_int64 ? " int64" : "") + " vs buffer " +
        std::to_string(live_nbytes) + " bytes)");
  }
}

#ifdef WGPU_BACKEND_ENABLE_PROFILING
// Profiling/attestation only; never compiled into a production build. Written
// during WebGPUGraph::execute without synchronization: the attestation
// harnesses that read them run one graph on one thread, so no atomics or
// locking are needed. To support concurrent profiled execution, make these
// per-instance behind a whole-record mutex (per-field atomics would not cover
// the conflict check's read-modify-write across both globals).
uint32_t g_last_route_mask = 0;
uint32_t g_last_route_conflict_count = 0;
#endif // WGPU_BACKEND_ENABLE_PROFILING

namespace {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
constexpr uint32_t kRoutePrefill = 1u << 0;
constexpr uint32_t kRouteK16 = 1u << 1;
constexpr uint32_t kRouteMaterializedAttention = 1u << 2;
constexpr uint32_t kRouteT0Steel = 1u << 3;
constexpr uint32_t kRouteT1Bk64 = 1u << 4;
constexpr uint32_t kRouteT1Bk64Qkv = 1u << 5;
constexpr uint32_t kRouteT2PairedGateUp = 1u << 6;
constexpr uint32_t kRouteFusedSwiGlu = 1u << 7;
constexpr uint32_t kRouteGenericFallback = 1u << 8;
// bit 1u << 9 is intentionally reserved (a retired route) and left unused.
constexpr uint32_t kRouteFlashDecoding = 1u << 10;
constexpr uint32_t kRouteK16CausalBound = 1u << 11;
constexpr uint32_t kRouteBicolSubgroup = 1u << 12;
constexpr uint32_t kRouteQwen3Q16K16 = 1u << 13;
constexpr uint32_t kRouteQwen3Q32K16 = 1u << 14;
bool should_timestamp_query() {
  return std::getenv("WEBGPU_TIMESTAMP_QUERY") != nullptr;
}
#endif // WGPU_BACKEND_ENABLE_PROFILING
} // namespace

#ifdef WGPU_BACKEND_ENABLE_PROFILING
void WebGPUGraph::record_active_route(const std::string& kernel_name) {
  uint32_t bits = 0;
  if (kernel_name == "sdpa_streaming_attention_qwen3_q32_k16_causal_bound") {
    bits = kRoutePrefill | kRouteK16CausalBound | kRouteQwen3Q32K16;
  } else if (kernel_name == "sdpa_streaming_attention_qwen3_k16_causal_bound") {
    bits = kRoutePrefill | kRouteK16CausalBound | kRouteQwen3Q16K16;
  } else if (
      kernel_name.rfind("sdpa_streaming_attention_", 0) == 0 &&
      kernel_name.find("k16_causal_bound") != std::string::npos) {
    bits = kRoutePrefill | kRouteK16CausalBound;
  } else if (kernel_name == "sdpa_streaming_attention_k16") {
    bits = kRoutePrefill | kRouteK16;
  } else if (
      kernel_name.rfind("sdpa_compute_", 0) == 0 ||
      kernel_name == "sdpa_softmax") {
    bits = kRoutePrefill | kRouteMaterializedAttention;
  } else if (kernel_name == "fd_split" || kernel_name == "fd_reduce") {
    bits = kRouteFlashDecoding;
  } else if (kernel_name == "linear_q4gsw_coop4_bicol_subgroup") {
    bits = kRouteBicolSubgroup;
  } else if (kernel_name.rfind("linear_q4gsw_bk64_qkv", 0) == 0) {
    bits = kRouteT1Bk64Qkv;
  } else if (kernel_name.rfind("linear_q4gsw_bk64", 0) == 0) {
    bits = kRouteT1Bk64;
  } else if (kernel_name.rfind("linear_q4gsw_paired_gate_up", 0) == 0) {
    bits = kRouteT2PairedGateUp;
  } else if (kernel_name == "silu_mul_fused") {
    bits = kRouteFusedSwiGlu;
  } else if (kernel_name.rfind("linear_q4gsw_steel", 0) == 0) {
    bits = kRouteT0Steel;
  } else if (kernel_name.rfind("linear_q4gsw", 0) == 0) {
    bits = kRouteGenericFallback;
  }

  constexpr uint32_t kAttentionRoutes = kRouteK16 | kRouteK16CausalBound |
      kRouteMaterializedAttention | kRouteFlashDecoding;
  const uint32_t new_attention = bits & kAttentionRoutes;
  const uint32_t prior_attention = g_last_route_mask & kAttentionRoutes;
  if (new_attention != 0 && prior_attention != 0 &&
      (new_attention & prior_attention) == 0) {
    ++g_last_route_conflict_count;
  }
  g_last_route_mask |= bits;
}
#endif // WGPU_BACKEND_ENABLE_PROFILING

WebGPUExecutionPlan WebGPUGraph::make_execution_plan(
    const WebGPUGraphExecutionOptions& options) const {
  const size_t n = dispatches_.size();
  std::vector<bool> enabled_dispatches(n, true);
  for (size_t i = 0; i < n; i++) {
    if (dispatches_[i].kind != WebGPUDispatch::Kind::Compute) {
      continue;
    }
    const bool zero_x = dispatches_[i].workgroup_count_x == 0;
    const bool zero_y = dispatches_[i].workgroup_count_y == 0;
    if (zero_x != zero_y) {
      throw std::runtime_error("WebGPU: dispatch has a half-zero grid");
    }
    enabled_dispatches[i] = !zero_x;
  }
  return plan_webgpu_execution(
      n,
      output_copies_.size(),
      execute_config_,
      suppressible_outputs_,
      options,
      enabled_dispatches);
}

size_t WebGPUGraph::execute(const WebGPUExecutionPlan& plan) {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  g_last_route_mask = 0;
  g_last_route_conflict_count = 0;
#endif // WGPU_BACKEND_ENABLE_PROFILING
  const size_t n = dispatches_.size();
  const size_t chunk = execute_config_.chunk_size;
  if (plan.copy_outputs.size() != output_copies_.size()) {
    throw std::runtime_error("WebGPU: execution plan output count mismatch");
  }
  for (const auto& dispatch_chunk : plan.dispatch_chunks) {
    for (size_t dispatch_index : dispatch_chunk) {
      if (dispatch_index >= n) {
        throw std::runtime_error(
            "WebGPU: execution plan dispatch index out of range");
      }
    }
  }

  if (plan.dispatch_chunks.empty()) {
    return 0;
  }

  if (chunk == 0 || n <= chunk) {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
    size_t active_compute_count = 0;
    for (size_t i : plan.dispatch_chunks.front()) {
      if (dispatches_[i].kind == WebGPUDispatch::Kind::Compute) {
        active_compute_count++;
      }
    }
    // Bench: timestamp-query pool, null unless env-gated + feature present.
    WebGPUQueryPool* qp = nullptr;
    if (should_timestamp_query() && active_compute_count > 0) {
      if (auto* ctx = get_default_webgpu_context()) {
        if (ctx->timestamp_supported) {
          if (!ctx->querypool ||
              ctx->querypool->capacity() < active_compute_count) {
            ctx->querypool = std::make_unique<WebGPUQueryPool>();
            ctx->querypool->initialize(
                device_, static_cast<uint32_t>(active_compute_count));
          }
          qp = ctx->querypool.get();
          qp->reset(static_cast<uint32_t>(active_compute_count));
        }
      }
    }
#endif // WGPU_BACKEND_ENABLE_PROFILING

    WGPUCommandEncoderDescriptor enc_desc = {};
    WGPUCommandEncoder encoder =
        wgpuDeviceCreateCommandEncoder(device_, &enc_desc);

    // One pass per dispatch: enforces storage RAW ordering across deps.
#ifdef WGPU_BACKEND_ENABLE_PROFILING
    uint32_t query_index = 0;
#endif
    for (const auto& dispatch_chunk : plan.dispatch_chunks) {
      for (size_t i : dispatch_chunk) {
        const auto& dispatch = dispatches_[i];
        if (dispatch.kind == WebGPUDispatch::Kind::Copy) {
          wgpuCommandEncoderCopyBufferToBuffer(
              encoder,
              dispatch.copy_src,
              0,
              dispatch.copy_dst,
              0,
              dispatch.copy_nbytes);
          continue;
        }
#ifdef WGPU_BACKEND_ENABLE_PROFILING
        record_active_route(dispatch.kernel_name);
#endif // WGPU_BACKEND_ENABLE_PROFILING
        WGPUComputePassDescriptor pass_desc = {};
#ifdef WGPU_BACKEND_ENABLE_PROFILING
        // tw must outlive BeginComputePass (the descriptor points at it).
        WGPUPassTimestampWrites tw = {};
        if (qp) {
          tw = qp->writes_for(query_index);
          pass_desc.timestampWrites = &tw;
        }
#endif // WGPU_BACKEND_ENABLE_PROFILING
        WGPUComputePassEncoder pass =
            wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
        wgpuComputePassEncoderSetPipeline(pass, dispatch.pipeline);
        wgpuComputePassEncoderSetBindGroup(
            pass, 0, dispatch.bind_group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(
            pass, dispatch.workgroup_count_x, dispatch.workgroup_count_y, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
#ifdef WGPU_BACKEND_ENABLE_PROFILING
        if (qp) {
          qp->record(
              query_index,
              dispatch.kernel_name,
              {dispatch.workgroup_count_x, dispatch.workgroup_count_y, 1},
              {1, 1, 1});
          query_index++;
        }
#endif // WGPU_BACKEND_ENABLE_PROFILING
      }
    }

    for (size_t i = 0; i < output_copies_.size(); i++) {
      const size_t logical_nbytes = tensors_[output_ids_[i]].cur_nbytes;
      if (!plan.copy_outputs[i] || logical_nbytes == 0) {
        continue;
      }
      const size_t copy_nbytes = storage_buffer_size(logical_nbytes);
      const auto& copy = output_copies_[i];
      wgpuCommandEncoderCopyBufferToBuffer(
          encoder, copy.src_buffer, 0, copy.staging_buffer, 0, copy_nbytes);
    }

#ifdef WGPU_BACKEND_ENABLE_PROFILING
    if (qp) {
      qp->resolve(encoder);
    }
#endif // WGPU_BACKEND_ENABLE_PROFILING

    WGPUCommandBufferDescriptor cmd_desc = {};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(queue_, 1, &cmd);

    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);

#ifdef WGPU_BACKEND_ENABLE_PROFILING
    if (qp) {
      qp->extract_results(instance_);
      qp->print_results();
    }
#endif // WGPU_BACKEND_ENABLE_PROFILING
    return 1;
  }

#ifdef WGPU_BACKEND_ENABLE_PROFILING
  if (should_timestamp_query()) {
    throw std::runtime_error(
        "WebGPU: WEBGPU_TIMESTAMP_QUERY is incompatible with chunked execute "
        "(multi-submit); disable chunking to use GPU timestamp queries");
  }
#endif // WGPU_BACKEND_ENABLE_PROFILING

  for (size_t chunk_index = 0; chunk_index < plan.dispatch_chunks.size();
       chunk_index++) {
    WGPUCommandEncoderDescriptor enc_desc = {};
    WGPUCommandEncoder encoder =
        wgpuDeviceCreateCommandEncoder(device_, &enc_desc);

    for (size_t i : plan.dispatch_chunks[chunk_index]) {
      if (dispatches_[i].kind == WebGPUDispatch::Kind::Copy) {
        wgpuCommandEncoderCopyBufferToBuffer(
            encoder,
            dispatches_[i].copy_src,
            0,
            dispatches_[i].copy_dst,
            0,
            dispatches_[i].copy_nbytes);
        continue;
      }
#ifdef WGPU_BACKEND_ENABLE_PROFILING
      record_active_route(dispatches_[i].kernel_name);
#endif // WGPU_BACKEND_ENABLE_PROFILING
      WGPUComputePassDescriptor pass_desc = {};
      WGPUComputePassEncoder pass =
          wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
      wgpuComputePassEncoderSetPipeline(pass, dispatches_[i].pipeline);
      wgpuComputePassEncoderSetBindGroup(
          pass, 0, dispatches_[i].bind_group, 0, nullptr);
      wgpuComputePassEncoderDispatchWorkgroups(
          pass,
          dispatches_[i].workgroup_count_x,
          dispatches_[i].workgroup_count_y,
          1);
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass);
    }

    if (chunk_index + 1 == plan.dispatch_chunks.size()) {
      for (size_t i = 0; i < output_copies_.size(); i++) {
        const size_t logical_nbytes = tensors_[output_ids_[i]].cur_nbytes;
        if (!plan.copy_outputs[i] || logical_nbytes == 0) {
          continue;
        }
        const size_t copy_nbytes = storage_buffer_size(logical_nbytes);
        const auto& copy = output_copies_[i];
        wgpuCommandEncoderCopyBufferToBuffer(
            encoder, copy.src_buffer, 0, copy.staging_buffer, 0, copy_nbytes);
      }
    }

    WGPUCommandBufferDescriptor cmd_desc = {};
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    wgpuQueueSubmit(queue_, 1, &cmd);

    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
  }
  return plan.dispatch_chunks.size();
}

namespace {

struct MapCallbackData {
  WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
};

using MapCallbackDataPtr = std::shared_ptr<MapCallbackData>;

void buffer_map_callback(
    WGPUMapAsyncStatus status,
    WGPUStringView /*message*/,
    void* userdata1,
    void* /*userdata2*/) {
  std::unique_ptr<MapCallbackDataPtr> data_owner(
      static_cast<MapCallbackDataPtr*>(userdata1));
  (*data_owner)->status = status;
}

} // namespace

void WebGPUGraph::copy_outputs(
    std::vector<OutputData>& outputs,
    const WebGPUExecutionPlan& plan) {
  if (plan.copy_outputs.size() != output_copies_.size()) {
    throw std::runtime_error("WebGPU: execution plan output count mismatch");
  }
  const size_t count = std::min(outputs.size(), output_staging_buffers_.size());

  // Reject all dtype/size mismatches before issuing an asynchronous map.
  for (size_t i = 0; i < count; i++) {
    if (!plan.copy_outputs[i] || outputs[i].nbytes == 0) {
      continue;
    }
    const auto& tensor = tensors_[output_ids_[i]];
    const size_t logical_nbytes = tensor.cur_nbytes;
    if (logical_nbytes == 0) {
      continue;
    }
    const size_t dst_nbytes = outputs[i].nbytes;
    const bool is_double_width =
        dst_nbytes % 2 == 0 && dst_nbytes / 2 == logical_nbytes;
    const bool widen_fp16 =
        is_double_width && !tensor.is_int && tensor.elem_size == 2;
    const bool widen_int32 =
        is_double_width && tensor.is_int && tensor.elem_size == 4;
    const bool buffer_is_fp16 = !tensor.is_int && tensor.elem_size == 2;
    if (buffer_is_fp16 && !outputs[i].host_is_fp32) {
      throw std::runtime_error(
          "WebGPU: fp16 device output requires an fp32 host tensor");
    }
    if (outputs[i].host_is_fp32 && buffer_is_fp16 && !widen_fp16) {
      throw std::runtime_error("WebGPU: fp16 output buffer size mismatch");
    }
    if (dst_nbytes != logical_nbytes && !widen_fp16 && !widen_int32) {
      throw std::runtime_error("WebGPU: output buffer size mismatch");
    }
  }

  for (size_t i = 0; i < count; i++) {
    if (!plan.copy_outputs[i] || outputs[i].nbytes == 0) {
      continue;
    }
    const auto& tensor = tensors_[output_ids_[i]];
    const size_t logical_nbytes = tensor.cur_nbytes;
    if (logical_nbytes == 0) {
      continue;
    }
    const size_t map_nbytes = storage_buffer_size(logical_nbytes);
    const size_t dst_nbytes = outputs[i].nbytes;
    const bool is_double_width =
        dst_nbytes % 2 == 0 && dst_nbytes / 2 == logical_nbytes;
    const bool widen_fp16 =
        is_double_width && !tensor.is_int && tensor.elem_size == 2;
    const bool widen_int32 =
        is_double_width && tensor.is_int && tensor.elem_size == 4;

    const auto cb_data = std::make_shared<MapCallbackData>();
    WGPUBufferMapCallbackInfo cb_info = {};
    cb_info.mode = WGPUCallbackMode_WaitAnyOnly;
    cb_info.callback = buffer_map_callback;
    cb_info.userdata1 = new MapCallbackDataPtr(cb_data);
    const WGPUFuture map_future = wgpuBufferMapAsync(
        output_staging_buffers_[i], WGPUMapMode_Read, 0, map_nbytes, cb_info);
    if (webgpu_wait(instance_, map_future) != WGPUWaitStatus_Success) {
      // Cancel the outstanding request, then drain its WaitAny-only callback
      // when possible. The callback owns a shared reference, so even a failed
      // drain cannot leave it pointing at stack-owned storage.
      // An undrained callback may still fire; leaking beats a use-after-free.
      wgpuBufferUnmap(output_staging_buffers_[i]);
      const WGPUWaitStatus drain_status = webgpu_wait(instance_, map_future);
      if (drain_status != WGPUWaitStatus_Success) {
        throw std::runtime_error(
            "WebGPU: output map cancellation callback did not drain");
      }
      throw std::runtime_error("WebGPU: WaitAny failed for output map");
    }
    if (cb_data->status != WGPUMapAsyncStatus_Success) {
      throw std::runtime_error("WebGPU buffer map failed for output");
    }
    const void* mapped = wgpuBufferGetConstMappedRange(
        output_staging_buffers_[i], 0, map_nbytes);
    if (mapped == nullptr) {
      wgpuBufferUnmap(output_staging_buffers_[i]);
      throw std::runtime_error("WebGPU mapped output range is null");
    }
    if (widen_fp16) {
      const auto* src =
          static_cast<const executorch::runtime::etensor::Half*>(mapped);
      auto* dst = static_cast<float*>(outputs[i].data);
      const size_t n = logical_nbytes / sizeof(*src);
      for (size_t k = 0; k < n; k++) {
        dst[k] = static_cast<float>(src[k]);
      }
    } else if (widen_int32) {
      // int64 host output backed by an int32 GPU buffer: widen (sign-extend).
      const int32_t* src = static_cast<const int32_t*>(mapped);
      int64_t* dst = static_cast<int64_t*>(outputs[i].data);
      const size_t n = logical_nbytes / sizeof(int32_t);
      for (size_t k = 0; k < n; k++) {
        dst[k] = static_cast<int64_t>(src[k]);
      }
    } else {
      std::memcpy(outputs[i].data, mapped, logical_nbytes);
    }
    wgpuBufferUnmap(output_staging_buffers_[i]);
  }
}

WebGPUMemoryStats WebGPUGraph::memory_stats() const {
  WebGPUMemoryStats stats;
  for (size_t i = 0; i < value_types_.size(); i++) {
    if (value_types_[i] == ValueType::Tensor && tensors_[i].nbytes > 0) {
      stats.num_tensors++;
      // Shared tensors are tracked via shared_buffer_sizes_; a deferred
      // prepack-routed constant has no buffer (no GPU memory) -> not counted.
      bool is_shared =
          i < tensor_mem_obj_ids_.size() && tensor_mem_obj_ids_[i] >= 0;
      if (!is_shared && tensors_[i].buffer != nullptr) {
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
