/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_linear_gemm_qkv_fused_wgsl.h>

#include <executorch/backends/vulkan/serialization/schema_generated.h>
#include <executorch/runtime/core/named_data_map.h>

#include <executorch/backends/webgpu/runtime/WebGPUCompat.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace executorch::backends::webgpu {

// vkgraph namespace is declared at global scope in the generated FlatBuffer
// header

namespace {

// Op name the AOT exporter emits for a prepacked constant (must match the
// serialized schema); compared in the prepack pre-scan below.
constexpr const char* kPrepackOpName = "et_vk.prepack.default";

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

// Uniform layout matching the fused-QKV WGSL Params struct (16B-aligned, 32B);
// identical to QuantizedLinear.cpp's Q4gswParams (kept local to this TU).
struct QkvFusedParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t K_packed;
  uint32_t group_size;
  uint32_t padded_N;
  uint32_t has_bias;
  uint32_t _pad;
};
static_assert(sizeof(QkvFusedParams) == 32, "QkvFusedParams must be 32 bytes");

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

WGPUBuffer WebGPUGraph::acquire_scratch(size_t nbytes) {
  nbytes = nbytes > 0 ? nbytes : 4;
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
  WGPUBufferDescriptor desc = {};
  desc.size = size;
  desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  desc.mappedAtCreation = true;
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device_, &desc);
  void* mapped = wgpuBufferGetMappedRange(buffer, 0, size);
  std::memcpy(mapped, data, size);
  wgpuBufferUnmap(buffer);
  uniform_buffer_bytes_ += size;
  return buffer;
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
      val = static_cast<int32_t>(static_cast<const int64_t*>(host)[offset]);
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
    for (auto& hook : tensor_resize_hooks_) {
      if (processing.count(hook.trigger_tensor_id) != 0) {
        hook.fn(*this);
      }
    }
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
    const executorch::runtime::NamedDataMap* named_data_map,
    bool f16_kv_cache,
    bool f16_accumulate_gemm) {
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
  named_data_map_ = named_data_map;

  // f16 KV cache (runtime opt-in): store K/V caches as f16 iff the opt-in is
  // set AND the device negotiated shader-f16 (fail-closed).
  const WebGPUContext* kv_ctx = get_default_webgpu_context();
  kv_f16_ = f16_kv_cache && (kv_ctx != nullptr && kv_ctx->shader_f16_supported);

  // f16-accumulate q4gsw steel prefill GEMM (runtime opt-in). QuantizedLinear
  // additionally gates the kernel on the negotiated shader-f16 feature.
  f16_accumulate_gemm_ = f16_accumulate_gemm;

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
        if (kv_cache_ids_.count(static_cast<int>(a->Get(j))) != 0) {
          throw std::runtime_error(
              "WebGPU f16 KV: cache tensor consumed by non-sdpa op '" + nm +
              "' would misread the f16 buffer");
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
            tensor.dims.push_back(static_cast<int64_t>(dims->Get(j)));
            numel *= dims->Get(j);
          }
        }
        tensor.elem_size = vk_datatype_size(vk_tensor->datatype());
        tensor.is_int = vk_datatype_is_int(vk_tensor->datatype());
        tensor.nbytes = numel * tensor.elem_size;
        // Live dims start == max (serialized upper bound); resize_input shrinks
        // them per call. Static graphs keep cur == max forever.
        tensor.cur_dims = tensor.dims;
        tensor.cur_nbytes = tensor.nbytes;

        // f16 KV cache: dedicated half-size array<f16> buffer. WebGPU
        // zero-initializes freshly-created buffers, so no explicit clear is
        // needed. Inert unless kv_f16_ (runtime opt-in) is set.
        if (kv_f16_ && kv_cache_ids_.count(i) != 0) {
          tensor.elem_size = 2;
          tensor.nbytes = numel * 2;
          tensor.cur_nbytes = tensor.nbytes;
          tensor_mem_obj_ids_[i] = -1;
          WGPUBufferDescriptor buf_desc = {};
          buf_desc.size = std::max(tensor.nbytes, size_t(4));
          buf_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
              WGPUBufferUsage_CopySrc;
          buf_desc.mappedAtCreation = false;
          tensor.buffer = wgpuDeviceCreateBuffer(device_, &buf_desc);
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
            buf_desc.size = std::max(tensor.nbytes, size_t(4));
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

  // QKV-concat fusion detection (WEBGPU_QKV_FUSE; default OFF -> both sets stay
  // empty -> the Phase-3 loop below runs verbatim = byte-identical). Find each
  // attention q/k/v triple: EXACTLY 3 et_vk.linear_q4gsw ops sharing args[0]
  // (the same input activation), in chain order q,k,v with the N-pattern
  // {2048,512,512}, on the steel route (K%16==0), group_size%16==0, no bias.
  // The fused kernel needs shader-f16 + a 256-thread WG, so gate on those (else
  // leave the triple to the normal per-linear handlers). qkv_fused_skip holds
  // all 3 op indices; qkv_anchor maps the FIRST op index -> its group, so the
  // fused dispatch is emitted IN-PLACE at the anchor (correct execution order).
  std::vector<QkvFusionGroup> qkv_groups;
  std::unordered_map<unsigned, size_t>
      qkv_first; // first triple op -> group (repoint buffers)
  std::unordered_map<unsigned, size_t>
      qkv_last; // last triple op  -> group (emit fused)
  std::unordered_map<unsigned, size_t>
      qkv_member; // any triple op   -> group (record dispatch)
  if (std::getenv("WEBGPU_QKV_FUSE") != nullptr && chain) {
    bool device_ok = false;
    {
      WGPULimits limits = {};
      const bool have =
          wgpuDeviceGetLimits(device_, &limits) == WGPUStatus_Success;
      bool f16 = false;
      if (auto* ctx = get_default_webgpu_context()) {
        f16 = ctx->shader_f16_supported;
      }
      device_ok =
          have && f16 && limits.maxComputeInvocationsPerWorkgroup >= 256u;
    }
    if (device_ok) {
      // Group linear_q4gsw op indices by input id, preserving chain order.
      std::unordered_map<int, std::vector<unsigned>> by_input;
      std::vector<int> input_order;
      for (unsigned i = 0; i < chain->size(); i++) {
        const auto* oc = chain->Get(i);
        if (oc->name()->str() != "et_vk.linear_q4gsw.default") {
          continue;
        }
        const auto* a = oc->args();
        if (!a || a->size() < 6) {
          continue;
        }
        const int inp = static_cast<int>(a->Get(0));
        if (by_input.find(inp) == by_input.end()) {
          input_order.push_back(inp);
        }
        by_input[inp].push_back(i);
      }
      auto op_arg = [&](unsigned oi, unsigned j) {
        return static_cast<int>(chain->Get(oi)->args()->Get(j));
      };
      for (int inp : input_order) {
        const auto& ops = by_input[inp];
        if (ops.size() != 3) {
          continue; // gate+up is a 2-group; o/down/lm_head are 1 each.
        }
        // args: [in, weight, scales, group_size, bias, out].
        const int wq = op_arg(ops[0], 1), sqid = op_arg(ops[0], 2),
                  bq = op_arg(ops[0], 4), oq = op_arg(ops[0], 5);
        const int wk = op_arg(ops[1], 1), skid = op_arg(ops[1], 2),
                  bk = op_arg(ops[1], 4), ok = op_arg(ops[1], 5);
        const int wv = op_arg(ops[2], 1), svid = op_arg(ops[2], 2),
                  bv = op_arg(ops[2], 4), ov = op_arg(ops[2], 5);
        const int gsid = op_arg(ops[0], 3);
        if (op_arg(ops[1], 3) != gsid || op_arg(ops[2], 3) != gsid) {
          continue; // all 3 must share the group_size scalar.
        }
        if (get_value_type(bq) == ValueType::Tensor ||
            get_value_type(bk) == ValueType::Tensor ||
            get_value_type(bv) == ValueType::Tensor) {
          continue; // fused kernel path assumes has_bias == 0.
        }
        const auto& twq = tensors_[wq];
        const auto& twk = tensors_[wk];
        const auto& twv = tensors_[wv];
        if (twq.dims.size() != 2 || twk.dims.size() != 2 ||
            twv.dims.size() != 2) {
          continue;
        }
        const uint32_t Nq = static_cast<uint32_t>(twq.dims[0]);
        const uint32_t Nk = static_cast<uint32_t>(twk.dims[0]);
        const uint32_t Nv = static_cast<uint32_t>(twv.dims[0]);
        if (Nq != 2048u || Nk != 512u || Nv != 512u) {
          continue; // kernel hardcodes N_Q=2048, N_KV=512 (Llama-3.2 GQA).
        }
        const uint32_t K_packed = static_cast<uint32_t>(twq.dims[1]);
        if (static_cast<uint32_t>(twk.dims[1]) != K_packed ||
            static_cast<uint32_t>(twv.dims[1]) != K_packed) {
          continue;
        }
        const auto& tin = tensors_[inp];
        if (tin.dims.empty()) {
          continue;
        }
        const uint32_t K = static_cast<uint32_t>(tin.dims.back());
        if (K == 0 || K % 16u != 0u || K_packed != (K + 1u) / 2u) {
          continue; // steel route stages a full BK=16 K-tile with no K-mask.
        }
        if (get_value_type(gsid) != ValueType::Int) {
          continue;
        }
        const int64_t gsv = get_int(gsid);
        if (gsv <= 0 || static_cast<uint32_t>(gsv) % 16u != 0u) {
          continue; // hoisted scale must be constant across the BK tile.
        }
        const uint32_t gs = static_cast<uint32_t>(gsv);
        const auto& tsq = tensors_[sqid];
        const auto& tsk = tensors_[skid];
        const auto& tsv = tensors_[svid];
        if (tsq.dims.size() != 2 || tsk.dims.size() != 2 ||
            tsv.dims.size() != 2) {
          continue;
        }
        const uint32_t num_groups = static_cast<uint32_t>(tsq.dims[0]);
        if (static_cast<uint32_t>(tsk.dims[0]) != num_groups ||
            static_cast<uint32_t>(tsv.dims[0]) != num_groups) {
          continue;
        }
        const uint32_t pNq = static_cast<uint32_t>(tsq.dims[1]);
        const uint32_t pNk = static_cast<uint32_t>(tsk.dims[1]);
        const uint32_t pNv = static_cast<uint32_t>(tsv.dims[1]);
        if (pNq < Nq || pNk < Nk || pNv < Nv ||
            num_groups < (K + gs - 1u) / gs) {
          continue;
        }
        // All source + destination buffers must be live (Phase 1/2 allocated).
        if (!twq.buffer || !twk.buffer || !twv.buffer || !tsq.buffer ||
            !tsk.buffer || !tsv.buffer || !tin.buffer || !tensors_[oq].buffer ||
            !tensors_[ok].buffer || !tensors_[ov].buffer) {
          continue;
        }

        QkvFusionGroup grp;
        grp.input_id = inp;
        grp.out_q = oq;
        grp.out_k = ok;
        grp.out_v = ov;
        grp.weight_q = wq;
        grp.weight_k = wk;
        grp.weight_v = wv;
        grp.scales_q = sqid;
        grp.scales_k = skid;
        grp.scales_v = svid;
        grp.Nq = Nq;
        grp.Nk = Nk;
        grp.Nv = Nv;
        grp.K = K;
        grp.K_packed = K_packed;
        grp.group_size = gs;
        grp.num_groups = num_groups;
        grp.padded_N_q = pNq;
        grp.padded_N_k = pNk;
        grp.padded_N_v = pNv;
        grp.op_idx[0] = ops[0];
        grp.op_idx[1] = ops[1];
        grp.op_idx[2] = ops[2];
        const size_t gidx = qkv_groups.size();
        qkv_groups.push_back(grp);
        qkv_first[ops[0]] = gidx;
        qkv_last[ops[2]] = gidx;
        qkv_member[ops[0]] = gidx;
        qkv_member[ops[1]] = gidx;
        qkv_member[ops[2]] = gidx;
      }
    }
  }

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

      // QKV fusion (M-gated): keep the 3 separate q/k/v linears AND add a fused
      // multi-output GEMM; the fused resize hook selects by LIVE M (prefill M>1
      // -> fused runs, the 3 zeroed; decode M==1 -> the 3 coop4 GEMVs run,
      // fused zeroed -- the fused 64x64 tile is ~4x slower than coop4 at M=1).
      // At the FIRST triple op, repoint the 3 outputs to FRESH distinct
      // buffers: the planner reuse-aliases q/k/v (each dies right after RoPE),
      // which is fatal for a simultaneous fused write, so BOTH paths use
      // non-aliased storage. All maps empty when the toggle is off (verbatim
      // path).
      {
        auto fit = qkv_first.find(i);
        if (fit != qkv_first.end()) {
          const auto& g = qkv_groups[fit->second];
          tensors_[g.out_q].buffer =
              create_scratch_buffer(tensors_[g.out_q].nbytes);
          tensors_[g.out_k].buffer =
              create_scratch_buffer(tensors_[g.out_k].nbytes);
          tensors_[g.out_v].buffer =
              create_scratch_buffer(tensors_[g.out_v].nbytes);
        }
      }

      webgpu_operator_registry().get_op_fn(op_name)(*this, args);

      {
        auto mit = qkv_member.find(i);
        if (mit != qkv_member.end()) {
          QkvFusionGroup& g = qkv_groups[mit->second];
          const size_t di = num_dispatches() - 1; // this linear's dispatch
          if (i == g.op_idx[0]) {
            g.sep_dispatch[0] = di;
            // Emit the fused dispatch RIGHT AFTER the q-linear (the anchor) so
            // at M>1 it writes q/k/v BEFORE any consumer. q/k/v may be
            // interleaved with rope in the chain, so emitting it at the LAST
            // triple op would let a consumer (rope-q) read still-unwritten
            // fresh_q -> garbage.
            add_qkv_fused_dispatch(g);
          } else if (i == g.op_idx[1]) {
            g.sep_dispatch[1] = di;
          } else {
            g.sep_dispatch[2] = di;
          }
        }
        auto lit = qkv_last.find(i);
        if (lit != qkv_last.end()) {
          // All 3 sep dispatch indices + the fused index are now known.
          add_qkv_fused_hook(qkv_groups[lit->second]);
        }
      }
    }
  }

  // Prepack nodes (Phase 3) materialized their constants directly into the
  // consumer buffers via materialize_constant; no separate copy pass needed.
  // The .pte bytes are freed right after build() returns (WebGPUBackend
  // processed->Free()), so clear the build-only source pointers.
  constant_data_ = nullptr;
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
  if (cs.nbytes == 0) {
    return;
  }
  if (cs.inline_offset != UINT64_MAX) {
    if (constant_data_ == nullptr) {
      throw std::runtime_error("WebGPU: inline constant data is null");
    }
    wgpuQueueWriteBuffer(
        queue_, dst, 0, constant_data_ + cs.inline_offset, cs.nbytes);
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
    wgpuQueueWriteBuffer(queue_, dst, 0, buf->data(), cs.nbytes);
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

void WebGPUGraph::add_qkv_fused_dispatch(QkvFusionGroup& g) {
  const uint32_t N = g.Nq + g.Nk + g.Nv; // fused output width (3072)

  const auto& in = tensors_[g.input_id];
  const auto& out_q = tensors_[g.out_q];
  const auto& out_k = tensors_[g.out_k];
  const auto& out_v = tensors_[g.out_v];
  const auto& wq = tensors_[g.weight_q];
  const auto& wk = tensors_[g.weight_k];
  const auto& wv = tensors_[g.weight_v];
  const auto& sq = tensors_[g.scales_q];
  const auto& sk = tensors_[g.scales_k];
  const auto& sv = tensors_[g.scales_v];

  // Buffers were repointed to FRESH distinct slots at the first triple op (see
  // the build() op-walk), so out_q/k/v no longer alias. Live M from the shared
  // input.
  uint64_t in_numel = 1;
  for (int64_t d : in.dims) {
    in_numel *= static_cast<uint64_t>(d);
  }
  const uint32_t M = static_cast<uint32_t>(in_numel / g.K);

  // Fused weight [N, K_packed]: a byte-contiguous row-stack of Wq;Wk;Wv (q4gsw
  // packs each output row independently along a shared K_packed, so stacking
  // along N is a flat append -- bit-exact). Fused scales [num_groups, N]: a
  // strided PER-GROUP-ROW gather (dest row stride N != the per-linear source
  // strides padded_N_{q,k,v}), NOT a flat append. Both dtor-freed via scratch.
  const uint64_t kp = static_cast<uint64_t>(g.K_packed); // packed bytes / row
  const uint64_t fs = sizeof(float);
  WGPUBuffer fused_weight = create_scratch_buffer(static_cast<size_t>(N) * kp);
  WGPUBuffer fused_scales = create_scratch_buffer(
      static_cast<size_t>(g.num_groups) * N * sizeof(float));

  // Sources are direct constants materialized in Phase 1 (or prepack outputs
  // materialized earlier in Phase 3); all writes are already enqueued on
  // queue_, so this build-time copy sees the materialized bytes.
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device_, nullptr);
  wgpuCommandEncoderCopyBufferToBuffer(
      enc, wq.buffer, 0, fused_weight, 0, static_cast<uint64_t>(g.Nq) * kp);
  wgpuCommandEncoderCopyBufferToBuffer(
      enc,
      wk.buffer,
      0,
      fused_weight,
      static_cast<uint64_t>(g.Nq) * kp,
      static_cast<uint64_t>(g.Nk) * kp);
  wgpuCommandEncoderCopyBufferToBuffer(
      enc,
      wv.buffer,
      0,
      fused_weight,
      static_cast<uint64_t>(g.Nq + g.Nk) * kp,
      static_cast<uint64_t>(g.Nv) * kp);
  for (uint32_t grp = 0; grp < g.num_groups; grp++) {
    const uint64_t dst_row = static_cast<uint64_t>(grp) * N * fs;
    wgpuCommandEncoderCopyBufferToBuffer(
        enc,
        sq.buffer,
        static_cast<uint64_t>(grp) * g.padded_N_q * fs,
        fused_scales,
        dst_row,
        static_cast<uint64_t>(g.Nq) * fs);
    wgpuCommandEncoderCopyBufferToBuffer(
        enc,
        sk.buffer,
        static_cast<uint64_t>(grp) * g.padded_N_k * fs,
        fused_scales,
        dst_row + static_cast<uint64_t>(g.Nq) * fs,
        static_cast<uint64_t>(g.Nk) * fs);
    wgpuCommandEncoderCopyBufferToBuffer(
        enc,
        sv.buffer,
        static_cast<uint64_t>(grp) * g.padded_N_v * fs,
        fused_scales,
        dst_row + static_cast<uint64_t>(g.Nq + g.Nk) * fs,
        static_cast<uint64_t>(g.Nv) * fs);
  }
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
  wgpuQueueSubmit(queue_, 1, &cmd);
  wgpuCommandBufferRelease(cmd);
  wgpuCommandEncoderRelease(enc);

  // Params UBO (owned; rewritten by the resize hook). padded_N == N (fused
  // scales row stride); has_bias == 0 (attention q/k/v are bias-less).
  QkvFusedParams params = {};
  params.M = M;
  params.N = N;
  params.K = g.K;
  params.K_packed = g.K_packed;
  params.group_size = g.group_size;
  params.padded_N = N;
  params.has_bias = 0;
  WGPUBufferDescriptor u_desc = {};
  u_desc.size = sizeof(QkvFusedParams);
  u_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  u_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device_, &u_desc);
  std::memcpy(
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(QkvFusedParams)),
      &params,
      sizeof(QkvFusedParams));
  wgpuBufferUnmap(uniform_buffer);
  add_uniform_buffer_bytes(sizeof(QkvFusedParams));

  // 4-byte dummy for the fixed bias binding (has_bias == 0).
  WGPUBuffer bias_dummy = create_scratch_buffer(4);

  // Bespoke 8-binding layout: 3 rw-storage outputs + 4 ro-storage + 1 uniform.
  // One-off shader/bgl/pipeline owned by the dispatch (matches
  // q4gsw_linear_impl).
  WGPUBindGroupLayoutEntry entries[8] = {};
  for (uint32_t i = 0; i < 3; i++) {
    entries[i].binding = i;
    entries[i].visibility = WGPUShaderStage_Compute;
    entries[i].buffer.type = WGPUBufferBindingType_Storage;
  }
  for (uint32_t i = 3; i < 7; i++) {
    entries[i].binding = i;
    entries[i].visibility = WGPUShaderStage_Compute;
    entries[i].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  }
  entries[7].binding = 7;
  entries[7].visibility = WGPUShaderStage_Compute;
  entries[7].buffer.type = WGPUBufferBindingType_Uniform;
  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 8;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device_, &bgl_desc);

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kQ4gswLinearGemmQkvFusedWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device_, &shader_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(device_, &pl_desc);

  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device_, &pipeline_desc);

  WGPUBindGroupEntry bg[8] = {};
  bg[0].binding = 0;
  bg[0].buffer = out_q.buffer;
  bg[0].size = out_q.nbytes;
  bg[1].binding = 1;
  bg[1].buffer = out_k.buffer;
  bg[1].size = out_k.nbytes;
  bg[2].binding = 2;
  bg[2].buffer = out_v.buffer;
  bg[2].size = out_v.nbytes;
  bg[3].binding = 3;
  bg[3].buffer = in.buffer;
  bg[3].size = in.nbytes;
  bg[4].binding = 4;
  bg[4].buffer = fused_weight;
  bg[4].size = static_cast<uint64_t>(N) * kp;
  bg[5].binding = 5;
  bg[5].buffer = fused_scales;
  bg[5].size = static_cast<uint64_t>(g.num_groups) * N * fs;
  bg[6].binding = 6;
  bg[6].buffer = bias_dummy;
  bg[6].size = 4;
  bg[7].binding = 7;
  bg[7].buffer = uniform_buffer;
  bg[7].size = sizeof(QkvFusedParams);
  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 8;
  bg_desc.entries = bg;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device_, &bg_desc);

  // 1D dispatch over ceil(M/BM) * ceil(N/BN) tiles (BM=BN=64), matching the
  // kernel's nbN = ceil(N/64) tile decode (NOT grid-strided).
  const uint32_t nbN = (N + 63u) / 64u;
  const uint32_t nbM = (M + 63u) / 64u;
  const size_t fused_idx =
      add_dispatch({pipeline, bind_group, nbN * nbM, "linear_q4gsw_qkv_fused"});
  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  own_uniform_buffer(uniform_buffer);
  g.fused_dispatch = fused_idx; // consumed by add_qkv_fused_hook at the last op
  g.fused_params = uniform_buffer;
}

// M-gate coordinator: registered at the LAST triple op (all dispatch indices
// known). Prefill (M>1): run the fused GEMM, zero the 3 separate linears.
// Decode (M==1): zero the fused, leave the 3 coop4 GEMVs (their own hooks set
// the decode wg) -- the fused 64x64 tile wastes 63/64 rows at M=1. Recomputes
// live M + the 3 output cur_dims + fused params. Inert on a static graph; a
// workgroup_count of 0 = no-op.
void WebGPUGraph::add_qkv_fused_hook(const QkvFusionGroup& g) {
  const int input_id = g.input_id, out_q_id = g.out_q, out_k_id = g.out_k,
            out_v_id = g.out_v;
  const uint32_t K = g.K, Kp = g.K_packed, gs = g.group_size, Nq = g.Nq,
                 Nk = g.Nk, Nv = g.Nv, Nf = g.Nq + g.Nk + g.Nv;
  const size_t fused_idx = g.fused_dispatch, sep0 = g.sep_dispatch[0],
               sep1 = g.sep_dispatch[1], sep2 = g.sep_dispatch[2];
  WGPUBuffer params_buf = g.fused_params;
  add_tensor_resize_hook(
      input_id,
      [input_id,
       out_q_id,
       out_k_id,
       out_v_id,
       K,
       Kp,
       gs,
       Nq,
       Nk,
       Nv,
       Nf,
       fused_idx,
       sep0,
       sep1,
       sep2,
       params_buf](WebGPUGraph& gr) {
        const auto& d = gr.cur_dims(input_id);
        uint64_t numel = 1;
        for (int64_t v : d) {
          numel *= static_cast<uint64_t>(v);
        }
        const uint32_t m = static_cast<uint32_t>(numel / K);
        std::vector<int64_t> oq = d;
        oq.back() = static_cast<int64_t>(Nq);
        std::vector<int64_t> ok = d;
        ok.back() = static_cast<int64_t>(Nk);
        std::vector<int64_t> ov = d;
        ov.back() = static_cast<int64_t>(Nv);
        gr.set_cur_dims(out_q_id, oq);
        gr.set_cur_dims(out_k_id, ok);
        gr.set_cur_dims(out_v_id, ov);
        QkvFusedParams p = {};
        p.M = m;
        p.N = Nf;
        p.K = K;
        p.K_packed = Kp;
        p.group_size = gs;
        p.padded_N = Nf;
        p.has_bias = 0;
        wgpuQueueWriteBuffer(gr.queue(), params_buf, 0, &p, sizeof(p));
        if (m > 1u) {
          const uint32_t nbN2 = (Nf + 63u) / 64u;
          const uint32_t nbM2 = (m + 63u) / 64u;
          gr.dispatch_at(fused_idx).workgroup_count_x = nbN2 * nbM2;
          gr.dispatch_at(sep0).workgroup_count_x = 0u;
          gr.dispatch_at(sep1).workgroup_count_x = 0u;
          gr.dispatch_at(sep2).workgroup_count_x = 0u;
        } else {
          gr.dispatch_at(fused_idx).workgroup_count_x = 0u;
        }
      });
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

    // Fast path: host and GPU element types match byte-for-byte.
    if (in.nbytes == live_nbytes) {
      wgpuQueueWriteBuffer(queue_, tensor.buffer, 0, in.data, live_nbytes);
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
      wgpuQueueWriteBuffer(
          queue_, tensor.buffer, 0, narrowed.data(), live_nbytes);
      continue;
    }

    throw std::runtime_error(
        "WebGPU: unsupported input copy for input " + std::to_string(i) +
        " (host " + std::to_string(in.nbytes) + " bytes" +
        (in.host_is_int64 ? " int64" : "") + " vs buffer " +
        std::to_string(live_nbytes) + " bytes)");
  }
}

namespace {
// Bench gate: compiled out unless WGPU_BACKEND_ENABLE_PROFILING; then the
// WEBGPU_TIMESTAMP_QUERY env var enables per-pass GPU timestamp queries.
bool should_timestamp_query() {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  static const bool enabled = std::getenv("WEBGPU_TIMESTAMP_QUERY") != nullptr;
  return enabled;
#else
  return false;
#endif
}
} // namespace

void WebGPUGraph::execute() {
  const size_t n = dispatches_.size();
  const size_t chunk = execute_config_.chunk_size;

  if (chunk == 0 || n <= chunk) {
#ifdef WGPU_BACKEND_ENABLE_PROFILING
    // Bench: timestamp-query pool, null unless env-gated + feature present.
    WebGPUQueryPool* qp = nullptr;
    if (should_timestamp_query() && n > 0) {
      if (auto* ctx = get_default_webgpu_context()) {
        if (ctx->timestamp_supported) {
          if (!ctx->querypool || ctx->querypool->capacity() < n) {
            ctx->querypool = std::make_unique<WebGPUQueryPool>();
            ctx->querypool->initialize(device_, static_cast<uint32_t>(n));
          }
          qp = ctx->querypool.get();
          qp->reset(static_cast<uint32_t>(n));
        }
      }
    }
#endif // WGPU_BACKEND_ENABLE_PROFILING

    WGPUCommandEncoderDescriptor enc_desc = {};
    WGPUCommandEncoder encoder =
        wgpuDeviceCreateCommandEncoder(device_, &enc_desc);

    // One pass per dispatch: enforces storage RAW ordering across deps.
    for (size_t i = 0; i < n; i++) {
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
      WGPUComputePassDescriptor pass_desc = {};
#ifdef WGPU_BACKEND_ENABLE_PROFILING
      // tw must outlive BeginComputePass (the descriptor points at it).
      WGPUPassTimestampWrites tw = {};
      if (qp) {
        tw = qp->writes_for(static_cast<uint32_t>(i));
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
            static_cast<uint32_t>(i),
            dispatch.kernel_name,
            {dispatch.workgroup_count_x, dispatch.workgroup_count_y, 1},
            {1, 1, 1});
      }
#endif // WGPU_BACKEND_ENABLE_PROFILING
    }

    for (const auto& copy : output_copies_) {
      wgpuCommandEncoderCopyBufferToBuffer(
          encoder, copy.src_buffer, 0, copy.staging_buffer, 0, copy.nbytes);
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
    return;
  }

  // GPU timestamp queries assume one submit; chunked execute is multi-submit.
  if (should_timestamp_query()) {
    throw std::runtime_error(
        "WebGPU: WEBGPU_TIMESTAMP_QUERY is incompatible with chunked execute "
        "(multi-submit); disable chunking to use GPU timestamp queries");
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
  WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error;
};

void buffer_map_callback(
    WGPUMapAsyncStatus status,
    WGPUStringView /*message*/,
    void* userdata1,
    void* /*userdata2*/) {
  auto* data = static_cast<MapCallbackData*>(userdata1);
  data->status = status;
}

} // namespace

void WebGPUGraph::copy_outputs(std::vector<std::pair<void*, size_t>>& outputs) {
  const size_t count = std::min(outputs.size(), output_staging_buffers_.size());

  std::vector<MapCallbackData> cb_data(count);
  std::vector<WGPUFuture> map_futures(count, WGPUFuture{});

  for (size_t i = 0; i < count; i++) {
    if (outputs[i].second == 0) {
      cb_data[i].status = WGPUMapAsyncStatus_Success;
      continue;
    }
    WGPUBufferMapCallbackInfo cb_info = {};
    cb_info.mode = WGPUCallbackMode_WaitAnyOnly;
    cb_info.callback = buffer_map_callback;
    cb_info.userdata1 = &cb_data[i];
    map_futures[i] = wgpuBufferMapAsync(
        output_staging_buffers_[i],
        WGPUMapMode_Read,
        0,
        outputs[i].second,
        cb_info);
  }

  for (size_t i = 0; i < count; i++) {
    if (outputs[i].second != 0 &&
        webgpu_wait(instance_, map_futures[i]) != WGPUWaitStatus_Success) {
      throw std::runtime_error("WebGPU: WaitAny failed for output map");
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
