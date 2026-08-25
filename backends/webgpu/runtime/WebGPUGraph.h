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
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <executorch/backends/webgpu/runtime/WebGPUDispatchMath.h>
#include <executorch/backends/webgpu/runtime/WebGPUExecutionOptions.h>
#include <executorch/runtime/core/named_data_map.h>

namespace executorch::backends::webgpu {

struct WebGPUTensor {
  WGPUBuffer buffer = nullptr;
  // Max (allocation) dims/nbytes: the serialized upper-bound shape. The GPU
  // buffer is sized from these and never reallocated (Vulkan allocate-at-max).
  std::vector<int64_t> dims;
  size_t nbytes = 0;
  // Live dims/nbytes for dynamic shapes; always <= the max. == max on a static
  // graph, so dynamic-resize logic keyed off these is inert there.
  std::vector<int64_t> cur_dims;
  size_t cur_nbytes = 0;
  // Serialized (GPU-side) element type, used to narrow wider host inputs.
  size_t elem_size = 0;
  bool is_int = false;
  // Exact BOOL tag for byte-packed WGSL storage.
  bool is_bool = false;
  // Exactly int8 (not uint8/bool), so int8-only ops can guard their dtype.
  bool is_int8 = false;
};

// Host-side view of one graph input, passed to copy_inputs.
struct InputData {
  const void* data = nullptr;
  size_t nbytes = 0;
  bool host_is_int64 = false;
  bool host_is_fp32 = false;
};

// Host destination for a graph output. host_is_fp32 gates the fp16->fp32 widen
// on readback (mirrors InputData's guard on the copy_inputs narrow path).
struct OutputData {
  void* data = nullptr;
  size_t nbytes = 0;
  bool host_is_fp32 = false;
};

struct WebGPUDispatch {
  WGPUComputePipeline pipeline = nullptr;
  WGPUBindGroup bind_group = nullptr;
  uint32_t workgroup_count_x = 1;
  std::string kernel_name; // bench label
  uint32_t workgroup_count_y = 1; // 2D fold (>65535); 1 = unchanged 1D path
  // DMA copy command; default Compute keeps existing positional inits valid.
  enum class Kind { Compute, Copy };
  Kind kind = Kind::Compute;
  WGPUBuffer copy_src = nullptr;
  WGPUBuffer copy_dst = nullptr;
  size_t copy_nbytes = 0;
};

struct WebGPUBufferBinding {
  WGPUBuffer buffer = nullptr;
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct WebGPUSpecializationConstant {
  std::string name;
  double value = 0.0;
};

struct WebGPUDispatchGrid {
  uint32_t x = 1;
  uint32_t y = 1;
};

struct WebGPUComputeDispatchDescriptor {
  std::string shader_name;
  std::string entry_point = "main";
  std::string kernel_name;
  std::vector<WebGPUBufferBinding> bindings;
  std::vector<WebGPUSpecializationConstant> constants;
  WebGPUDispatchGrid grid;
};

std::string make_compute_pipeline_key(
    const WebGPUComputeDispatchDescriptor& descriptor);

void validate_compute_dispatch_descriptor(
    const WebGPUComputeDispatchDescriptor& descriptor);

struct OutputCopy {
  WGPUBuffer src_buffer = nullptr;
  WGPUBuffer staging_buffer = nullptr;
  size_t nbytes = 0;
};

// CPU-side record for a prepack-routed constant; mirrors Vulkan's TensorRef
// (sizes + a data reference, not a live GPU tensor). The prepack node is the
// sole materialization, so the constant needs no eager GPU buffer.
struct ConstantSource {
  uint64_t inline_offset = UINT64_MAX; // offset into constant_data_; else key
  std::string named_key; // non-empty => fetch from named_data_map_
  size_t nbytes = 0;
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
  int num_cached_pipelines = 0;
  int num_cached_shaders = 0;

  size_t total_bytes() const {
    return tensor_buffer_bytes + staging_buffer_bytes + uniform_buffer_bytes;
  }
};

struct WebGPUGraphConfig {
  bool f16_kv_cache = false;
  bool f16_accumulate_gemm = false;
  int sdpa_query_tile = 0;
  bool record_q4gsw_decode_route = false;
};

class WebGPUGraph {
 public:
  WebGPUGraph();
  ~WebGPUGraph();

  // Build the graph from a deserialized VkGraph flatbuffer and constant data.
  // The flatbuffer_data pointer must remain valid during build().
  void build(
      const void* flatbuffer_data,
      const uint8_t* constant_data,
      size_t constant_data_size,
      const executorch::runtime::NamedDataMap* named_data_map = nullptr,
      WebGPUGraphConfig config = {});

  // Copy input tensor data from host pointers into GPU buffers.
  void copy_inputs(const std::vector<InputData>& inputs);

  WebGPUExecutionPlan make_execution_plan(
      const WebGPUGraphExecutionOptions& options) const;

  // Execute the dispatches selected by a plan created for this graph. Returns
  // the number of GPU queue submissions performed.
  size_t execute(const WebGPUExecutionPlan& plan);

  // Copy output tensor data from GPU buffers back to host pointers.
  // Uses mapAsync + ASYNCIFY in Wasm.
  void copy_outputs(
      std::vector<OutputData>& outputs,
      const WebGPUExecutionPlan& plan);

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
  // Int values of a serialized IntList (e.g. permute dims). int64 (FlatBuffer
  // [long]) to match the schema and the get_int convention.
  const std::vector<int64_t>& get_int_list(int id) const {
    return int_lists_[id];
  }
  // Member value ids of a serialized ValueList (op multi-output list).
  const std::vector<int>& get_value_list(int id) const {
    return value_lists_[id];
  }
  bool get_bool(int id) const {
    return bools_[id];
  }
  // String value (e.g. gelu's `approximate` kwarg).
  const std::string& get_string(int id) const {
    return strings_[id];
  }

  // Live-scalar (SymInt) API; mirrors the Vulkan SymInt/ParamsBuffer UBO.
  // set_symint writes the buffer + marks dirty only if the value changed.
  void set_symint(int id, int32_t val);
  // read_symint throws (fail-loud) if id is not a SymInt.
  int32_t read_symint(int id) const {
    return symints_.at(id).value;
  }
  // symint_buffer throws (fail-loud) if id is not a SymInt.
  WGPUBuffer symint_buffer(int id) const {
    return symints_.at(id).buffer;
  }

  // Records that a SymInt's value is read from input_tensor[index] along dim.
  struct SymIntSource {
    int symint_id;
    int input_tensor_id;
    int dim;
    int index;
  };
  void
  add_symint_source(int symint_id, int input_tensor_id, int dim, int index) {
    symint_sources_.push_back({symint_id, input_tensor_id, dim, index});
  }
  const std::vector<SymIntSource>& symint_sources() const {
    return symint_sources_;
  }

  // Records that a SymInt is a tensor's live dim size (sym_size.int), read from
  // cur_dims at execute; distinct from SymIntSource (a scalar data element).
  struct SymIntDimSource {
    int symint_id;
    int tensor_id;
    int dim;
  };
  void add_symint_dim_source(int symint_id, int tensor_id, int dim) {
    symint_dim_sources_.push_back({symint_id, tensor_id, dim});
  }

  bool tensor_has_dynamic_dims(int tensor_id) const {
    return dynamic_tensor_ids_.count(tensor_id) != 0;
  }

  bool has_dynamic_shapes() const {
    return !dynamic_tensor_ids_.empty();
  }

  // Execute-time select_as_symint read; mirrors Vulkan select_as_symint_impl.
  void update_symints_from_inputs(const std::vector<InputData>& inputs);

  // Per-SymInt resize hook; mirrors Vulkan DynamicDispatchNode::trigger_resize.
  void add_resize_hook(int symint_id, std::function<void(WebGPUGraph&)> fn) {
    if (symint_id < 0 || symint_id >= num_values() ||
        get_value_type(symint_id) != ValueType::SymInt) {
      throw std::runtime_error("WebGPU resize: trigger must be a SymInt");
    }
    if (!fn) {
      throw std::runtime_error("WebGPU resize: null SymInt resize hook");
    }
    resize_hooks_.push_back({symint_id, std::move(fn)});
  }

  template <typename Context>
  void add_resize_hook(
      int symint_id,
      void (*fn)(WebGPUGraph&, const Context&),
      Context context) {
    if (fn == nullptr) {
      throw std::runtime_error("WebGPU resize: null SymInt resize hook");
    }
    add_resize_hook(
        symint_id, [fn, context = std::move(context)](WebGPUGraph& graph) {
          fn(graph, context);
        });
  }

  // Set a graph input's live dims (<= max) + dirty it; static path stays inert.
  void resize_input(int value_id, const std::vector<int64_t>& new_dims);
  // Set a tensor's live dims (an op resize hook calls this for its output to
  // cascade to consumers); validates the new dims fit the max, never reallocs.
  void set_cur_dims(int value_id, const std::vector<int64_t>& new_dims);
  const std::vector<int64_t>& cur_dims(int value_id) const {
    return tensors_[value_id].cur_dims;
  }

  // Per-tensor resize hook; mirrors Vulkan ExecuteNode::resize_fn. Runs in
  // propagate_resize when trigger_tensor_id is dirty.
  void add_tensor_resize_hook(
      int trigger_tensor_id,
      std::function<void(WebGPUGraph&)> fn) {
    if (trigger_tensor_id < 0 || trigger_tensor_id >= num_values() ||
        get_value_type(trigger_tensor_id) != ValueType::Tensor) {
      throw std::runtime_error("WebGPU resize: trigger must be a Tensor");
    }
    if (!fn) {
      throw std::runtime_error("WebGPU resize: null tensor resize hook");
    }
    tensor_resize_hooks_.push_back({trigger_tensor_id, std::move(fn)});
  }

  template <typename Context>
  void add_tensor_resize_hook(
      int trigger_tensor_id,
      void (*fn)(WebGPUGraph&, const Context&),
      Context context) {
    if (fn == nullptr) {
      throw std::runtime_error("WebGPU resize: null tensor resize hook");
    }
    add_tensor_resize_hook(
        trigger_tensor_id,
        [fn, context = std::move(context)](WebGPUGraph& graph) {
          fn(graph, context);
        });
  }

  // Run hooks for changed SymInts and tensors, then clear; call before execute.
  void propagate_resize();

  // Mutable dispatch access for resize hooks (to rewrite workgroup_count_x).
  WebGPUDispatch& dispatch_at(size_t i) {
    return dispatches_[i];
  }
  size_t num_dispatches() const {
    return dispatches_.size();
  }

  size_t register_dispatch_route_group(
      const std::vector<utils::DispatchRange>& ranges) {
    validate_dynamic_dispatch_route_ranges(ranges);
    return dispatch_routes_.register_group(
        dispatches_.size(), ranges, [&](size_t i) {
          return dispatches_[i].kind == WebGPUDispatch::Kind::Compute;
        });
  }

  void select_dispatch_route(
      size_t group,
      size_t active_route,
      const std::vector<utils::WgCount>& active_grids) {
    dispatch_routes_.select(
        group, active_route, active_grids, [&](size_t i, utils::WgCount grid) {
          dispatches_[i].workgroup_count_x = grid.x;
          dispatches_[i].workgroup_count_y = grid.y;
        });
  }

  WGPUDevice device() const {
    return device_;
  }
  WGPUQueue queue() const {
    return queue_;
  }

  // Returns the new dispatch's index (resize hooks rewrite it via dispatch_at).
  size_t add_dispatch(WebGPUDispatch dispatch) {
    dispatches_.push_back(dispatch);
    return dispatches_.size() - 1;
  }

  // 2D sibling of add_dispatch (sets workgroup_count_y); returns the index.
  size_t add_dispatch_2d(
      WGPUComputePipeline pipeline,
      WGPUBindGroup bind_group,
      uint32_t count_x,
      uint32_t count_y) {
    WebGPUDispatch d;
    d.pipeline = pipeline;
    d.bind_group = bind_group;
    d.workgroup_count_x = count_x;
    d.workgroup_count_y = count_y;
    return add_dispatch(d);
  }

  // In-graph buffer-to-buffer DMA (e.g. flat copy); returns the dispatch index.
  size_t add_buffer_copy(WGPUBuffer src, WGPUBuffer dst, size_t nbytes) {
    WebGPUDispatch d;
    d.kind = WebGPUDispatch::Kind::Copy;
    d.copy_src = src;
    d.copy_dst = dst;
    d.copy_nbytes = nbytes;
    d.kernel_name = "flat_copy";
    dispatches_.push_back(d);
    return dispatches_.size() - 1;
  }

  // Materialize a recorded prepack-routed constant into dst via one CPU->GPU
  // transfer. Build-time only (the .pte bytes are freed after build()).
  // Mirrors Vulkan prepack_standard.
  void materialize_constant(int const_value_id, WGPUBuffer dst);

  void add_uniform_buffer_bytes(size_t bytes) {
    uniform_buffer_bytes_ += bytes;
  }

  // Keep a uniform alive for the graph's lifetime; released in the dtor.
  void own_uniform_buffer(WGPUBuffer buffer) {
    owned_uniform_buffers_.push_back(buffer);
  }

  template <typename Block>
  WGPUBuffer create_params_buffer(const Block& data) {
    static_assert(
        std::is_trivially_copyable<Block>::value,
        "WebGPU parameter blocks must be trivially copyable");
    static_assert(
        sizeof(Block) % 4u == 0u,
        "WebGPU parameter blocks must have a 4-byte-aligned size");
    WGPUBuffer buffer = make_uniform_buffer(&data, sizeof(Block));
    try {
      own_uniform_buffer(buffer);
    } catch (...) {
      wgpuBufferRelease(buffer);
      throw;
    }
    return buffer;
  }

  // Graph-owned scratch storage buffer for fused-op intermediates (e.g. SDPA).
  WGPUBuffer create_scratch_buffer(size_t nbytes);

  // Reusable scratch pool for SINGLE-OP-LIFETIME fused-op scratch (SDPA
  // attn_weights/softmax, FlashDecoding partials). acquire_scratch() reuses a
  // free slot (best-fit, size in [n,2n]) or creates one; the caller RELEASES it
  // at op-lowering scope exit (use ScopedScratch), so N layers' scratch reuses
  // a small constant of buffers instead of N x held to graph teardown.
  // Correctness: WebGPU/Dawn auto-inserts RAW hazard barriers between
  // dispatches on a shared storage buffer regardless of pass structure -- the
  // SAME guarantee mem_obj_id aliasing already relies on -- so reuse is
  // bit-identical. Never hand a still-in_use slot to a co-live requester.
  WGPUBuffer acquire_scratch(size_t nbytes);
  void release_scratch(WGPUBuffer buffer);
  // RAII: releases an acquired scratch slot when the op-lowering scope exits
  // (leak-safe vs early returns).
  struct ScopedScratch {
    WebGPUGraph* g = nullptr;
    WGPUBuffer buf = nullptr;
    ScopedScratch(WebGPUGraph* graph, WGPUBuffer b) : g(graph), buf(b) {}
    ~ScopedScratch() {
      if (g && buf) {
        g->release_scratch(buf);
      }
    }
    ScopedScratch(const ScopedScratch&) = delete;
    ScopedScratch& operator=(const ScopedScratch&) = delete;
    operator WGPUBuffer() const {
      return buf;
    }
  };

  // Create a mapped-at-creation uniform buffer from `size` bytes and track it
  // in the memory stats. Shared helper for ops needing a uniform Params buffer.
  WGPUBuffer make_uniform_buffer(const void* data, size_t size);

  size_t add_compute_dispatch(
      const WebGPUComputeDispatchDescriptor& descriptor);

  template <typename Context>
  size_t add_dynamic_compute_dispatch(
      const WebGPUComputeDispatchDescriptor& descriptor,
      int trigger_tensor_id,
      WebGPUDispatchGrid (*pick_grid)(const WebGPUGraph&, const Context&),
      Context context) {
    if (pick_grid == nullptr) {
      throw std::runtime_error("WebGPU dynamic dispatch: null grid picker");
    }
    return add_dynamic_compute_dispatch_impl(
        descriptor,
        trigger_tensor_id,
        [pick_grid, context = std::move(context)](const WebGPUGraph& graph) {
          return pick_grid(graph, context);
        });
  }

  WGPUShaderModule get_or_create_shader(
      const std::string& key,
      const char* wgsl_source);

  WGPUComputePipeline get_or_create_pipeline(
      const std::string& key,
      WGPUShaderModule shader,
      WGPUPipelineLayout layout);

  WGPUBindGroupLayout get_or_create_bgl(
      const std::string& key,
      const WGPUBindGroupLayoutEntry* entries,
      uint32_t count);

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

  enum class ValueType {
    Tensor,
    Int,
    Double,
    Bool,
    Null,
    String,
    SymInt,
    ValueList,
    IntList
  };

  ValueType get_value_type(int id) const {
    return value_types_[id];
  }

  // Memory-aliasing group id for the tensor's shared buffer, or -1 if it has
  // none; fusion passes use this to reject candidates aliased with something
  // the planner may reuse outside the fusion's control.
  int mem_obj_id(int id) const {
    return tensor_mem_obj_ids_[id];
  }

  // True if id is a prepack-routed constant with a recorded source (inline
  // offset or named-data-map key); fusion passes require direct constants.
  bool has_constant_source(int id) const {
    return constant_sources_.count(id) != 0;
  }

 public:
  // True when the sdpa K/V cache is stored f16-packed (runtime opt-in).
  bool kv_f16() const {
    return kv_f16_;
  }

  // True when the q4gsw steel prefill GEMM uses the lossy f16-accumulate kernel
  // (runtime opt-in; perplexity-gated, not bit-exact).
  bool f16_accumulate_gemm() const {
    return config_.f16_accumulate_gemm;
  }

  // Runtime-selected SDPA query-tile candidate; 0 = geometry default (Q16),
  // 32 = Q32 candidate.
  int sdpa_query_tile() const {
    return config_.sdpa_query_tile;
  }

  const WebGPUGraphConfig& config() const {
    return config_;
  }

 private:
#ifdef WGPU_BACKEND_ENABLE_PROFILING
  void record_active_route(const std::string& kernel_name);
#endif // WGPU_BACKEND_ENABLE_PROFILING

  bool kv_f16_ = false;
  std::unordered_set<int> kv_cache_ids_;
  WebGPUGraphConfig config_;

 private:
  WGPUInstance instance_ = nullptr;
  WGPUDevice device_ = nullptr;
  WGPUQueue queue_ = nullptr;

  // Flat arrays indexed by value ID. Only the relevant one is populated
  // per ID based on value_types_.
  std::vector<ValueType> value_types_;
  std::vector<WebGPUTensor> tensors_;
  std::vector<int64_t> ints_;
  std::vector<std::vector<int64_t>> int_lists_;
  std::vector<std::vector<int>> value_lists_;
  std::vector<double> doubles_;
  std::vector<bool> bools_;
  std::vector<std::string> strings_;

  // SymInt (live scalar): id -> {live Uniform buffer, current value}, sparse.
  struct SymIntSlot {
    WGPUBuffer buffer = nullptr;
    int32_t value = 0;
  };
  std::unordered_map<int, SymIntSlot> symints_;
  std::vector<SymIntSource> symint_sources_;
  std::vector<SymIntDimSource> symint_dim_sources_;
  std::unordered_set<int> dynamic_tensor_ids_;

  // Resize hooks + the set of SymInts changed since the last propagate_resize.
  struct ResizeHook {
    int symint_id;
    std::function<void(WebGPUGraph&)> fn;
  };
  std::vector<ResizeHook> resize_hooks_;
  std::unordered_set<int> dirty_symints_;

  // Tensor-shape resize hooks + the set of tensors changed since the last
  // propagate_resize (mirrors the SymInt pair above, for dynamic shapes).
  struct TensorResizeHook {
    int trigger_tensor_id;
    std::function<void(WebGPUGraph&)> fn;
  };
  std::vector<TensorResizeHook> tensor_resize_hooks_;
  std::unordered_set<int> dirty_tensors_;

  // Dynamic grids are stored separately so ordinary dispatches remain compact.
  // The graph owns each dispatch index and picker; ops only provide typed
  // context and a named grid function.
  struct DynamicDispatchGrid {
    size_t dispatch_index;
    int trigger_tensor_id;
    std::function<WebGPUDispatchGrid(const WebGPUGraph&)> pick_grid;
  };
  std::vector<DynamicDispatchGrid> dynamic_dispatch_grids_;

  struct PendingDynamicDispatchGrid {
    size_t dispatch_index;
    WebGPUDispatchGrid grid;
  };
  std::vector<PendingDynamicDispatchGrid> pending_dynamic_dispatch_grids_;

  size_t add_dynamic_compute_dispatch_impl(
      const WebGPUComputeDispatchDescriptor& descriptor,
      int trigger_tensor_id,
      std::function<WebGPUDispatchGrid(const WebGPUGraph&)> pick_grid);
  void validate_dynamic_dispatch_route_ranges(
      const std::vector<utils::DispatchRange>& ranges) const;

  std::vector<int> input_ids_;
  std::vector<int> output_ids_;

  // Memory aliasing: tensors with the same mem_obj_id share a WGPUBuffer.
  std::vector<int> tensor_mem_obj_ids_;
  std::vector<WGPUBuffer> shared_buffers_;
  std::vector<size_t> shared_buffer_sizes_;

  // Long-lived scratch storage buffers for fused ops (e.g. SDPA temporaries).
  std::vector<WGPUBuffer> scratch_buffers_;

  // Reusable scratch pool: single-op-lifetime buffers recycled across ops
  // (acquire_scratch/release_scratch). Each slot is freed in the dtor. See
  // acquire_scratch() for the reuse policy.
  struct ScratchSlot {
    WGPUBuffer buffer = nullptr;
    size_t size = 0;
    bool in_use = false;
  };
  std::vector<ScratchSlot> scratch_pool_;

  // Uniform buffers owned for the graph's lifetime; released in the dtor.
  std::vector<WGPUBuffer> owned_uniform_buffers_;

  // Staging buffers for reading back outputs (MapRead | CopyDst).
  std::vector<WGPUBuffer> output_staging_buffers_;

  // Pre-computed output copy descriptors for execute().
  std::vector<OutputCopy> output_copies_;

  std::vector<SuppressibleOutput> suppressible_outputs_;

  std::vector<WebGPUDispatch> dispatches_;
  utils::DispatchRouteRegistry dispatch_routes_;

  // Prepack-routed constant sources (offset/named-key + size); the prepack node
  // materializes these once. constant_data_/named_data_map_ point at the .pte
  // bytes and are valid only during build().
  const uint8_t* constant_data_ = nullptr;
  size_t constant_data_size_ = 0;
  const executorch::runtime::NamedDataMap* named_data_map_ = nullptr;
  std::unordered_map<int, ConstantSource> constant_sources_;

  ExecuteConfig execute_config_;

  // Caches for reusing GPU objects across dispatches.
  std::unordered_map<std::string, WGPUShaderModule> shader_cache_;
  std::unordered_map<std::string, WGPUComputePipeline> pipeline_cache_;
  std::unordered_map<std::string, WGPUBindGroupLayout> bgl_cache_;

  size_t uniform_buffer_bytes_ = 0;
};

#ifdef WGPU_BACKEND_ENABLE_PROFILING
extern uint32_t g_last_route_mask;
extern uint32_t g_last_route_conflict_count;
#endif // WGPU_BACKEND_ENABLE_PROFILING

} // namespace executorch::backends::webgpu
