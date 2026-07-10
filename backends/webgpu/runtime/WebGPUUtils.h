/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/webgpu/runtime/WebGPUDispatchMath.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>

#include <webgpu/webgpu.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu::utils {

// Product of dims (live element count); used by dynamic-resize hooks. Delegates
// to numel (single impl; keeps the negative-dim guard, no caller churn).
inline uint64_t numel_of(const std::vector<int64_t>& dims) {
  return numel(dims);
}

// Clamp workgroup size to device limit (SwiftShader caps at 128).
inline uint32_t clamp_workgroup_size(WGPUDevice device, uint32_t desired) {
  WGPULimits limits = {};
  if (wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
      limits.maxComputeInvocationsPerWorkgroup > 0) {
    return std::min(desired, limits.maxComputeInvocationsPerWorkgroup);
  }
  return desired;
}

struct WgCount {
  uint32_t x;
  uint32_t y;
};

// Device's max workgroups per dispatch dimension; the WebGPU spec-default floor
// (65535) if the query fails — never under-reports a real device's capacity.
inline uint32_t queried_max_workgroups(WGPUDevice device) {
  WGPULimits limits = {};
  return wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
          limits.maxComputeWorkgroupsPerDimension > 0
      ? limits.maxComputeWorkgroupsPerDimension
      : 65535u;
}

// Pure 2D fold of a 1D workgroup count (device-free, unit-testable): {count,1}
// when count <= max, else a near-square {x, y} with x ~ ceil(sqrt(count)) so
// the launched grid stays close to count. A flat {max, div_up(count, max)}
// split would leave up to ~half the workgroups inactive when count just exceeds
// max, and inactive workgroups still cost launch/scheduling; the near-square
// split keeps the waste to O(sqrt(count)). Throws if even a max*max grid is too
// small (a 3rd dispatch dimension, out of scope). The shader reconstructs the
// linear index from @builtin(num_workgroups), so any x/y factoring works.
// Now delegates to the DispatchMath fold (single fold impl); wg_size=1 makes
// the returned grid's stride_x collapse harmlessly.
inline WgCount fold_workgroup_count_2d(
    uint32_t count,
    uint32_t max_count,
    const char* op_name) {
  DispatchGrid g = compute_dispatch_grid_from_limits(
      count, /*wg_size=*/1, max_count, op_name);
  return {g.count_x, g.count_y};
}

// 1D dispatch count (mirrors Vulkan div_up); throws if > device limit.
inline uint32_t compute_1d_workgroup_count(
    WGPUDevice device,
    uint32_t num_threads,
    uint32_t workgroup_size,
    const char* op_name) {
  uint32_t count = div_up(num_threads, workgroup_size);
  if (count > queried_max_workgroups(device)) {
    throw std::runtime_error(
        std::string("WebGPU ") + op_name +
        ": workgroup count exceeds the 1D dispatch limit");
  }
  return count;
}

// 2D dispatch count: fold the 1D count across x/y when it exceeds the per-dim
// limit (lifts the cap, e.g. for SDPA prefill). Same fast path as compute_1d.
inline WgCount compute_2d_workgroup_count(
    WGPUDevice device,
    uint32_t num_threads,
    uint32_t workgroup_size,
    const char* op_name) {
  uint32_t count = div_up(num_threads, workgroup_size);
  return fold_workgroup_count_2d(
      count, queried_max_workgroups(device), op_name);
}

inline DispatchGrid compute_dispatch_grid(
    WGPUDevice device,
    uint32_t num_threads,
    uint32_t desired_wg,
    const char* op_name) {
  // Single limits query shared by wg-size clamp + max-dim (avoid 2 queries).
  WGPULimits limits = {};
  bool got_limits = wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success;
  uint32_t wg = (got_limits && limits.maxComputeInvocationsPerWorkgroup > 0)
      ? std::min(desired_wg, limits.maxComputeInvocationsPerWorkgroup)
      : desired_wg;
  if (wg == 0) {
    wg = 1;
  }
  uint32_t max_dim = (got_limits && limits.maxComputeWorkgroupsPerDimension > 0)
      ? limits.maxComputeWorkgroupsPerDimension
      : 65535u; // WebGPU spec-default floor
  uint32_t total = div_up(num_threads, wg); // workgroups needed (1D)
  return compute_dispatch_grid_from_limits(total, wg, max_dim, op_name);
}

// 2D tile grid for tiled kernels: {div_up(n, tile), div_up(m, tile)} workgroups
// (one per tile); throws if either dim exceeds the device's per-dim limit.
inline WgCount compute_tile_grid_2d(
    WGPUDevice device,
    uint32_t n,
    uint32_t m,
    uint32_t tile,
    const char* op_name) {
  uint32_t max_wgs = queried_max_workgroups(device);
  uint32_t x = div_up(n, tile);
  uint32_t y = div_up(m, tile);
  if (x > max_wgs || y > max_wgs) {
    throw std::runtime_error(
        std::string("WebGPU ") + op_name +
        ": tile grid exceeds dispatch limit");
  }
  return {x, y};
}

// For "one workgroup per row" kernels (native_layer_norm, sdpa_softmax) where
// num_rows IS the workgroup count already, unlike compute_dispatch_grid's
// element-count-needing-division use case. wg_size=1 makes the returned
// grid's stride_x collapse to exactly count_x, giving a correct flat
// *workgroup*-index decode (row_idx = workgroup_id.y * stride_x +
// workgroup_id.x).
inline DispatchGrid compute_row_dispatch_grid(
    WGPUDevice device,
    uint32_t num_rows,
    const char* op_name) {
  return compute_dispatch_grid_from_limits(
      num_rows, /*wg_size=*/1, queried_max_workgroups(device), op_name);
}

// Create a uniform buffer mapped-at-creation, copy `size` bytes in, and unmap.
inline WGPUBuffer
make_uniform(WGPUDevice device, const void* data, size_t size) {
  WGPUBufferDescriptor desc = {};
  desc.size = size;
  desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  desc.mappedAtCreation = true;
  WGPUBuffer buf = wgpuDeviceCreateBuffer(device, &desc);
  if (!buf) {
    throw std::runtime_error("make_uniform: buffer creation failed");
  }
  void* ptr = wgpuBufferGetMappedRange(buf, 0, size);
  if (!ptr) {
    wgpuBufferRelease(buf);
    throw std::runtime_error("make_uniform: mapped range is null");
  }
  std::memcpy(ptr, data, size);
  wgpuBufferUnmap(buf);
  return buf;
}

// Clamp a 1D workgroup count to the device limit, for grid-stride kernels that
// loop over any excess work (vs compute_1d_workgroup_count, which throws).
inline uint32_t clamp_workgroup_count(WGPUDevice device, uint32_t desired) {
  return std::min(desired, queried_max_workgroups(device));
}

// Zero-filled storage buffer; used as a dummy binding for an optional tensor
// arg (bias/mask/affine) the shader gates off and never reads.
inline WGPUBuffer make_storage_zeros(WGPUDevice device, size_t size) {
  WGPUBufferDescriptor desc = {};
  desc.size = size;
  desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
  desc.mappedAtCreation = true;
  WGPUBuffer buf = wgpuDeviceCreateBuffer(device, &desc);
  if (!buf) {
    throw std::runtime_error("make_storage_zeros: buffer creation failed");
  }
  void* ptr = wgpuBufferGetMappedRange(buf, 0, size);
  if (!ptr) {
    wgpuBufferRelease(buf);
    throw std::runtime_error("make_storage_zeros: mapped range is null");
  }
  std::memset(ptr, 0, size);
  wgpuBufferUnmap(buf);
  return buf;
}

// Validates a dim is a multiple of 4 (vec4-alignment precondition for a
// vec4-typed buffer view); throws loud, mirroring this backend's no-silent-
// return convention. Was independently duplicated in sdpa/Sdpa.cpp,
// sdpa_fd_decode/SdpaFdDecode.cpp, and et_vk_sdpa/EtVkSdpa.cpp.
inline void
check_vec4_aligned(uint32_t dim, const char* op_name, const char* dim_name) {
  if (dim % 4 != 0) {
    throw std::runtime_error(
        std::string("WebGPU ") + op_name + ": " + dim_name +
        " must be a multiple of 4");
  }
}

// fp32 byte-size guard (no runtime dtype): the serialized bytes must equal the
// element count times sizeof(float), else the tensor is not fp32. Returns the
// element count; replaces the `numel(dims)*sizeof(float) != nbytes` check
// duplicated across the fp32-only op handlers.
inline uint64_t
check_fp32(const WebGPUTensor& t, const char* op_name, const char* label) {
  uint64_t n = numel(t.dims);
  if (t.nbytes != n * sizeof(float)) {
    throw std::runtime_error(
        std::string("WebGPU ") + op_name + ": " + label + " must be fp32");
  }
  return n;
}

// Elementwise unary I/O guard: both buffers bound, byte counts fp32-aligned,
// and input/output sizes equal. Mirrors the guard inlined in every elementwise
// op handler (gelu/sigmoid/...).
inline void check_elementwise_fp32_io(
    const WebGPUTensor& in,
    const WebGPUTensor& out,
    const char* op_name) {
  if (in.buffer == nullptr || out.buffer == nullptr) {
    throw std::runtime_error(std::string(op_name) + ": null buffer binding");
  }
  if (in.nbytes % sizeof(float) != 0 || out.nbytes % sizeof(float) != 0) {
    throw std::runtime_error(
        std::string(op_name) + ": operand not 4-byte aligned");
  }
  if (in.nbytes != out.nbytes) {
    throw std::runtime_error(
        std::string(op_name) + ": input/output size mismatch");
  }
}

// Narrow a u64 element/byte count to u32 for a dispatch/param field; throws if
// it would truncate (the count exceeds the u32 addressing range).
inline uint32_t checked_u32(uint64_t v, const char* op_name) {
  if (v > 0xFFFFFFFFull) {
    throw std::runtime_error(
        std::string("WebGPU ") + op_name + ": output too large");
  }
  return static_cast<uint32_t>(v);
}

// Extract a Scalar arg as float: Double -> cast, Int -> cast, Null -> fallback.
// Any other type throws (this backend's no-silent-return convention: an
// unexpected scalar type is a validation failure, not a "use the default").
// Scope: the 3 already-identical Double/Int/Null-permissive copies only
// (addmm's beta/alpha, constant_pad_nd's value, native_layer_norm's epsilon)
// -- NOT et_vk.sdpa/sdpa's scale extraction, which is a stricter
// Double-or-Null-only policy (rejects Int) that would silently change
// behavior if forced into this signature.
inline float scalar_or(WebGPUGraph& graph, int id, float fallback) {
  const auto vt = graph.get_value_type(id);
  if (vt == WebGPUGraph::ValueType::Double) {
    return static_cast<float>(graph.get_double(id));
  }
  if (vt == WebGPUGraph::ValueType::Int) {
    return static_cast<float>(graph.get_int(id));
  }
  if (vt == WebGPUGraph::ValueType::Null) {
    return fallback;
  }
  throw std::runtime_error("scalar_or: expected Double, Int, or None");
}

// Resolves an optional tensor arg (bias/mask/affine) to the binding to use:
// the real tensor's buffer+size if present, else a zero-filled dummy. Caller
// releases `.owned_dummy` (if non-null) after the dispatch is recorded.
struct OptionalBinding {
  WGPUBuffer buffer;
  uint64_t nbytes;
  WGPUBuffer owned_dummy; // non-null only when a dummy was allocated
};

inline OptionalBinding make_optional_binding(
    WGPUDevice device,
    bool present,
    WGPUBuffer real_buffer,
    uint64_t real_nbytes) {
  if (present) {
    return {real_buffer, real_nbytes, nullptr};
  }
  // 16 bytes (not 4): WebGPU validates a binding's size against the shader's
  // DECLARED type at dispatch time regardless of which branch runs, and a
  // vec4<f32>-typed binding requires >=16 bytes even when never read.
  WGPUBuffer dummy = make_storage_zeros(device, 16);
  return {dummy, 16, dummy};
}

// One compute-shader binding: a bind-group-layout entry + its bind-group
// entry share every field except `type` (layout) vs `buffer`/`size` (bind).
struct BindingSpec {
  uint32_t binding;
  WGPUBufferBindingType type;
  WGPUBuffer buffer;
  uint64_t size;
};

// Owns the shader module, bind-group layout, and pipeline layout, releasing
// them on destruction. `pipeline` and `bind_group` are NOT released here —
// every op hands them to WebGPUGraph::add_dispatch, which keeps them alive
// for the life of the graph (mirrors the "kept by dispatch" comment repeated
// across every op handler).
struct ComputePipelineBundle {
  WGPUShaderModule shader = nullptr;
  WGPUBindGroupLayout bind_group_layout = nullptr;
  WGPUPipelineLayout pipeline_layout = nullptr;
  WGPUComputePipeline pipeline = nullptr;
  WGPUBindGroup bind_group = nullptr;

  ComputePipelineBundle() = default;
  ComputePipelineBundle(const ComputePipelineBundle&) = delete;
  ComputePipelineBundle& operator=(const ComputePipelineBundle&) = delete;
  ComputePipelineBundle& operator=(ComputePipelineBundle&&) = delete;

  ComputePipelineBundle(ComputePipelineBundle&& other) noexcept
      : shader(other.shader),
        bind_group_layout(other.bind_group_layout),
        pipeline_layout(other.pipeline_layout),
        pipeline(other.pipeline),
        bind_group(other.bind_group) {
    other.shader = nullptr;
    other.bind_group_layout = nullptr;
    other.pipeline_layout = nullptr;
    other.pipeline = nullptr;
    other.bind_group = nullptr;
  }

  ~ComputePipelineBundle() {
    if (shader != nullptr) {
      wgpuShaderModuleRelease(shader);
    }
    if (bind_group_layout != nullptr) {
      wgpuBindGroupLayoutRelease(bind_group_layout);
    }
    if (pipeline_layout != nullptr) {
      wgpuPipelineLayoutRelease(pipeline_layout);
    }
  }
};

// Builds shader module -> bind-group layout -> pipeline layout -> compute
// pipeline -> bind group from one binding list, replacing the ~50-line
// sequence duplicated (with varying binding counts) across every op handler.
inline ComputePipelineBundle make_compute_pipeline(
    WGPUDevice device,
    const char* wgsl_source,
    const std::vector<BindingSpec>& bindings,
    const WGPUConstantEntry* constants = nullptr,
    size_t constant_count = 0,
    const char* entry_point = "main") {
  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {wgsl_source, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  std::vector<WGPUBindGroupEntry> bind_entries(bindings.size());
  for (size_t i = 0; i < bindings.size(); i++) {
    bind_entries[i] = {};
    bind_entries[i].binding = bindings[i].binding;
    bind_entries[i].buffer = bindings[i].buffer;
    bind_entries[i].size = bindings[i].size;
  }

  // layout = nullptr => WebGPU auto-derives the bind-group layout from the
  // shader's statically-used @group/@binding declarations, replacing the
  // hand-built bind-group layout + pipeline layout. Precondition: every
  // declared binding must be statically referenced by the shader (optional
  // bindings backed by a dummy buffer are, under a runtime guard) -- auto
  // layout omits an unreferenced binding, whose bind-group entry would then
  // mismatch. BindingSpec.type is unused under auto layout (kept so call sites
  // need no change).
  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = nullptr;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {entry_point, WGPU_STRLEN};
  pipeline_desc.compute.constantCount = constant_count;
  pipeline_desc.compute.constants = constants;
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  // Owned reference (must be released) -> handed to the bundle dtor.
  WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = bind_entries.size();
  bg_desc.entries = bind_entries.data();
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  ComputePipelineBundle bundle;
  bundle.shader = shader;
  bundle.bind_group_layout = bgl;
  bundle.pipeline_layout = nullptr; // none created under auto layout
  bundle.pipeline = pipeline;
  bundle.bind_group = bind_group;
  return bundle;
}

// The {wg_size, stride_x} override-constant pair every 2D-spill dispatch
// builds from its DispatchGrid; was hand-rolled identically at 7 call sites.
inline std::array<WGPUConstantEntry, 2> make_grid_constants(
    const DispatchGrid& grid) {
  std::array<WGPUConstantEntry, 2> constants = {};
  constants[0].key = {"wg_size", WGPU_STRLEN};
  constants[0].value = static_cast<double>(grid.wg_size);
  constants[1].key = {"stride_x", WGPU_STRLEN};
  constants[1].value = static_cast<double>(grid.stride_x);
  return constants;
}

// The 1-element sibling of make_grid_constants: the single wg_size override-
// constant, hand-rolled at 5 call sites.
inline WGPUConstantEntry make_wg_size_constant(uint32_t wg_size) {
  WGPUConstantEntry constant = {};
  constant.key = {"wg_size", WGPU_STRLEN};
  constant.value = static_cast<double>(wg_size);
  return constant;
}

} // namespace executorch::backends::webgpu::utils
