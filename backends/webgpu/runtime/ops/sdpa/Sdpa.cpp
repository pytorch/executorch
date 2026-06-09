/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa/sdpa_compute_attn_weights_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa/sdpa_compute_out_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa/sdpa_softmax_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/update_cache/update_cache_wgsl.h>

#include <webgpu/webgpu.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace executorch::backends::webgpu {

namespace {

// Uniform param structs (all 16-byte aligned, matching the WGSL Params).
struct UpdateCacheParams {
  uint32_t numel;
  uint32_t dst_offset;
  uint32_t cache_numel;
  uint32_t _pad0;
};
static_assert(sizeof(UpdateCacheParams) == 16, "UpdateCacheParams must be 16B");

struct AttnWeightsParams {
  uint32_t S;
  uint32_t Hq;
  uint32_t Hkv;
  uint32_t D;
  uint32_t context_len;
  uint32_t input_pos;
  uint32_t g;
  float scale;
};
static_assert(sizeof(AttnWeightsParams) == 32, "AttnWeightsParams must be 32B");

struct SoftmaxParams {
  uint32_t num_rows;
  uint32_t row_width;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(SoftmaxParams) == 16, "SoftmaxParams must be 16B");

struct ComputeOutParams {
  uint32_t S;
  uint32_t Hq;
  uint32_t Hkv;
  uint32_t D;
  uint32_t context_len;
  uint32_t g;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(ComputeOutParams) == 32, "ComputeOutParams must be 32B");

// Create a uniform buffer initialized with the given bytes.
WGPUBuffer
make_uniform_buffer(WebGPUGraph& graph, const void* data, size_t size) {
  WGPUDevice device = graph.device();
  WGPUBufferDescriptor desc = {};
  desc.size = size;
  desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  desc.mappedAtCreation = true;
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &desc);
  void* mapped = wgpuBufferGetMappedRange(buffer, 0, size);
  std::memcpy(mapped, data, size);
  wgpuBufferUnmap(buffer);
  graph.add_uniform_buffer_bytes(size);
  return buffer;
}

// A buffer + its byte size, for binding.
struct BufferBinding {
  WGPUBuffer buffer;
  uint64_t size;
};

// Build one dispatch (pipeline + bind group) and record it on the graph.
void build_dispatch(
    WebGPUGraph& graph,
    const char* wgsl_source,
    const BufferBinding* storage_bindings,
    uint32_t n_storage, // includes the rw output at index 0
    WGPUBuffer uniform_buffer,
    uint64_t uniform_size,
    uint32_t workgroup_count_x,
    uint32_t wg_size,
    bool retain_uniform = false) {
  WGPUDevice device = graph.device();

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {wgsl_source, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  // Bind group layout: storage entries then the uniform.
  constexpr uint32_t kMaxEntries = 8;
  WGPUBindGroupLayoutEntry bgl_entries[kMaxEntries] = {};
  const uint32_t uniform_binding = n_storage;
  for (uint32_t i = 0; i < n_storage; i++) {
    bgl_entries[i].binding = i;
    bgl_entries[i].visibility = WGPUShaderStage_Compute;
    bgl_entries[i].buffer.type = (i == 0)
        ? WGPUBufferBindingType_Storage
        : WGPUBufferBindingType_ReadOnlyStorage;
  }
  bgl_entries[uniform_binding].binding = uniform_binding;
  bgl_entries[uniform_binding].visibility = WGPUShaderStage_Compute;
  bgl_entries[uniform_binding].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = n_storage + 1;
  bgl_desc.entries = bgl_entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  // QK/AV/update_cache have an `override wg_size`; softmax (0) keeps a const.
  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  if (wg_size != 0) {
    pipeline_desc.compute.constantCount = 1;
    pipeline_desc.compute.constants = &wg_size_constant;
  }
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  WGPUBindGroupEntry bg_entries[kMaxEntries] = {};
  for (uint32_t i = 0; i < n_storage; i++) {
    bg_entries[i].binding = i;
    bg_entries[i].buffer = storage_bindings[i].buffer;
    bg_entries[i].size = storage_bindings[i].size;
  }
  bg_entries[uniform_binding].binding = uniform_binding;
  bg_entries[uniform_binding].buffer = uniform_buffer;
  bg_entries[uniform_binding].size = uniform_size;

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = n_storage + 1;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch({pipeline, bind_group, workgroup_count_x});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  if (retain_uniform) {
    // Graph owns it so a resize hook can rewrite it; freed in the dtor.
    graph.own_uniform_buffer(uniform_buffer);
  } else {
    // Drop our ref; the bind group keeps the uniform alive.
    wgpuBufferRelease(uniform_buffer);
  }
}

// llama.sdpa_with_kv_cache.default args mirror the Vulkan impl.
void sdpa_with_kv_cache_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int q_id = args.at(0);
  const int k_id = args.at(1);
  const int v_id = args.at(2);
  const int k_cache_id = args.at(3);
  const int v_cache_id = args.at(4);
  const int input_pos_id = args.at(5);
  // arg 6 (seq_len) is derived from q; args 7-9 validated below.
  const int attn_mask_id = args.at(7);
  const int drop_p_id = args.at(8);
  const int is_causal_id = args.at(9);
  const int scale_id = args.at(10);
  const int out_id = args.at(11);

  const auto& q = graph.get_tensor(q_id);
  const auto& k = graph.get_tensor(k_id);
  const auto& v = graph.get_tensor(v_id);
  const auto& k_cache = graph.get_tensor(k_cache_id);
  const auto& v_cache = graph.get_tensor(v_cache_id);
  const auto& out = graph.get_tensor(out_id);

  if (q.dims.size() < 3 || k.dims.size() < 3 || k_cache.dims.size() < 3) {
    throw std::runtime_error("WebGPU sdpa: q/k/k_cache must be rank >= 3");
  }

  // q [1, S, Hq, D]; k/v [1, S, Hkv, D]; caches [1, Cmax, Hkv, D].
  const size_t qn = q.dims.size();
  const int64_t S = q.dims[qn - 3];
  const int64_t Hq = q.dims[qn - 2];
  const int64_t D = q.dims[qn - 1];

  const size_t kn = k.dims.size();
  const int64_t Hkv = k.dims[kn - 2];

  const size_t cn = k_cache.dims.size();
  const int64_t Cmax = k_cache.dims[cn - 3];

  // Validate B == 1 (leading dims must all be 1).
  for (size_t i = 0; i + 3 < qn; i++) {
    if (q.dims[i] != 1) {
      throw std::runtime_error("WebGPU sdpa: only batch size 1 is supported");
    }
  }
  if (S <= 0 || Hq <= 0 || D <= 0 || Hkv <= 0 || Cmax <= 0) {
    throw std::runtime_error("WebGPU sdpa: non-positive dimension");
  }
  if (Hq % Hkv != 0) {
    throw std::runtime_error("WebGPU sdpa: Hq must be a multiple of Hkv (GQA)");
  }
  const int64_t g = Hq / Hkv;

  // Mirrors Vulkan SDPA: q/k_cache head_dim + k_cache/v_cache shape must match.
  if (D != k_cache.dims[cn - 1]) {
    throw std::runtime_error("WebGPU sdpa: q and k_cache head_dim mismatch");
  }
  if (k_cache.dims != v_cache.dims) {
    throw std::runtime_error("WebGPU sdpa: k_cache and v_cache shape mismatch");
  }

  // fp32-only: validate byte counts against fp32 element counts.
  auto numel = [](const WebGPUTensor& t) {
    uint64_t n = 1;
    for (int64_t d : t.dims) {
      n *= static_cast<uint64_t>(d);
    }
    return n;
  };
  if (q.nbytes != numel(q) * sizeof(float) ||
      k.nbytes != numel(k) * sizeof(float) ||
      v.nbytes != numel(v) * sizeof(float) ||
      out.nbytes != numel(out) * sizeof(float)) {
    throw std::runtime_error("WebGPU sdpa: fp32-only (byte-size mismatch)");
  }

  // input_pos: build-time Int (baked) OR runtime SymInt (dynamic decode).
  int64_t input_pos = 0;
  const auto input_pos_type = graph.get_value_type(input_pos_id);
  const bool dynamic_pos = input_pos_type == WebGPUGraph::ValueType::SymInt;
  if (dynamic_pos) {
    input_pos = graph.read_symint(input_pos_id); // build placeholder (e.g. 0)
  } else if (input_pos_type == WebGPUGraph::ValueType::Int) {
    input_pos = graph.get_int(input_pos_id);
  } else {
    // No silent default-to-0; mirrors Vulkan get_or_create_int_param_buffer.
    throw std::runtime_error("WebGPU sdpa: input_pos must be Int or SymInt");
  }
  const int64_t context_len = S + input_pos;
  if (context_len <= 0 || context_len > Cmax) {
    throw std::runtime_error("WebGPU sdpa: context_len exceeds cache capacity");
  }

  // scale arg is None (use 1/sqrt(D)) or an explicit Double; reject others.
  float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto scale_type = graph.get_value_type(scale_id);
  if (scale_type == WebGPUGraph::ValueType::Double) {
    scale = static_cast<float>(graph.get_double(scale_id));
  } else if (scale_type != WebGPUGraph::ValueType::Null) {
    throw std::runtime_error("WebGPU sdpa: scale must be None or a Double");
  }

  // Unsupported attention args must be absent/default; mirrors Vulkan
  // SDPA.cpp:587-593 (scale is handled above as an intentional extension).
  using VT = WebGPUGraph::ValueType;
  if (graph.get_value_type(attn_mask_id) != VT::Null) {
    throw std::runtime_error("WebGPU sdpa: attn_mask is not supported");
  }
  const auto drop_type = graph.get_value_type(drop_p_id);
  if (!(drop_type == VT::Null ||
        (drop_type == VT::Double && graph.get_double(drop_p_id) == 0.0))) {
    throw std::runtime_error("WebGPU sdpa: only dropout_p=0 is supported");
  }
  const auto causal_type = graph.get_value_type(is_causal_id);
  if (!(causal_type == VT::Null ||
        (causal_type == VT::Bool && graph.get_bool(is_causal_id)))) {
    throw std::runtime_error("WebGPU sdpa: only is_causal=true is supported");
  }

  // KV cache written in place; only attn_weights/softmax need scratch.
  const uint64_t aw_floats = static_cast<uint64_t>(Hq) *
      static_cast<uint64_t>(S) * static_cast<uint64_t>(context_len);
  // Dynamic input_pos: size+bind scratch for Cmax (no realloc; covers any ctx).
  const uint64_t aw_cap_floats = static_cast<uint64_t>(Hq) *
      static_cast<uint64_t>(S) *
      static_cast<uint64_t>(dynamic_pos ? Cmax : context_len);
  const uint64_t aw_bytes = aw_cap_floats * sizeof(float);
  WGPUBuffer attn_weights = graph.create_scratch_buffer(aw_bytes);
  WGPUBuffer attn_weights_softmax = graph.create_scratch_buffer(aw_bytes);

  // Dynamic input_pos: the resize hook rewrites these per step.
  WGPUBuffer uc_k_buf = nullptr, uc_v_buf = nullptr, qk_buf = nullptr,
             softmax_buf = nullptr, av_buf = nullptr;
  size_t qk_idx = 0;

  const WGPUDevice device = graph.device();
  const uint32_t uc_wg =
      utils::clamp_workgroup_size(device, kUpdateCacheWorkgroupSizeX);
  const uint32_t qk_wg = utils::clamp_workgroup_size(
      device, kSdpaComputeAttnWeightsWorkgroupSizeX);
  const uint32_t av_wg =
      utils::clamp_workgroup_size(device, kSdpaComputeOutWorkgroupSizeX);

  // Dispatches 1-2: write new K/V into the caches (reuses update_cache).
  const uint64_t kv_numel = static_cast<uint64_t>(S) *
      static_cast<uint64_t>(Hkv) * static_cast<uint64_t>(D);
  const uint32_t kv_dst_offset = static_cast<uint32_t>(
      static_cast<uint64_t>(input_pos) * static_cast<uint64_t>(Hkv) *
      static_cast<uint64_t>(D));
  {
    const uint32_t wgc = utils::compute_1d_workgroup_count(
        device, static_cast<uint32_t>(kv_numel), uc_wg, "update_cache(K)");
    UpdateCacheParams uc = {};
    uc.numel = static_cast<uint32_t>(kv_numel);
    uc.dst_offset = kv_dst_offset;
    uc.cache_numel = static_cast<uint32_t>(numel(k_cache));
    WGPUBuffer ubuf = make_uniform_buffer(graph, &uc, sizeof(uc));
    BufferBinding bindings[2] = {
        {k_cache.buffer, k_cache.nbytes}, {k.buffer, k.nbytes}};
    build_dispatch(
        graph,
        kUpdateCacheWGSL,
        bindings,
        2,
        ubuf,
        sizeof(uc),
        wgc,
        uc_wg,
        dynamic_pos);
    uc_k_buf = ubuf;
  }
  {
    const uint32_t wgc = utils::compute_1d_workgroup_count(
        device, static_cast<uint32_t>(kv_numel), uc_wg, "update_cache(V)");
    UpdateCacheParams uc = {};
    uc.numel = static_cast<uint32_t>(kv_numel);
    uc.dst_offset = kv_dst_offset;
    uc.cache_numel = static_cast<uint32_t>(numel(v_cache));
    WGPUBuffer ubuf = make_uniform_buffer(graph, &uc, sizeof(uc));
    BufferBinding bindings[2] = {
        {v_cache.buffer, v_cache.nbytes}, {v.buffer, v.nbytes}};
    build_dispatch(
        graph,
        kUpdateCacheWGSL,
        bindings,
        2,
        ubuf,
        sizeof(uc),
        wgc,
        uc_wg,
        dynamic_pos);
    uc_v_buf = ubuf;
  }

  // --- Dispatch 3: QK -> attn_weights. One thread per (h,s,c) element.
  {
    const uint32_t wgc = utils::compute_1d_workgroup_count(
        device, static_cast<uint32_t>(aw_floats), qk_wg, "QK");
    AttnWeightsParams p = {};
    p.S = static_cast<uint32_t>(S);
    p.Hq = static_cast<uint32_t>(Hq);
    p.Hkv = static_cast<uint32_t>(Hkv);
    p.D = static_cast<uint32_t>(D);
    p.context_len = static_cast<uint32_t>(context_len);
    p.input_pos = static_cast<uint32_t>(input_pos);
    p.g = static_cast<uint32_t>(g);
    p.scale = scale;
    WGPUBuffer ubuf = make_uniform_buffer(graph, &p, sizeof(p));
    BufferBinding bindings[3] = {
        {attn_weights, aw_bytes},
        {q.buffer, q.nbytes},
        {k_cache.buffer, k_cache.nbytes}};
    build_dispatch(
        graph,
        kSdpaComputeAttnWeightsWGSL,
        bindings,
        3,
        ubuf,
        sizeof(p),
        wgc,
        qk_wg,
        dynamic_pos);
    qk_buf = ubuf;
    qk_idx = graph.num_dispatches() - 1;
  }

  // Dispatch 4: softmax, one workgroup per (h,s) row of width context_len.
  {
    // One workgroup per (h,s) row; wg_size 1 keeps the device dispatch check.
    const uint32_t wgc = utils::compute_1d_workgroup_count(
        device, static_cast<uint32_t>(Hq * S), 1, "softmax");
    SoftmaxParams p = {};
    p.num_rows = wgc;
    p.row_width = static_cast<uint32_t>(context_len);
    WGPUBuffer ubuf = make_uniform_buffer(graph, &p, sizeof(p));
    BufferBinding bindings[2] = {
        {attn_weights_softmax, aw_bytes}, {attn_weights, aw_bytes}};
    build_dispatch(
        graph,
        kSdpaSoftmaxWGSL,
        bindings,
        2,
        ubuf,
        sizeof(p),
        wgc,
        0,
        dynamic_pos);
    softmax_buf = ubuf;
  }

  // --- Dispatch 5: AV -> out. One thread per (s,h,d) output element.
  {
    const uint64_t out_floats = static_cast<uint64_t>(S) *
        static_cast<uint64_t>(Hq) * static_cast<uint64_t>(D);
    const uint32_t wgc = utils::compute_1d_workgroup_count(
        device, static_cast<uint32_t>(out_floats), av_wg, "AV");
    ComputeOutParams p = {};
    p.S = static_cast<uint32_t>(S);
    p.Hq = static_cast<uint32_t>(Hq);
    p.Hkv = static_cast<uint32_t>(Hkv);
    p.D = static_cast<uint32_t>(D);
    p.context_len = static_cast<uint32_t>(context_len);
    p.g = static_cast<uint32_t>(g);
    WGPUBuffer ubuf = make_uniform_buffer(graph, &p, sizeof(p));
    BufferBinding bindings[3] = {
        {out.buffer, out.nbytes},
        {attn_weights_softmax, aw_bytes},
        {v_cache.buffer, v_cache.nbytes}};
    build_dispatch(
        graph,
        kSdpaComputeOutWGSL,
        bindings,
        3,
        ubuf,
        sizeof(p),
        wgc,
        av_wg,
        dynamic_pos);
    av_buf = ubuf;
  }

  // Per-step recompute hook; mirrors Vulkan DynamicDispatchNode.
  if (dynamic_pos) {
    graph.add_resize_hook(
        input_pos_id,
        [input_pos_id,
         S,
         Hq,
         Hkv,
         D,
         Cmax,
         g,
         scale,
         qk_idx,
         qk_wg,
         uc_k_buf,
         uc_v_buf,
         qk_buf,
         softmax_buf,
         av_buf](WebGPUGraph& gr) {
          const int32_t pos = gr.read_symint(input_pos_id);
          const int64_t ctx = S + pos;
          if (ctx <= 0 || ctx > Cmax) {
            throw std::runtime_error(
                "WebGPU sdpa: context_len exceeds cache capacity");
          }
          const uint32_t kv_off = static_cast<uint32_t>(
              static_cast<uint64_t>(pos) * static_cast<uint64_t>(Hkv) *
              static_cast<uint64_t>(D));
          const uint64_t aw_floats = static_cast<uint64_t>(Hq) *
              static_cast<uint64_t>(S) * static_cast<uint64_t>(ctx);

          UpdateCacheParams uc = {};
          uc.numel = static_cast<uint32_t>(
              static_cast<uint64_t>(S) * static_cast<uint64_t>(Hkv) *
              static_cast<uint64_t>(D));
          uc.dst_offset = kv_off;
          uc.cache_numel = static_cast<uint32_t>(
              static_cast<uint64_t>(Cmax) * static_cast<uint64_t>(Hkv) *
              static_cast<uint64_t>(D));
          wgpuQueueWriteBuffer(gr.queue(), uc_k_buf, 0, &uc, sizeof(uc));
          wgpuQueueWriteBuffer(gr.queue(), uc_v_buf, 0, &uc, sizeof(uc));

          AttnWeightsParams qp = {};
          qp.S = static_cast<uint32_t>(S);
          qp.Hq = static_cast<uint32_t>(Hq);
          qp.Hkv = static_cast<uint32_t>(Hkv);
          qp.D = static_cast<uint32_t>(D);
          qp.context_len = static_cast<uint32_t>(ctx);
          qp.input_pos = static_cast<uint32_t>(pos);
          qp.g = static_cast<uint32_t>(g);
          qp.scale = scale;
          wgpuQueueWriteBuffer(gr.queue(), qk_buf, 0, &qp, sizeof(qp));
          const uint32_t qk_wgc = utils::compute_1d_workgroup_count(
              gr.device(),
              static_cast<uint32_t>(aw_floats),
              qk_wg,
              "QK(resize)");
          gr.dispatch_at(qk_idx).workgroup_count_x = qk_wgc;

          SoftmaxParams sp = {};
          sp.num_rows = static_cast<uint32_t>(Hq * S);
          sp.row_width = static_cast<uint32_t>(ctx);
          wgpuQueueWriteBuffer(gr.queue(), softmax_buf, 0, &sp, sizeof(sp));

          ComputeOutParams op = {};
          op.S = static_cast<uint32_t>(S);
          op.Hq = static_cast<uint32_t>(Hq);
          op.Hkv = static_cast<uint32_t>(Hkv);
          op.D = static_cast<uint32_t>(D);
          op.context_len = static_cast<uint32_t>(ctx);
          op.g = static_cast<uint32_t>(g);
          wgpuQueueWriteBuffer(gr.queue(), av_buf, 0, &op, sizeof(op));
        });
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(sdpa_with_kv_cache.default, sdpa_with_kv_cache_impl);
}

} // namespace executorch::backends::webgpu
