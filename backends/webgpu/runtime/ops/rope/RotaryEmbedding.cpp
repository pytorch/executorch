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
#include <executorch/backends/webgpu/runtime/ops/rope/rotary_embedding_hf_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/rope/rotary_embedding_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct (16-byte aligned, 32 bytes).
struct RotaryParams {
  uint32_t n_heads;
  uint32_t seq;
  uint32_t head_dim;
  uint32_t half_dim;
  uint32_t num_pairs;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};
static_assert(sizeof(RotaryParams) == 32, "RotaryParams must be 32 bytes");

// A rope dispatch: its param-uniform (rewritten on resize) and its index in the
// graph's dispatch list (so a resize hook can update the workgroup count).
struct RopeDispatch {
  WGPUBuffer uniform;
  size_t dispatch_index;
};

// Rotate one (x->out) with its own pipeline; freqs shared between xq and xk.
RopeDispatch add_rope_dispatch(
    WebGPUGraph& graph,
    WGPUDevice device,
    uint32_t wg_size,
    const WebGPUTensor& x,
    const WebGPUTensor& out,
    const WebGPUTensor& freqs_cos,
    const WebGPUTensor& freqs_sin,
    uint32_t n_heads,
    uint32_t seq,
    uint32_t head_dim,
    uint32_t workgroup_count) {
  const uint32_t half_dim = head_dim / 2u;
  // out.dims == in.dims (asserted in impl), so this matches the caller's wgc.
  const uint32_t num_pairs =
      static_cast<uint32_t>(utils::numel_of(out.dims) / 2u);

  RotaryParams params = {};
  params.n_heads = n_heads;
  params.seq = seq;
  params.head_dim = head_dim;
  params.half_dim = half_dim;
  params.num_pairs = num_pairs;

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(RotaryParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped =
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(RotaryParams));
  std::memcpy(mapped, &params, sizeof(RotaryParams));
  wgpuBufferUnmap(uniform_buffer);
  graph.add_uniform_buffer_bytes(sizeof(RotaryParams));

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kRotaryEmbeddingWGSL,
      {
          {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
          {1, WGPUBufferBindingType_ReadOnlyStorage, x.buffer, x.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           freqs_cos.buffer,
           freqs_cos.nbytes},
          {3,
           WGPUBufferBindingType_ReadOnlyStorage,
           freqs_sin.buffer,
           freqs_sin.nbytes},
          {4,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(RotaryParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_index = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count,
       "apply_rotary_emb"});

  // Graph owns it so a resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(uniform_buffer);
  return {uniform_buffer, dispatch_index};
}

// Resize hook body: recompute S/num_pairs + both dispatches; out follows xq/xk.
void resize_rope(
    WebGPUGraph& g,
    int xq_id,
    int xk_id,
    int xq_out_id,
    int xk_out_id,
    uint32_t n_heads_q,
    uint32_t n_heads_k,
    uint32_t head_dim,
    uint32_t half_dim,
    uint32_t wg_size,
    size_t q_idx,
    size_t k_idx,
    WGPUBuffer q_ubuf,
    WGPUBuffer k_ubuf) {
  const auto& qd = g.cur_dims(xq_id);
  const auto& kd = g.cur_dims(xk_id);
  if (qd.size() < 3 || kd.size() < 3) {
    throw std::runtime_error("apply_rotary_emb(resize): q/k rank must be >= 3");
  }
  const uint32_t s = static_cast<uint32_t>(qd[qd.size() - 3]);
  const uint64_t qn = utils::numel_of(qd);
  const uint64_t kn = utils::numel_of(kd);
  // pk = pq (seq=s); require k's seq == s, not silently q's.
  if (static_cast<uint32_t>(kd[kd.size() - 3]) != s) {
    throw std::runtime_error(
        "apply_rotary_emb(resize): q and k seq lengths differ");
  }
  // freqs stay max-allocated; shader indexes by position (S = prefix).
  RotaryParams pq = {};
  pq.n_heads = n_heads_q;
  pq.seq = s;
  pq.head_dim = head_dim;
  pq.half_dim = half_dim;
  pq.num_pairs = static_cast<uint32_t>(qn / 2u);
  RotaryParams pk = pq;
  pk.n_heads = n_heads_k;
  pk.num_pairs = static_cast<uint32_t>(kn / 2u);
  wgpuQueueWriteBuffer(g.queue(), q_ubuf, 0, &pq, sizeof(pq));
  wgpuQueueWriteBuffer(g.queue(), k_ubuf, 0, &pk, sizeof(pk));
  g.dispatch_at(q_idx).workgroup_count_x = utils::compute_1d_workgroup_count(
      g.device(),
      static_cast<uint32_t>(qn / 2u),
      wg_size,
      "apply_rotary_emb(resize)");
  g.dispatch_at(k_idx).workgroup_count_x = utils::compute_1d_workgroup_count(
      g.device(),
      static_cast<uint32_t>(kn / 2u),
      wg_size,
      "apply_rotary_emb(resize)");
  g.set_cur_dims(xq_out_id, qd);
  g.set_cur_dims(xk_out_id, kd);
}

// args: [xq, xk, freqs_cos, freqs_sin, out_list(ValueList[xq_out, xk_out])].
void apply_rotary_emb_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int xq_id = args.at(0);
  const int xk_id = args.at(1);
  const int freqs_cos_id = args.at(2);
  const int freqs_sin_id = args.at(3);

  const std::vector<int>& out_list = graph.get_value_list(args.at(4));
  if (out_list.size() != 2) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: expected an output ValueList of size 2");
  }

  WGPUDevice device = graph.device();

  const auto& xq = graph.get_tensor(xq_id);
  const auto& xk = graph.get_tensor(xk_id);
  const auto& freqs_cos = graph.get_tensor(freqs_cos_id);
  const auto& freqs_sin = graph.get_tensor(freqs_sin_id);
  const auto& xq_out = graph.get_tensor(out_list[0]);
  const auto& xk_out = graph.get_tensor(out_list[1]);

  // Vulkan shape contract: xq/xk (B,S,n_heads,head_dim), freqs (S,head_dim/2).
  if (xq.dims.size() < 3 || xk.dims.size() < 3 || freqs_cos.dims.size() < 2) {
    throw std::runtime_error("WebGPU apply_rotary_emb: malformed dims");
  }
  const uint32_t head_dim = static_cast<uint32_t>(xq.dims.back());
  const uint32_t seq = static_cast<uint32_t>(xq.dims[xq.dims.size() - 3]);
  const uint32_t n_heads_q = static_cast<uint32_t>(xq.dims[xq.dims.size() - 2]);
  const uint32_t n_heads_k = static_cast<uint32_t>(xk.dims[xk.dims.size() - 2]);
  const uint32_t seq_k = static_cast<uint32_t>(xk.dims[xk.dims.size() - 3]);
  const uint32_t half_dim = static_cast<uint32_t>(freqs_cos.dims.back());

  if (head_dim == 0 || head_dim % 2 != 0) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: head_dim must be a nonzero multiple of 2");
  }
  if (static_cast<uint32_t>(xk.dims.back()) != head_dim || seq_k != seq) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: xq/xk head_dim and seq must match");
  }
  if (half_dim * 2u != head_dim) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: head_dim != 2 * freqs_cos last dim");
  }
  if (freqs_cos.dims != freqs_sin.dims) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: freqs_cos and freqs_sin shapes differ");
  }

  if (xq.buffer == nullptr || xk.buffer == nullptr ||
      freqs_cos.buffer == nullptr || freqs_sin.buffer == nullptr ||
      xq_out.buffer == nullptr || xk_out.buffer == nullptr) {
    throw std::runtime_error("WebGPU apply_rotary_emb: null buffer binding");
  }

  // All tensors are fp32; output shapes equal their inputs.
  const uint64_t xq_numel = utils::numel_of(xq.dims);
  const uint64_t xk_numel = utils::numel_of(xk.dims);
  const uint64_t freqs_numel = utils::numel_of(freqs_cos.dims);
  if (freqs_numel != static_cast<uint64_t>(seq) * half_dim ||
      xq.nbytes != xq_numel * sizeof(float) ||
      xk.nbytes != xk_numel * sizeof(float) ||
      freqs_cos.nbytes != freqs_numel * sizeof(float) ||
      freqs_sin.nbytes != freqs_numel * sizeof(float) ||
      xq_out.nbytes != xq_numel * sizeof(float) ||
      xk_out.nbytes != xk_numel * sizeof(float)) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: dtype/byte-size mismatch (all fp32) or "
        "freqs shape != [seq, head_dim/2]");
  }

  if (xq_numel / 2u > UINT32_MAX || xk_numel / 2u > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: pair count exceeds uint32 dispatch range");
  }

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kRotaryEmbeddingWorkgroupSizeX);
  // Validate both dispatches before any GPU-object alloc (no leak on throw).
  const uint32_t xq_wgc = utils::compute_1d_workgroup_count(
      device,
      static_cast<uint32_t>(xq_numel / 2u),
      wg_size,
      "apply_rotary_emb");
  const uint32_t xk_wgc = utils::compute_1d_workgroup_count(
      device,
      static_cast<uint32_t>(xk_numel / 2u),
      wg_size,
      "apply_rotary_emb");

  RopeDispatch q_disp = add_rope_dispatch(
      graph,
      device,
      wg_size,
      xq,
      xq_out,
      freqs_cos,
      freqs_sin,
      n_heads_q,
      seq,
      head_dim,
      xq_wgc);
  RopeDispatch k_disp = add_rope_dispatch(
      graph,
      device,
      wg_size,
      xk,
      xk_out,
      freqs_cos,
      freqs_sin,
      n_heads_k,
      seq,
      head_dim,
      xk_wgc);
  WGPUBuffer q_ubuf = q_disp.uniform;
  WGPUBuffer k_ubuf = k_disp.uniform;
  const size_t q_idx = q_disp.dispatch_index;
  const size_t k_idx = k_disp.dispatch_index;

  // Dynamic shapes: recompute S/num_pairs + both dispatches; out follows xq/xk.
  const int xq_out_id = out_list[0];
  const int xk_out_id = out_list[1];
  // Register on both xq and xk so the recompute fires whichever is marked dirty
  // (q and k co-resize on S; resize_rope is idempotent, so a double-fire when
  // both are dirty is harmless).
  auto rope_hook = [xq_id,
                    xk_id,
                    xq_out_id,
                    xk_out_id,
                    n_heads_q,
                    n_heads_k,
                    head_dim,
                    half_dim,
                    wg_size,
                    q_idx,
                    k_idx,
                    q_ubuf,
                    k_ubuf](WebGPUGraph& g) {
    resize_rope(
        g,
        xq_id,
        xk_id,
        xq_out_id,
        xk_out_id,
        n_heads_q,
        n_heads_k,
        head_dim,
        half_dim,
        wg_size,
        q_idx,
        k_idx,
        q_ubuf,
        k_ubuf);
  };
  graph.add_tensor_resize_hook(xq_id, rope_hook);
  graph.add_tensor_resize_hook(xk_id, rope_hook);
}

// HuggingFace rotate-half RoPE (Qwen3 etc.). Structural sibling of the
// interleaved path above (same one-thread-per-pair scalar dispatch, wg_size,
// and resize hook); differs only in element pairing (i with i+half_dim vs
// even/odd), a full [max_seq, rotary_dim] freqs table, and a start_pos offset.
// Mirrors Vulkan's et_vk.apply_rotary_emb_hf
// (backends/vulkan/runtime/graph/ops/impl/RotaryEmbedding.cpp:211).

// Uniform layout matching the HF WGSL Params struct (32 bytes).
struct RotaryHfParams {
  uint32_t n_heads;
  uint32_t seq;
  uint32_t head_dim;
  uint32_t half_dim;
  uint32_t num_pairs;
  uint32_t rotary_dim;
  uint32_t start_pos;
  uint32_t _pad0;
};
static_assert(sizeof(RotaryHfParams) == 32, "RotaryHfParams must be 32 bytes");

RopeDispatch add_rope_hf_dispatch(
    WebGPUGraph& graph,
    WGPUDevice device,
    uint32_t wg_size,
    const WebGPUTensor& x,
    const WebGPUTensor& out,
    const WebGPUTensor& freqs_cos,
    const WebGPUTensor& freqs_sin,
    uint32_t n_heads,
    uint32_t seq,
    uint32_t head_dim,
    uint32_t half_dim,
    uint32_t rotary_dim,
    uint32_t start_pos,
    uint32_t workgroup_count) {
  const uint32_t num_pairs =
      static_cast<uint32_t>(utils::numel_of(out.dims) / 2u);

  RotaryHfParams params = {};
  params.n_heads = n_heads;
  params.seq = seq;
  params.head_dim = head_dim;
  params.half_dim = half_dim;
  params.num_pairs = num_pairs;
  params.rotary_dim = rotary_dim;
  params.start_pos = start_pos;

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(RotaryHfParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped =
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(RotaryHfParams));
  std::memcpy(mapped, &params, sizeof(RotaryHfParams));
  wgpuBufferUnmap(uniform_buffer);
  graph.add_uniform_buffer_bytes(sizeof(RotaryHfParams));

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kRotaryEmbeddingHfWGSL,
      {
          {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
          {1, WGPUBufferBindingType_ReadOnlyStorage, x.buffer, x.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           freqs_cos.buffer,
           freqs_cos.nbytes},
          {3,
           WGPUBufferBindingType_ReadOnlyStorage,
           freqs_sin.buffer,
           freqs_sin.nbytes},
          {4,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(RotaryHfParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_index = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count,
       "apply_rotary_emb_hf"});

  graph.own_uniform_buffer(uniform_buffer);
  return {uniform_buffer, dispatch_index};
}

// Resize hook body: recompute S/num_pairs + (dynamic) start_pos for both
// dispatches; out follows xq/xk. Fires on xq/xk seq resize and, when start_pos
// is a runtime SymInt (KV-cache decode), on each start_pos change; idempotent.
void resize_rope_hf(
    WebGPUGraph& g,
    int xq_id,
    int xk_id,
    int xq_out_id,
    int xk_out_id,
    int start_pos_id,
    bool dynamic_pos,
    uint32_t baked_start_pos,
    uint32_t n_heads_q,
    uint32_t n_heads_k,
    uint32_t head_dim,
    uint32_t half_dim,
    uint32_t rotary_dim,
    uint32_t wg_size,
    size_t q_idx,
    size_t k_idx,
    WGPUBuffer q_ubuf,
    WGPUBuffer k_ubuf) {
  const auto& qd = g.cur_dims(xq_id);
  const auto& kd = g.cur_dims(xk_id);
  if (qd.size() < 3 || kd.size() < 3) {
    throw std::runtime_error(
        "apply_rotary_emb_hf(resize): q/k rank must be >= 3");
  }
  const uint32_t s = static_cast<uint32_t>(qd[qd.size() - 3]);
  const uint64_t qn = utils::numel_of(qd);
  const uint64_t kn = utils::numel_of(kd);
  if (static_cast<uint32_t>(kd[kd.size() - 3]) != s) {
    throw std::runtime_error(
        "apply_rotary_emb_hf(resize): q and k seq lengths differ");
  }
  uint32_t start_pos = baked_start_pos;
  if (dynamic_pos) {
    const int32_t pos = g.read_symint(start_pos_id);
    if (pos < 0) {
      throw std::runtime_error(
          "apply_rotary_emb_hf(resize): start_pos must be non-negative");
    }
    start_pos = static_cast<uint32_t>(pos);
  }
  RotaryHfParams pq = {};
  pq.n_heads = n_heads_q;
  pq.seq = s;
  pq.head_dim = head_dim;
  pq.half_dim = half_dim;
  pq.num_pairs = static_cast<uint32_t>(qn / 2u);
  pq.rotary_dim = rotary_dim;
  pq.start_pos = start_pos;
  RotaryHfParams pk = pq;
  pk.n_heads = n_heads_k;
  pk.num_pairs = static_cast<uint32_t>(kn / 2u);
  wgpuQueueWriteBuffer(g.queue(), q_ubuf, 0, &pq, sizeof(pq));
  wgpuQueueWriteBuffer(g.queue(), k_ubuf, 0, &pk, sizeof(pk));
  g.dispatch_at(q_idx).workgroup_count_x = utils::compute_1d_workgroup_count(
      g.device(),
      static_cast<uint32_t>(qn / 2u),
      wg_size,
      "apply_rotary_emb_hf(resize)");
  g.dispatch_at(k_idx).workgroup_count_x = utils::compute_1d_workgroup_count(
      g.device(),
      static_cast<uint32_t>(kn / 2u),
      wg_size,
      "apply_rotary_emb_hf(resize)");
  g.set_cur_dims(xq_out_id, qd);
  g.set_cur_dims(xk_out_id, kd);
}

// args: [xq, xk, freqs_cos, freqs_sin, start_pos, out_list(ValueList[xq_out,
// xk_out])]. freqs is the FULL [max_seq, rotary_dim] table (start_pos offsets
// into it), unlike the pre-sliced interleaved freqs.
void apply_rotary_emb_hf_impl(
    WebGPUGraph& graph,
    const std::vector<int>& args) {
  const int xq_id = args.at(0);
  const int xk_id = args.at(1);
  const int freqs_cos_id = args.at(2);
  const int freqs_sin_id = args.at(3);
  const int start_pos_id = args.at(4);

  const std::vector<int>& out_list = graph.get_value_list(args.at(5));
  if (out_list.size() != 2) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: expected an output ValueList of size 2");
  }

  WGPUDevice device = graph.device();

  const auto& xq = graph.get_tensor(xq_id);
  const auto& xk = graph.get_tensor(xk_id);
  const auto& freqs_cos = graph.get_tensor(freqs_cos_id);
  const auto& freqs_sin = graph.get_tensor(freqs_sin_id);
  const auto& xq_out = graph.get_tensor(out_list[0]);
  const auto& xk_out = graph.get_tensor(out_list[1]);

  // Shape contract: xq/xk (B,S,n_heads,head_dim), freqs (max_seq, rotary_dim).
  if (xq.dims.size() < 3 || xk.dims.size() < 3 || freqs_cos.dims.size() < 2) {
    throw std::runtime_error("WebGPU apply_rotary_emb_hf: malformed dims");
  }
  const uint32_t head_dim = static_cast<uint32_t>(xq.dims.back());
  const uint32_t seq = static_cast<uint32_t>(xq.dims[xq.dims.size() - 3]);
  const uint32_t n_heads_q = static_cast<uint32_t>(xq.dims[xq.dims.size() - 2]);
  const uint32_t n_heads_k = static_cast<uint32_t>(xk.dims[xk.dims.size() - 2]);
  const uint32_t seq_k = static_cast<uint32_t>(xk.dims[xk.dims.size() - 3]);
  const uint32_t max_seq =
      static_cast<uint32_t>(freqs_cos.dims[freqs_cos.dims.size() - 2]);
  const uint32_t rotary_dim = static_cast<uint32_t>(freqs_cos.dims.back());

  if (head_dim == 0 || head_dim % 2 != 0) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: head_dim must be a nonzero multiple of 2");
  }
  if (static_cast<uint32_t>(xk.dims.back()) != head_dim || seq_k != seq) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: xq/xk head_dim and seq must match");
  }
  // Full rotary only (rotary_dim == head_dim); partial-rotary passthrough is a
  // documented follow-up (Qwen3 uses full RoPE). Throw rather than mis-rotate.
  if (rotary_dim != head_dim) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: partial rotary (rotary_dim != head_dim) "
        "not supported");
  }
  if (freqs_cos.dims != freqs_sin.dims) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: freqs_cos and freqs_sin shapes differ");
  }
  if (max_seq < seq) {
    throw std::runtime_error("WebGPU apply_rotary_emb_hf: freqs max_seq < seq");
  }

  if (xq.buffer == nullptr || xk.buffer == nullptr ||
      freqs_cos.buffer == nullptr || freqs_sin.buffer == nullptr ||
      xq_out.buffer == nullptr || xk_out.buffer == nullptr) {
    throw std::runtime_error("WebGPU apply_rotary_emb_hf: null buffer binding");
  }

  const uint32_t half_dim = rotary_dim / 2u;

  // start_pos: build-time Int (baked) OR runtime SymInt (dynamic decode);
  // mirrors sdpa's input_pos handling.
  int64_t start_pos = 0;
  const auto start_pos_type = graph.get_value_type(start_pos_id);
  const bool dynamic_pos = start_pos_type == WebGPUGraph::ValueType::SymInt;
  if (dynamic_pos) {
    start_pos = graph.read_symint(start_pos_id); // build placeholder (e.g. 0)
  } else if (start_pos_type == WebGPUGraph::ValueType::Int) {
    start_pos = graph.get_int(start_pos_id);
  } else {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: start_pos must be Int or SymInt");
  }
  if (start_pos < 0) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: start_pos must be non-negative");
  }

  // All tensors are fp32; output shapes equal their inputs.
  const uint64_t xq_numel = utils::numel_of(xq.dims);
  const uint64_t xk_numel = utils::numel_of(xk.dims);
  const uint64_t freqs_numel = utils::numel_of(freqs_cos.dims);
  if (freqs_numel != static_cast<uint64_t>(max_seq) * rotary_dim ||
      xq.nbytes != xq_numel * sizeof(float) ||
      xk.nbytes != xk_numel * sizeof(float) ||
      freqs_cos.nbytes != freqs_numel * sizeof(float) ||
      freqs_sin.nbytes != freqs_numel * sizeof(float) ||
      xq_out.nbytes != xq_numel * sizeof(float) ||
      xk_out.nbytes != xk_numel * sizeof(float)) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: dtype/byte-size mismatch (all fp32) or "
        "freqs shape != [max_seq, rotary_dim]");
  }

  if (xq_numel / 2u > UINT32_MAX || xk_numel / 2u > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: pair count exceeds uint32 dispatch range");
  }

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kRotaryEmbeddingHfWorkgroupSizeX);
  // Validate both dispatches before any GPU-object alloc (no leak on throw).
  const uint32_t xq_wgc = utils::compute_1d_workgroup_count(
      device,
      static_cast<uint32_t>(xq_numel / 2u),
      wg_size,
      "apply_rotary_emb_hf");
  const uint32_t xk_wgc = utils::compute_1d_workgroup_count(
      device,
      static_cast<uint32_t>(xk_numel / 2u),
      wg_size,
      "apply_rotary_emb_hf");

  RopeDispatch q_disp = add_rope_hf_dispatch(
      graph,
      device,
      wg_size,
      xq,
      xq_out,
      freqs_cos,
      freqs_sin,
      n_heads_q,
      seq,
      head_dim,
      half_dim,
      rotary_dim,
      static_cast<uint32_t>(start_pos),
      xq_wgc);
  RopeDispatch k_disp = add_rope_hf_dispatch(
      graph,
      device,
      wg_size,
      xk,
      xk_out,
      freqs_cos,
      freqs_sin,
      n_heads_k,
      seq,
      head_dim,
      half_dim,
      rotary_dim,
      static_cast<uint32_t>(start_pos),
      xk_wgc);
  WGPUBuffer q_ubuf = q_disp.uniform;
  WGPUBuffer k_ubuf = k_disp.uniform;
  const size_t q_idx = q_disp.dispatch_index;
  const size_t k_idx = k_disp.dispatch_index;

  const int xq_out_id = out_list[0];
  const int xk_out_id = out_list[1];
  const uint32_t baked_start_pos = static_cast<uint32_t>(start_pos);
  auto rope_hook = [xq_id,
                    xk_id,
                    xq_out_id,
                    xk_out_id,
                    start_pos_id,
                    dynamic_pos,
                    baked_start_pos,
                    n_heads_q,
                    n_heads_k,
                    head_dim,
                    half_dim,
                    rotary_dim,
                    wg_size,
                    q_idx,
                    k_idx,
                    q_ubuf,
                    k_ubuf](WebGPUGraph& g) {
    resize_rope_hf(
        g,
        xq_id,
        xk_id,
        xq_out_id,
        xk_out_id,
        start_pos_id,
        dynamic_pos,
        baked_start_pos,
        n_heads_q,
        n_heads_k,
        head_dim,
        half_dim,
        rotary_dim,
        wg_size,
        q_idx,
        k_idx,
        q_ubuf,
        k_ubuf);
  };
  graph.add_tensor_resize_hook(xq_id, rope_hook);
  graph.add_tensor_resize_hook(xk_id, rope_hook);
  // Dynamic decode: re-fire when the runtime start_pos SymInt changes.
  if (dynamic_pos) {
    graph.add_resize_hook(start_pos_id, rope_hook);
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.apply_rotary_emb.default, apply_rotary_emb_impl);
  WEBGPU_REGISTER_OP(
      et_vk.apply_rotary_emb_hf.default, apply_rotary_emb_hf_impl);
}

} // namespace executorch::backends::webgpu
