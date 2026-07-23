/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUShaderRegistry.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

constexpr const char* kRotaryShader = "rotary_embedding";
constexpr const char* kRotaryHfShader = "rotary_embedding_hf";

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

template <typename Params>
RopeDispatch add_rope_dispatch(
    WebGPUGraph& graph,
    const char* shader_name,
    const char* kernel_name,
    const WebGPUTensor& x,
    const WebGPUTensor& out,
    const WebGPUTensor& freqs_cos,
    const WebGPUTensor& freqs_sin,
    const Params& params,
    uint32_t workgroup_count,
    uint32_t wg_size) {
  WGPUBuffer uniform_buffer = graph.create_params_buffer(params);
  WebGPUComputeDispatchDescriptor descriptor;
  descriptor.shader_name = shader_name;
  descriptor.kernel_name = kernel_name;
  descriptor.bindings = {
      {out.buffer, 0u, out.nbytes},
      {x.buffer, 0u, x.nbytes},
      {freqs_cos.buffer, 0u, freqs_cos.nbytes},
      {freqs_sin.buffer, 0u, freqs_sin.nbytes},
      {uniform_buffer, 0u, sizeof(Params)}};
  descriptor.constants = {{"wg_size", static_cast<double>(wg_size)}};
  descriptor.grid = {workgroup_count, 1u};
  const size_t dispatch_index = graph.add_compute_dispatch(descriptor);
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

  if (xq_numel > UINT32_MAX || xk_numel > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: element index exceeds uint32 range");
  }

  const uint32_t wg_size = utils::clamp_workgroup_size(
      graph.device(), get_webgpu_shader_info(kRotaryShader).workgroup_size_x);
  // Validate both dispatches before any GPU-object alloc (no leak on throw).
  const uint32_t xq_wgc = utils::compute_1d_workgroup_count(
      graph.device(),
      static_cast<uint32_t>(xq_numel / 2u),
      wg_size,
      "apply_rotary_emb");
  const uint32_t xk_wgc = utils::compute_1d_workgroup_count(
      graph.device(),
      static_cast<uint32_t>(xk_numel / 2u),
      wg_size,
      "apply_rotary_emb");

  RotaryParams q_params = {};
  q_params.n_heads = n_heads_q;
  q_params.seq = seq;
  q_params.head_dim = head_dim;
  q_params.half_dim = half_dim;
  q_params.num_pairs = static_cast<uint32_t>(xq_numel / 2u);
  RotaryParams k_params = q_params;
  k_params.n_heads = n_heads_k;
  k_params.num_pairs = static_cast<uint32_t>(xk_numel / 2u);
  const RopeDispatch q_disp = add_rope_dispatch(
      graph,
      kRotaryShader,
      "apply_rotary_emb",
      xq,
      xq_out,
      freqs_cos,
      freqs_sin,
      q_params,
      xq_wgc,
      wg_size);
  const RopeDispatch k_disp = add_rope_dispatch(
      graph,
      kRotaryShader,
      "apply_rotary_emb",
      xk,
      xk_out,
      freqs_cos,
      freqs_sin,
      k_params,
      xk_wgc,
      wg_size);
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

// Mirrors Vulkan's full-dimension HuggingFace rotate-half RoPE.
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

struct RotaryHfGeometry {
  uint32_t head_dim;
  uint32_t seq;
  uint32_t n_heads_q;
  uint32_t n_heads_k;
  uint32_t max_seq;
  uint32_t rotary_dim;
  uint32_t half_dim;
  uint64_t xq_numel;
  uint64_t xk_numel;
};

RotaryHfGeometry validate_rope_hf_inputs(
    const WebGPUTensor& x,
    const WebGPUTensor& xk,
    const WebGPUTensor& freqs_cos,
    const WebGPUTensor& freqs_sin,
    const WebGPUTensor& x_out,
    const WebGPUTensor& xk_out) {
  if (x.dims.size() < 3 || xk.dims.size() < 3 || freqs_cos.dims.size() < 2) {
    throw std::runtime_error("WebGPU apply_rotary_emb_hf: malformed dims");
  }
  RotaryHfGeometry geometry = {};
  geometry.head_dim = static_cast<uint32_t>(x.dims.back());
  geometry.seq = static_cast<uint32_t>(x.dims[x.dims.size() - 3]);
  geometry.n_heads_q = static_cast<uint32_t>(x.dims[x.dims.size() - 2]);
  geometry.n_heads_k = static_cast<uint32_t>(xk.dims[xk.dims.size() - 2]);
  geometry.max_seq =
      static_cast<uint32_t>(freqs_cos.dims[freqs_cos.dims.size() - 2]);
  geometry.rotary_dim = static_cast<uint32_t>(freqs_cos.dims.back());
  if (geometry.head_dim == 0 || geometry.head_dim % 2 != 0) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: head_dim must be a nonzero multiple of 2");
  }
  if (static_cast<uint32_t>(xk.dims.back()) != geometry.head_dim ||
      static_cast<uint32_t>(xk.dims[xk.dims.size() - 3]) != geometry.seq) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: xq/xk head_dim and seq must match");
  }
  if (geometry.rotary_dim != geometry.head_dim) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: partial rotary (rotary_dim != head_dim) "
        "not supported");
  }
  if (freqs_cos.dims != freqs_sin.dims) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: freqs_cos and freqs_sin shapes differ");
  }
  if (geometry.max_seq < geometry.seq) {
    throw std::runtime_error("WebGPU apply_rotary_emb_hf: freqs max_seq < seq");
  }
  if (x.buffer == nullptr || xk.buffer == nullptr ||
      freqs_cos.buffer == nullptr || freqs_sin.buffer == nullptr ||
      x_out.buffer == nullptr || xk_out.buffer == nullptr) {
    throw std::runtime_error("WebGPU apply_rotary_emb_hf: null buffer binding");
  }

  geometry.half_dim = geometry.rotary_dim / 2u;
  geometry.xq_numel = utils::numel_of(x.dims);
  geometry.xk_numel = utils::numel_of(xk.dims);
  const uint64_t freqs_numel = utils::numel_of(freqs_cos.dims);
  if (freqs_numel !=
          static_cast<uint64_t>(geometry.max_seq) * geometry.rotary_dim ||
      x.nbytes != geometry.xq_numel * sizeof(float) ||
      xk.nbytes != geometry.xk_numel * sizeof(float) ||
      freqs_cos.nbytes != freqs_numel * sizeof(float) ||
      freqs_sin.nbytes != freqs_numel * sizeof(float) ||
      x_out.nbytes != geometry.xq_numel * sizeof(float) ||
      xk_out.nbytes != geometry.xk_numel * sizeof(float)) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: dtype/byte-size mismatch (all fp32) or "
        "freqs shape != [max_seq, rotary_dim]");
  }
  if (geometry.xq_numel > UINT32_MAX || geometry.xk_numel > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: element index exceeds uint32 range");
  }
  return geometry;
}

struct RotaryHfResizeContext {
  int xq_id;
  int xk_id;
  int xq_out_id;
  int xk_out_id;
  int start_pos_id;
  bool dynamic_pos;
  uint32_t baked_start_pos;
  uint32_t n_heads_q;
  uint32_t n_heads_k;
  uint32_t head_dim;
  uint32_t half_dim;
  uint32_t rotary_dim;
  uint32_t max_seq;
  uint32_t wg_size;
  size_t q_idx;
  size_t k_idx;
  WGPUBuffer q_ubuf;
  WGPUBuffer k_ubuf;
};

void resize_rope_hf(WebGPUGraph& graph, const RotaryHfResizeContext& context) {
  const auto& q_dims = graph.cur_dims(context.xq_id);
  const auto& k_dims = graph.cur_dims(context.xk_id);
  if (q_dims.size() < 3 || k_dims.size() < 3) {
    throw std::runtime_error(
        "apply_rotary_emb_hf(resize): q/k rank must be >= 3");
  }
  const uint32_t seq = static_cast<uint32_t>(q_dims[q_dims.size() - 3]);
  if (static_cast<uint32_t>(k_dims[k_dims.size() - 3]) != seq) {
    throw std::runtime_error(
        "apply_rotary_emb_hf(resize): q and k seq lengths differ");
  }
  if (static_cast<uint32_t>(q_dims.back()) != context.head_dim ||
      static_cast<uint32_t>(k_dims.back()) != context.head_dim ||
      static_cast<uint32_t>(q_dims[q_dims.size() - 2]) != context.n_heads_q ||
      static_cast<uint32_t>(k_dims[k_dims.size() - 2]) != context.n_heads_k) {
    throw std::runtime_error(
        "apply_rotary_emb_hf(resize): q/k head geometry changed");
  }
  const uint64_t q_numel = utils::numel_of(q_dims);
  const uint64_t k_numel = utils::numel_of(k_dims);
  uint32_t start_pos = context.baked_start_pos;
  if (context.dynamic_pos) {
    const int64_t pos = graph.read_symint(context.start_pos_id);
    if (pos < 0) {
      throw std::runtime_error(
          "apply_rotary_emb_hf(resize): start_pos must be non-negative");
    }
    start_pos = static_cast<uint32_t>(pos);
  }
  if (static_cast<uint64_t>(start_pos) + seq > context.max_seq) {
    throw std::runtime_error(
        "apply_rotary_emb_hf(resize): start_pos + seq exceeds freqs max_seq");
  }
  RotaryHfParams q_params = {};
  q_params.n_heads = context.n_heads_q;
  q_params.seq = seq;
  q_params.head_dim = context.head_dim;
  q_params.half_dim = context.half_dim;
  q_params.num_pairs = static_cast<uint32_t>(q_numel / 2u);
  q_params.rotary_dim = context.rotary_dim;
  q_params.start_pos = start_pos;
  RotaryHfParams k_params = q_params;
  k_params.n_heads = context.n_heads_k;
  k_params.num_pairs = static_cast<uint32_t>(k_numel / 2u);
  wgpuQueueWriteBuffer(
      graph.queue(), context.q_ubuf, 0, &q_params, sizeof(q_params));
  wgpuQueueWriteBuffer(
      graph.queue(), context.k_ubuf, 0, &k_params, sizeof(k_params));
  graph.dispatch_at(context.q_idx).workgroup_count_x =
      utils::compute_1d_workgroup_count(
          graph.device(),
          static_cast<uint32_t>(q_numel / 2u),
          context.wg_size,
          "rope_hf_q(resize)");
  graph.dispatch_at(context.k_idx).workgroup_count_x =
      utils::compute_1d_workgroup_count(
          graph.device(),
          static_cast<uint32_t>(k_numel / 2u),
          context.wg_size,
          "rope_hf_k(resize)");
  graph.set_cur_dims(context.xq_out_id, q_dims);
  graph.set_cur_dims(context.xk_out_id, k_dims);
}

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

  const auto& xq = graph.get_tensor(xq_id);
  const auto& xk = graph.get_tensor(xk_id);
  const auto& freqs_cos = graph.get_tensor(freqs_cos_id);
  const auto& freqs_sin = graph.get_tensor(freqs_sin_id);
  const auto& xq_out = graph.get_tensor(out_list[0]);
  const auto& xk_out = graph.get_tensor(out_list[1]);

  const RotaryHfGeometry geometry =
      validate_rope_hf_inputs(xq, xk, freqs_cos, freqs_sin, xq_out, xk_out);

  // Decode uses a SymInt position; static graphs use an Int.
  int64_t start_pos = 0;
  const auto start_pos_type = graph.get_value_type(start_pos_id);
  const bool dynamic_pos = start_pos_type == WebGPUGraph::ValueType::SymInt;
  if (dynamic_pos) {
    start_pos = graph.read_symint(start_pos_id);
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
  if (static_cast<uint64_t>(start_pos) + geometry.seq > geometry.max_seq) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: start_pos + seq exceeds freqs max_seq");
  }

  const uint32_t wg_size = utils::clamp_workgroup_size(
      graph.device(), get_webgpu_shader_info(kRotaryHfShader).workgroup_size_x);
  const uint32_t xq_wgc = utils::compute_1d_workgroup_count(
      graph.device(),
      static_cast<uint32_t>(geometry.xq_numel / 2u),
      wg_size,
      "apply_rotary_emb_hf");
  const uint32_t xk_wgc = utils::compute_1d_workgroup_count(
      graph.device(),
      static_cast<uint32_t>(geometry.xk_numel / 2u),
      wg_size,
      "apply_rotary_emb_hf");

  RotaryHfParams q_params = {};
  q_params.n_heads = geometry.n_heads_q;
  q_params.seq = geometry.seq;
  q_params.head_dim = geometry.head_dim;
  q_params.half_dim = geometry.half_dim;
  q_params.num_pairs = static_cast<uint32_t>(geometry.xq_numel / 2u);
  q_params.rotary_dim = geometry.rotary_dim;
  q_params.start_pos = static_cast<uint32_t>(start_pos);
  RotaryHfParams k_params = q_params;
  k_params.n_heads = geometry.n_heads_k;
  k_params.num_pairs = static_cast<uint32_t>(geometry.xk_numel / 2u);

  const RopeDispatch q_dispatch = add_rope_dispatch(
      graph,
      kRotaryHfShader,
      "apply_rotary_emb_hf",
      xq,
      xq_out,
      freqs_cos,
      freqs_sin,
      q_params,
      xq_wgc,
      wg_size);
  const RopeDispatch k_dispatch = add_rope_dispatch(
      graph,
      kRotaryHfShader,
      "apply_rotary_emb_hf",
      xk,
      xk_out,
      freqs_cos,
      freqs_sin,
      k_params,
      xk_wgc,
      wg_size);

  const RotaryHfResizeContext resize_context = {
      xq_id,
      xk_id,
      out_list[0],
      out_list[1],
      start_pos_id,
      dynamic_pos,
      static_cast<uint32_t>(start_pos),
      geometry.n_heads_q,
      geometry.n_heads_k,
      geometry.head_dim,
      geometry.half_dim,
      geometry.rotary_dim,
      geometry.max_seq,
      wg_size,
      q_dispatch.dispatch_index,
      k_dispatch.dispatch_index,
      q_dispatch.uniform,
      k_dispatch.uniform};
  auto rope_hf_hook = [resize_context](WebGPUGraph& g) {
    resize_rope_hf(g, resize_context);
  };
  graph.add_tensor_resize_hook(xq_id, rope_hf_hook);
  graph.add_tensor_resize_hook(xk_id, rope_hf_hook);
  if (dynamic_pos) {
    graph.add_resize_hook(start_pos_id, rope_hf_hook);
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.apply_rotary_emb.default, apply_rotary_emb_impl);
  WEBGPU_REGISTER_OP(
      et_vk.apply_rotary_emb_hf.default, apply_rotary_emb_hf_impl);
}

} // namespace executorch::backends::webgpu
