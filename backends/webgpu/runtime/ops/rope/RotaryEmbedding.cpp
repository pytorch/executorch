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

enum class RopeGridPolicy {
  OneDimensional,
  FoldedTwoDimensional,
};

struct RopeGridContext {
  int tensor_id;
  uint32_t workgroup_size;
  RopeGridPolicy policy;
  const char* op_name;
};

WebGPUDispatchGrid pick_rope_grid(
    const WebGPUGraph& graph,
    const RopeGridContext& context) {
  const uint32_t num_pairs = static_cast<uint32_t>(
      utils::numel_of(graph.cur_dims(context.tensor_id)) / 2u);
  if (context.policy == RopeGridPolicy::OneDimensional) {
    return {
        utils::compute_1d_workgroup_count(
            graph.device(), num_pairs, context.workgroup_size, context.op_name),
        1u};
  }
  const utils::WgCount grid = utils::compute_2d_workgroup_count(
      graph.device(), num_pairs, context.workgroup_size, context.op_name);
  return {grid.x, grid.y};
}

void preflight_rope_grids(
    const WebGPUGraph& graph,
    const RopeGridContext& q_context,
    const RopeGridContext& k_context) {
  (void)pick_rope_grid(graph, q_context);
  (void)pick_rope_grid(graph, k_context);
}

template <typename Params>
WGPUBuffer add_rope_dispatch(
    WebGPUGraph& graph,
    const char* shader_name,
    const char* kernel_name,
    const WebGPUTensor& x,
    const WebGPUTensor& out,
    const WebGPUTensor& freqs_cos,
    const WebGPUTensor& freqs_sin,
    const Params& params,
    int trigger_tensor_id,
    const RopeGridContext& grid_context,
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
  graph.add_dynamic_compute_dispatch(
      descriptor, trigger_tensor_id, pick_rope_grid, grid_context);
  return uniform_buffer;
}

struct RotaryGeometry {
  uint32_t head_dim;
  uint32_t seq;
  uint32_t n_heads_q;
  uint32_t n_heads_k;
  uint32_t half_dim;
  uint64_t xq_numel;
  uint64_t xk_numel;
};

RotaryGeometry validate_rope_inputs(
    const WebGPUTensor& xq,
    const WebGPUTensor& xk,
    const WebGPUTensor& freqs_cos,
    const WebGPUTensor& freqs_sin,
    const WebGPUTensor& xq_out,
    const WebGPUTensor& xk_out) {
  if (xq.dims.size() < 3 || xk.dims.size() < 3 || freqs_cos.dims.size() < 2) {
    throw std::runtime_error("WebGPU apply_rotary_emb: malformed dims");
  }
  RotaryGeometry geometry = {};
  geometry.head_dim = static_cast<uint32_t>(xq.dims.back());
  geometry.seq = static_cast<uint32_t>(xq.dims[xq.dims.size() - 3]);
  geometry.n_heads_q = static_cast<uint32_t>(xq.dims[xq.dims.size() - 2]);
  geometry.n_heads_k = static_cast<uint32_t>(xk.dims[xk.dims.size() - 2]);
  const uint32_t seq_k = static_cast<uint32_t>(xk.dims[xk.dims.size() - 3]);
  geometry.half_dim = static_cast<uint32_t>(freqs_cos.dims.back());

  if (geometry.head_dim == 0 || geometry.head_dim % 2 != 0) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: head_dim must be a nonzero multiple of 2");
  }
  if (static_cast<uint32_t>(xk.dims.back()) != geometry.head_dim ||
      seq_k != geometry.seq) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: xq/xk head_dim and seq must match");
  }
  if (geometry.half_dim * 2u != geometry.head_dim) {
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

  geometry.xq_numel = utils::numel_of(xq.dims);
  geometry.xk_numel = utils::numel_of(xk.dims);
  const uint64_t freqs_numel = utils::numel_of(freqs_cos.dims);
  if (freqs_numel != static_cast<uint64_t>(geometry.seq) * geometry.half_dim ||
      xq.nbytes != geometry.xq_numel * sizeof(float) ||
      xk.nbytes != geometry.xk_numel * sizeof(float) ||
      freqs_cos.nbytes != freqs_numel * sizeof(float) ||
      freqs_sin.nbytes != freqs_numel * sizeof(float) ||
      xq_out.nbytes != geometry.xq_numel * sizeof(float) ||
      xk_out.nbytes != geometry.xk_numel * sizeof(float)) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: dtype/byte-size mismatch (all fp32) or "
        "freqs shape != [seq, head_dim/2]");
  }
  if (geometry.xq_numel > UINT32_MAX || geometry.xk_numel > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb: element index exceeds uint32 range");
  }
  return geometry;
}

struct RotaryResizeContext {
  int xq_id;
  int xk_id;
  int xq_out_id;
  int xk_out_id;
  uint32_t n_heads_q;
  uint32_t n_heads_k;
  uint32_t head_dim;
  uint32_t half_dim;
  WGPUBuffer q_uniform;
  WGPUBuffer k_uniform;
};

// Resize hook body: update parameters and outputs; the graph owns grid refresh.
void resize_rope(WebGPUGraph& graph, const RotaryResizeContext& context) {
  const auto& q_dims = graph.cur_dims(context.xq_id);
  const auto& k_dims = graph.cur_dims(context.xk_id);
  if (q_dims.size() < 3 || k_dims.size() < 3) {
    throw std::runtime_error("apply_rotary_emb(resize): q/k rank must be >= 3");
  }
  const uint32_t seq = static_cast<uint32_t>(q_dims[q_dims.size() - 3]);
  const uint64_t q_numel = utils::numel_of(q_dims);
  const uint64_t k_numel = utils::numel_of(k_dims);
  // pk = pq (seq=s); require k's seq == s, not silently q's.
  if (static_cast<uint32_t>(k_dims[k_dims.size() - 3]) != seq) {
    throw std::runtime_error(
        "apply_rotary_emb(resize): q and k seq lengths differ");
  }
  // freqs stay max-allocated; shader indexes by position (S = prefix).
  RotaryParams q_params = {};
  q_params.n_heads = context.n_heads_q;
  q_params.seq = seq;
  q_params.head_dim = context.head_dim;
  q_params.half_dim = context.half_dim;
  q_params.num_pairs = static_cast<uint32_t>(q_numel / 2u);
  RotaryParams k_params = q_params;
  k_params.n_heads = context.n_heads_k;
  k_params.num_pairs = static_cast<uint32_t>(k_numel / 2u);
  wgpuQueueWriteBuffer(
      graph.queue(), context.q_uniform, 0, &q_params, sizeof(q_params));
  wgpuQueueWriteBuffer(
      graph.queue(), context.k_uniform, 0, &k_params, sizeof(k_params));
  graph.set_cur_dims(context.xq_out_id, q_dims);
  graph.set_cur_dims(context.xk_out_id, k_dims);
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

  const RotaryGeometry geometry =
      validate_rope_inputs(xq, xk, freqs_cos, freqs_sin, xq_out, xk_out);

  const uint32_t wg_size = utils::clamp_workgroup_size(
      graph.device(), get_webgpu_shader_info(kRotaryShader).workgroup_size_x);
  const RopeGridContext q_grid = {
      xq_id, wg_size, RopeGridPolicy::OneDimensional, "apply_rotary_emb"};
  const RopeGridContext k_grid = {
      xk_id, wg_size, RopeGridPolicy::OneDimensional, "apply_rotary_emb"};
  preflight_rope_grids(graph, q_grid, k_grid);

  RotaryParams q_params = {};
  q_params.n_heads = geometry.n_heads_q;
  q_params.seq = geometry.seq;
  q_params.head_dim = geometry.head_dim;
  q_params.half_dim = geometry.half_dim;
  q_params.num_pairs = static_cast<uint32_t>(geometry.xq_numel / 2u);
  RotaryParams k_params = q_params;
  k_params.n_heads = geometry.n_heads_k;
  k_params.num_pairs = static_cast<uint32_t>(geometry.xk_numel / 2u);
  const WGPUBuffer q_uniform = add_rope_dispatch(
      graph,
      kRotaryShader,
      "apply_rotary_emb",
      xq,
      xq_out,
      freqs_cos,
      freqs_sin,
      q_params,
      xq_id,
      q_grid,
      wg_size);
  const WGPUBuffer k_uniform = add_rope_dispatch(
      graph,
      kRotaryShader,
      "apply_rotary_emb",
      xk,
      xk_out,
      freqs_cos,
      freqs_sin,
      k_params,
      xk_id,
      k_grid,
      wg_size);

  // Register on both xq and xk so the recompute fires whichever is marked dirty
  // (q and k co-resize on S; resize_rope is idempotent, so a double-fire when
  // both are dirty is harmless).
  const RotaryResizeContext resize_context = {
      xq_id,
      xk_id,
      out_list[0],
      out_list[1],
      geometry.n_heads_q,
      geometry.n_heads_k,
      geometry.head_dim,
      geometry.half_dim,
      q_uniform,
      k_uniform};
  graph.add_tensor_resize_hook(xq_id, resize_rope, resize_context);
  graph.add_tensor_resize_hook(xk_id, resize_rope, resize_context);
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
  if (x.dims.size() < 3 || xk.dims.size() != x.dims.size() ||
      freqs_cos.dims.size() != 2) {
    throw std::runtime_error("WebGPU apply_rotary_emb_hf: malformed dims");
  }
  if (x_out.dims != x.dims || xk_out.dims != xk.dims) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: output shapes must match q/k inputs");
  }
  for (size_t i = 0; i + 3 < x.dims.size(); i++) {
    if (x.dims[i] != xk.dims[i]) {
      throw std::runtime_error(
          "WebGPU apply_rotary_emb_hf: q/k batch dimensions differ");
    }
  }
  const auto positive_u32 = [](int64_t value, const char* label) {
    if (value <= 0 || static_cast<uint64_t>(value) > UINT32_MAX) {
      throw std::runtime_error(
          std::string("WebGPU apply_rotary_emb_hf: invalid ") + label);
    }
    return static_cast<uint32_t>(value);
  };
  RotaryHfGeometry geometry = {};
  geometry.head_dim = positive_u32(x.dims.back(), "head_dim");
  geometry.seq = positive_u32(x.dims[x.dims.size() - 3], "sequence length");
  geometry.n_heads_q =
      positive_u32(x.dims[x.dims.size() - 2], "query head count");
  geometry.n_heads_k =
      positive_u32(xk.dims[xk.dims.size() - 2], "key head count");
  geometry.max_seq = positive_u32(freqs_cos.dims[0], "frequency row count");
  geometry.rotary_dim = positive_u32(freqs_cos.dims[1], "rotary_dim");
  if (geometry.head_dim % 2 != 0) {
    throw std::runtime_error(
        "WebGPU apply_rotary_emb_hf: head_dim must be a nonzero multiple of 2");
  }
  if (xk.dims.back() != static_cast<int64_t>(geometry.head_dim) ||
      xk.dims[xk.dims.size() - 3] != static_cast<int64_t>(geometry.seq)) {
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
  const WebGPUTensor* tensors[] = {
      &x, &xk, &freqs_cos, &freqs_sin, &x_out, &xk_out};
  for (const WebGPUTensor* tensor : tensors) {
    if (tensor->is_int || tensor->elem_size != sizeof(float)) {
      throw std::runtime_error(
          "WebGPU apply_rotary_emb_hf: all tensors must be fp32");
    }
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
  if (geometry.xq_numel == 0 || geometry.xk_numel == 0 ||
      geometry.xq_numel > UINT32_MAX || geometry.xk_numel > UINT32_MAX) {
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
  WGPUBuffer q_uniform;
  WGPUBuffer k_uniform;
};

void resize_rope_hf(WebGPUGraph& graph, const RotaryHfResizeContext& context) {
  const auto& q_dims = graph.cur_dims(context.xq_id);
  const auto& k_dims = graph.cur_dims(context.xk_id);
  if (q_dims.size() < 3 || k_dims.size() != q_dims.size()) {
    throw std::runtime_error(
        "apply_rotary_emb_hf(resize): q/k rank must be >= 3");
  }
  const int64_t seq_value = q_dims[q_dims.size() - 3];
  if (seq_value <= 0 || static_cast<uint64_t>(seq_value) > UINT32_MAX) {
    throw std::runtime_error(
        "apply_rotary_emb_hf(resize): invalid sequence length");
  }
  const uint32_t seq = static_cast<uint32_t>(seq_value);
  if (k_dims[k_dims.size() - 3] != seq_value) {
    throw std::runtime_error(
        "apply_rotary_emb_hf(resize): q and k seq lengths differ");
  }
  if (q_dims.back() != static_cast<int64_t>(context.head_dim) ||
      k_dims.back() != static_cast<int64_t>(context.head_dim) ||
      q_dims[q_dims.size() - 2] != static_cast<int64_t>(context.n_heads_q) ||
      k_dims[k_dims.size() - 2] != static_cast<int64_t>(context.n_heads_k)) {
    throw std::runtime_error(
        "apply_rotary_emb_hf(resize): q/k head geometry changed");
  }
  for (size_t i = 0; i + 3 < q_dims.size(); i++) {
    if (q_dims[i] != k_dims[i]) {
      throw std::runtime_error(
          "apply_rotary_emb_hf(resize): q/k batch dimensions differ");
    }
  }
  const uint64_t q_numel = utils::numel_of(q_dims);
  const uint64_t k_numel = utils::numel_of(k_dims);
  if (q_numel == 0 || k_numel == 0 || q_numel > UINT32_MAX ||
      k_numel > UINT32_MAX) {
    throw std::runtime_error(
        "apply_rotary_emb_hf(resize): element index exceeds uint32 range");
  }
  uint32_t start_pos = context.baked_start_pos;
  if (context.dynamic_pos) {
    const int64_t pos = graph.read_symint(context.start_pos_id);
    if (pos < 0 || static_cast<uint64_t>(pos) > UINT32_MAX) {
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
      graph.queue(), context.q_uniform, 0, &q_params, sizeof(q_params));
  wgpuQueueWriteBuffer(
      graph.queue(), context.k_uniform, 0, &k_params, sizeof(k_params));
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
  const RopeGridContext q_grid = {
      xq_id,
      wg_size,
      RopeGridPolicy::FoldedTwoDimensional,
      "apply_rotary_emb_hf"};
  const RopeGridContext k_grid = {
      xk_id,
      wg_size,
      RopeGridPolicy::FoldedTwoDimensional,
      "apply_rotary_emb_hf"};
  preflight_rope_grids(graph, q_grid, k_grid);

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

  const WGPUBuffer q_uniform = add_rope_dispatch(
      graph,
      kRotaryHfShader,
      "apply_rotary_emb_hf",
      xq,
      xq_out,
      freqs_cos,
      freqs_sin,
      q_params,
      xq_id,
      q_grid,
      wg_size);
  const WGPUBuffer k_uniform = add_rope_dispatch(
      graph,
      kRotaryHfShader,
      "apply_rotary_emb_hf",
      xk,
      xk_out,
      freqs_cos,
      freqs_sin,
      k_params,
      xk_id,
      k_grid,
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
      q_uniform,
      k_uniform};
  graph.add_tensor_resize_hook(xq_id, resize_rope_hf, resize_context);
  graph.add_tensor_resize_hook(xk_id, resize_rope_hf, resize_context);
  if (dynamic_pos) {
    graph.add_resize_hook(start_pos_id, resize_rope_hf, resize_context);
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.apply_rotary_emb.default, apply_rotary_emb_impl);
  WEBGPU_REGISTER_OP(
      et_vk.apply_rotary_emb_hf.default, apply_rotary_emb_hf_impl);
}

} // namespace executorch::backends::webgpu
