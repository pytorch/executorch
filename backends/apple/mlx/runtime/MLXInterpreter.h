//
// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#pragma once

#include "MLXExecutor.h"

#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/mlx.h>
#include <mlx/ops.h>

namespace executorch {
namespace backends {
namespace mlx {

// =============================================================================
// Op implementations
// =============================================================================

namespace ops {

using namespace ::mlx::core;

// ----- Helper to resolve int or Vid -----
inline int32_t resolve_int(
    const std::variant<int64_t, Vid<int32_t>>& v,
    const ExecutionState& st) {
  if (std::holds_alternative<int64_t>(v)) {
    return static_cast<int32_t>(std::get<int64_t>(v));
  }
  return st.const_value_ref<int32_t>(std::get<Vid<int32_t>>(v));
}

// ----- GELU implementation (tanh approximation) -----
inline array gelu_impl(const array& x, StreamOrDevice s = {}) {
  constexpr float sqrt_2_over_pi = 0.7978845608f;
  auto dtype = x.dtype();

  auto x3 = multiply(x, multiply(x, x, s), s);
  auto term = multiply(array(0.044715f, dtype), x3, s);
  auto inner = add(x, term, s);
  inner = multiply(array(sqrt_2_over_pi, dtype), inner, s);
  auto tanh_val = tanh(inner, s);
  auto one_plus_tanh = add(array(1.0f, dtype), tanh_val, s);
  auto out = multiply(x, one_plus_tanh, s);
  out = multiply(array(0.5f, dtype), out, s);
  return out;
}

// ----- Noop -----
inline void exec_noop(const NoopNode&, ExecutionState&, StreamOrDevice) {
  // Do nothing
}

// ----- Addmm -----
inline void
exec_addmm(const AddmmNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& mat1 = st.const_tensor_ref(n.mat1);
  const auto& mat2 = st.const_tensor_ref(n.mat2);

  array Y = matmul(mat1, mat2, s);

  if (n.bias) {
    const auto& b = st.const_tensor_ref(*n.bias);
    Y = add(Y, b, s);
  }

  st.set_tensor(n.out, std::move(Y));
}

// ----- Linear -----
inline void
exec_linear(const LinearNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& X = st.const_tensor_ref(n.x);
  auto W = st.const_tensor_ref(n.weight);
  W = transpose(W, {1, 0}, s);

  array Y = matmul(X, W, s);

  if (n.bias) {
    const auto& b = st.const_tensor_ref(*n.bias);
    Y = add(Y, b, s);
  }

  st.set_tensor(n.out, std::move(Y));
}

// ----- Item Int -----
inline void
exec_item_int(const ItemIntNode& n, ExecutionState& st, StreamOrDevice) {
  int item = st.const_tensor_ref(n.x).item<int>();
  st.set_value(n.out, item);
}

// ----- Expand Dims -----
inline void exec_expand_dims(
    const ExpandDimsNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  st.set_tensor(n.out, expand_dims(st.const_tensor_ref(n.x), n.axis, s));
}

// ----- Tile -----
inline void exec_tile(const TileNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, tile(st.const_tensor_ref(n.x), n.reps, s));
}

// ----- Take Along Axis -----
inline void exec_take_along_axis(
    const TakeAlongAxisNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  st.set_tensor(
      n.out,
      take_along_axis(
          st.const_tensor_ref(n.x), st.const_tensor_ref(n.indices), n.axis, s));
}

// ----- RMS Norm -----
inline void
exec_rms_norm(const RMSNormNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  const auto& w = st.const_tensor_ref(n.weight);
  st.set_tensor(n.out, fast::rms_norm(x, w, n.eps, s));
}

// ----- Layer Norm -----
inline void
exec_layer_norm(const LayerNormNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);

  std::optional<array> w = std::nullopt;
  if (n.weight) {
    w = st.const_tensor_ref(*n.weight);
  }
  std::optional<array> bias = std::nullopt;
  if (n.bias) {
    bias = st.const_tensor_ref(*n.bias);
  }
  st.set_tensor(n.out, fast::layer_norm(x, w, bias, n.eps, s));
}

// ----- RoPE -----
inline void exec_rope(const RopeNode& n, ExecutionState& st, StreamOrDevice s) {
  const array& x = st.const_tensor_ref(n.x);
  const int offset = st.const_value_ref<int32_t>(n.pos);

  std::optional<array> freqs_arr = std::nullopt;
  if (n.freqs) {
    freqs_arr = st.const_tensor_ref(*n.freqs);
  }

  array out = fast::rope(
      x, n.head_dim, n.traditional, n.base, n.scale, offset, freqs_arr, s);

  st.set_tensor(n.out, std::move(out));
}

// ----- SDPA -----
inline void exec_sdpa(const SdpaNode& n, ExecutionState& st, StreamOrDevice s) {
  array Q = st.const_tensor_ref(n.q);
  array K = st.const_tensor_ref(n.k);
  array V = st.const_tensor_ref(n.v);

  std::string mask_mode = "";
  std::optional<array> mask_arr = std::nullopt;
  std::optional<array> sinks = std::nullopt;

  if (n.mask) {
    array M = st.const_tensor_ref(*n.mask);
    if (M.dtype() != Q.dtype()) {
      M = astype(M, Q.dtype(), s);
    }
    mask_arr = std::move(M);
  }
  if (n.causal) {
    mask_mode = "causal";
  }

  array out = fast::scaled_dot_product_attention(
      Q, K, V, static_cast<float>(n.scale), mask_mode, mask_arr, sinks, s);
  st.set_tensor(n.out, std::move(out));
}

// ----- Add -----
inline void exec_add(const AddNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, add(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

// ----- Add Scalar -----
inline void
exec_add_scalar(const AddScalarNode& n, ExecutionState& st, StreamOrDevice) {
  int32_t a = resolve_int(n.a, st);
  int32_t b = resolve_int(n.b, st);
  st.set_value(n.out, a + b);
}

// ----- Sym Size -----
inline void
exec_sym_size(const SymSizeNode& n, ExecutionState& st, StreamOrDevice) {
  const array& a = st.const_tensor_ref(n.a);
  int rank = static_cast<int>(a.ndim());
  int dim = n.dim;
  if (dim < 0) {
    dim += rank;
  }
  if (dim < 0 || dim >= rank) {
    throw std::out_of_range("SYM_SIZE: dim out of range");
  }
  int32_t size = static_cast<int32_t>(a.shape()[dim]);
  st.set_value(n.out, size);
}

// ----- Mul -----
inline void exec_mul(const MulNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, multiply(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

// ----- Conv1D -----
inline void
exec_conv1d(const Conv1DNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  const auto& w = st.const_tensor_ref(n.w);
  auto out = conv1d(x, w, n.stride, n.padding, n.dilation, n.groups, s);
  st.set_tensor(n.out, std::move(out));
}

// ----- GELU -----
inline void exec_gelu(const GeluNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  st.set_tensor(n.out, gelu_impl(x, s));
}

// ----- ARange -----
inline void
exec_arange(const ARangeNode& n, ExecutionState& st, StreamOrDevice s) {
  // Get start, stop, step - may be literal int64 or dynamic Vid
  int start_val = resolve_int(n.start, st);
  int stop_val = resolve_int(n.stop, st);
  int step_val = resolve_int(n.step, st);

  st.set_tensor(
      n.out, arange(start_val, stop_val, step_val, to_mlx_dtype(n.dtype), s));
}

// ----- SiLU -----
inline void exec_silu(const SiluNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  st.set_tensor(n.out, multiply(x, sigmoid(x, s), s));
}

// ----- Reshape -----
inline void
exec_reshape(const ReshapeNode& n, ExecutionState& st, StreamOrDevice) {
  auto new_shape = to_shape(n.shape, st);
  st.set_tensor(n.out, reshape(st.const_tensor_ref(n.x), new_shape));
}

// ----- Transpose -----
inline void
exec_transpose(const TransposeNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, transpose(st.const_tensor_ref(n.x), n.perm, s));
}

// ----- Contiguous -----
inline void
exec_contiguous(const ContiguousNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, contiguous(st.const_tensor_ref(n.x), false, s));
}

// ----- Id Copy -----
inline void
exec_id_copy(const IdCopyNode& n, ExecutionState& st, StreamOrDevice) {
  st.set_tensor(n.out, st.const_tensor_ref(n.x));
}

// ----- Gather -----
inline void
exec_gather(const GatherNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out,
      take(st.const_tensor_ref(n.table_), st.const_tensor_ref(n.ids), 0, s));
}

// ----- Slice -----
inline void
exec_slice(const SliceNode& n, ExecutionState& st, StreamOrDevice s) {
  const array& x = st.const_tensor_ref(n.x);
  const int rank = static_cast<int>(x.ndim());

  int axis = resolve_int(n.axis, st);
  int start = resolve_int(n.start, st);
  int stop = resolve_int(n.stop, st);

  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank) {
    throw std::out_of_range("Slice: axis out of range");
  }

  std::vector<int> vstart(rank, 0);
  std::vector<int> vstop;
  vstop.reserve(rank);
  auto sh = x.shape();
  for (int i = 0; i < rank; ++i) {
    vstop.push_back(static_cast<int>(sh[i]));
  }

  const int dim = vstop[axis];
  if (start < 0)
    start += dim;
  start = std::max(0, std::min(start, dim));
  if (stop < 0)
    stop += dim;
  stop = std::max(0, std::min(stop, dim));

  vstart[axis] = start;
  vstop[axis] = stop;

  st.set_tensor(n.out, slice(x, to_shape(vstart), to_shape(vstop), s));
}

// ----- Cast -----
inline void exec_cast(const CastNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, astype(st.const_tensor_ref(n.x), to_mlx_dtype(n.dtype), s));
}

// ----- Quantized Linear -----
inline void exec_quantized_linear(
    const QuantizedLinearNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  array X = st.const_tensor_ref(n.x);
  array Wq = st.const_tensor_ref(n.w);
  array Sc = st.const_tensor_ref(n.scales);

  std::optional<array> Qb = std::nullopt;
  if (n.biases) {
    Qb = st.const_tensor_ref(*n.biases);
  }

  array Y = quantized_matmul(
      X,
      Wq,
      Sc,
      Qb,
      /*transpose=*/true,
      n.group_size,
      n.bits,
      n.mode,
      s);

  if (n.bias) {
    const auto& b = st.const_tensor_ref(*n.bias);
    Y = add(Y, b, s);
  }

  if (to_mlx_dtype(n.out_dtype) != Y.dtype()) {
    Y = astype(Y, to_mlx_dtype(n.out_dtype), s);
  }

  st.set_tensor(n.out, std::move(Y));
}

// ----- Concat -----
inline void
exec_concat(const ConcatNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out,
      concatenate(
          {st.const_tensor_ref(n.a), st.const_tensor_ref(n.b)}, n.axis, s));
}

// ----- Full -----
inline void exec_full(const FullNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, full(to_shape(n.shape), n.v, to_mlx_dtype(n.dtype), s));
}

// ----- Zeros -----
inline void
exec_zeros(const ZerosNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, zeros(to_shape(n.shape), to_mlx_dtype(n.dtype), s));
}

// ----- Ones -----
inline void exec_ones(const OnesNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, ones(to_shape(n.shape), to_mlx_dtype(n.dtype), s));
}

// ----- Argmax -----
inline void
exec_argmax(const ArgmaxNode& n, ExecutionState& st, StreamOrDevice s) {
  array idx = argmax(st.const_tensor_ref(n.x), n.axis, s);
  st.set_tensor(n.out, std::move(idx));
}

// ----- Slice Update -----
inline void exec_slice_update(
    const SliceUpdateNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  array& dst = st.tensor_ref(n.dst);
  const array& upd = st.const_tensor_ref(n.update);

  const int rank = static_cast<int>(dst.ndim());

  int axis = resolve_int(n.axis, st);
  int start = resolve_int(n.start, st);
  int stop = resolve_int(n.stop, st);

  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank) {
    throw std::out_of_range("SliceUpdate: axis out of range");
  }

  std::vector<int> vstart(rank, 0);
  std::vector<int> vstop;
  vstop.reserve(rank);
  auto sh = dst.shape();
  for (int i = 0; i < rank; ++i) {
    vstop.push_back(static_cast<int>(sh[i]));
  }

  const int dst_dim = vstop[axis];

  if (start < 0)
    start += dst_dim;
  start = std::max(0, std::min(start, dst_dim));
  if (stop < 0)
    stop += dst_dim;
  stop = std::max(0, std::min(stop, dst_dim));

  vstart[axis] = start;
  vstop[axis] = stop;

  dst = slice_update(dst, upd, to_shape(vstart), to_shape(vstop), s);
}

// ----- Quantized Gather -----
inline void exec_quantized_gather(
    const QuantizedGatherNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  array ids = st.const_tensor_ref(n.ids);
  array Wq = st.const_tensor_ref(n.table_q);
  array Sc = st.const_tensor_ref(n.scales);

  std::optional<array> Qb = std::nullopt;
  if (n.biases) {
    Qb = st.const_tensor_ref(*n.biases);
  }

  array Wq_sel = take(Wq, ids, 0, s);
  array Sc_sel = take(Sc, ids, 0, s);
  std::optional<array> Qb_sel = std::nullopt;
  if (Qb) {
    Qb_sel = take(*Qb, ids, 0, s);
  }

  array Y = dequantize(
      Wq_sel,
      Sc_sel,
      Qb_sel,
      n.group_size,
      n.bits,
      n.mode,
      std::nullopt, // dtype - let MLX infer
      s);

  if (to_mlx_dtype(n.out_dtype) != Y.dtype()) {
    Y = astype(Y, to_mlx_dtype(n.out_dtype), s);
  }

  st.set_tensor(n.out, std::move(Y));
}

} // namespace ops

// =============================================================================
// Interpreter - dispatch loop
// =============================================================================

class Interpreter {
 public:
  void run(
      const MLXProgram& prog,
      ExecutionState& st,
      StreamOrDevice stream = {}) const {
    size_t idx = 0;
    for (const auto& instr : prog.instructions) {
      st.begin_op(idx, op_name(instr.op));
      dispatch(instr, st, stream);
      st.end_op();
      ++idx;
    }
  }

 private:
  void dispatch(const Instruction& instr, ExecutionState& st, StreamOrDevice s)
      const {
    switch (instr.op) {
      case OpCode::NOOP:
        ops::exec_noop(std::get<NoopNode>(instr.node), st, s);
        break;
      case OpCode::ADDMM:
        ops::exec_addmm(std::get<AddmmNode>(instr.node), st, s);
        break;
      case OpCode::LINEAR:
        ops::exec_linear(std::get<LinearNode>(instr.node), st, s);
        break;
      case OpCode::ITEM_INT:
        ops::exec_item_int(std::get<ItemIntNode>(instr.node), st, s);
        break;
      case OpCode::EXPAND_DIMS:
        ops::exec_expand_dims(std::get<ExpandDimsNode>(instr.node), st, s);
        break;
      case OpCode::TILE:
        ops::exec_tile(std::get<TileNode>(instr.node), st, s);
        break;
      case OpCode::TAKE_ALONG_AXIS:
        ops::exec_take_along_axis(
            std::get<TakeAlongAxisNode>(instr.node), st, s);
        break;
      case OpCode::RMS_NORM:
        ops::exec_rms_norm(std::get<RMSNormNode>(instr.node), st, s);
        break;
      case OpCode::LAYER_NORM:
        ops::exec_layer_norm(std::get<LayerNormNode>(instr.node), st, s);
        break;
      case OpCode::ROPE:
        ops::exec_rope(std::get<RopeNode>(instr.node), st, s);
        break;
      case OpCode::SDPA:
        ops::exec_sdpa(std::get<SdpaNode>(instr.node), st, s);
        break;
      case OpCode::ADD:
        ops::exec_add(std::get<AddNode>(instr.node), st, s);
        break;
      case OpCode::ADD_SCALAR:
        ops::exec_add_scalar(std::get<AddScalarNode>(instr.node), st, s);
        break;
      case OpCode::SYM_SIZE:
        ops::exec_sym_size(std::get<SymSizeNode>(instr.node), st, s);
        break;
      case OpCode::MUL:
        ops::exec_mul(std::get<MulNode>(instr.node), st, s);
        break;
      case OpCode::CONV1D:
        ops::exec_conv1d(std::get<Conv1DNode>(instr.node), st, s);
        break;
      case OpCode::GELU:
        ops::exec_gelu(std::get<GeluNode>(instr.node), st, s);
        break;
      case OpCode::ARANGE:
        ops::exec_arange(std::get<ARangeNode>(instr.node), st, s);
        break;
      case OpCode::SILU:
        ops::exec_silu(std::get<SiluNode>(instr.node), st, s);
        break;
      case OpCode::RESHAPE:
        ops::exec_reshape(std::get<ReshapeNode>(instr.node), st, s);
        break;
      case OpCode::TRANSPOSE:
        ops::exec_transpose(std::get<TransposeNode>(instr.node), st, s);
        break;
      case OpCode::CONTIGUOUS:
        ops::exec_contiguous(std::get<ContiguousNode>(instr.node), st, s);
        break;
      case OpCode::ID_COPY:
        ops::exec_id_copy(std::get<IdCopyNode>(instr.node), st, s);
        break;
      case OpCode::GATHER:
        ops::exec_gather(std::get<GatherNode>(instr.node), st, s);
        break;
      case OpCode::SLICE:
        ops::exec_slice(std::get<SliceNode>(instr.node), st, s);
        break;
      case OpCode::CAST:
        ops::exec_cast(std::get<CastNode>(instr.node), st, s);
        break;
      case OpCode::QUANTIZED_LINEAR:
        ops::exec_quantized_linear(
            std::get<QuantizedLinearNode>(instr.node), st, s);
        break;
      case OpCode::CONCAT:
        ops::exec_concat(std::get<ConcatNode>(instr.node), st, s);
        break;
      case OpCode::FULL:
        ops::exec_full(std::get<FullNode>(instr.node), st, s);
        break;
      case OpCode::ZEROS:
        ops::exec_zeros(std::get<ZerosNode>(instr.node), st, s);
        break;
      case OpCode::ONES:
        ops::exec_ones(std::get<OnesNode>(instr.node), st, s);
        break;
      case OpCode::ARGMAX:
        ops::exec_argmax(std::get<ArgmaxNode>(instr.node), st, s);
        break;
      case OpCode::SLICE_UPDATE:
        ops::exec_slice_update(std::get<SliceUpdateNode>(instr.node), st, s);
        break;
      case OpCode::QUANTIZED_GATHER:
        ops::exec_quantized_gather(
            std::get<QuantizedGatherNode>(instr.node), st, s);
        break;
      case OpCode::SENTINEL:
        break;
    }
  }
};

} // namespace mlx
} // namespace backends
} // namespace executorch
