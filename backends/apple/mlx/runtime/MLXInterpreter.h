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

// ----- Utility: Infer -1 dimensions in shape -----
/**
 * Infer -1 dimensions in a shape based on input tensor size.
 *
 * PyTorch allows -1 in shapes to mean "infer this dimension from total size".
 * MLX requires concrete positive integers, so we must resolve -1 values.
 *
 * @param shape The shape to resolve (may contain -1)
 * @param input_size Total number of elements in the input tensor
 * @return Resolved shape with all positive integers
 * @throws std::runtime_error if shape has multiple -1 or incompatible sizes
 */
inline std::vector<int> infer_shape_with_minus_one(
    const std::vector<int>& shape,
    size_t input_size) {
  std::vector<int> resolved_shape = shape;
  int neg_one_idx = -1;
  int64_t known_size = 1; // Use int64_t to avoid overflow

  // Find -1 dimension and compute product of known dimensions
  for (size_t i = 0; i < resolved_shape.size(); i++) {
    if (resolved_shape[i] == -1) {
      if (neg_one_idx != -1) {
        throw std::runtime_error("infer_shape: only one dimension can be -1");
      }
      neg_one_idx = static_cast<int>(i);
    } else {
      known_size *= static_cast<int64_t>(resolved_shape[i]);
    }
  }

  // Infer the -1 dimension if present
  if (neg_one_idx != -1) {
    int64_t input_size_i64 = static_cast<int64_t>(input_size);
    if (input_size_i64 % known_size != 0) {
      throw std::runtime_error(
          "infer_shape: cannot infer dimension - size mismatch");
    }
    int64_t inferred_dim = input_size_i64 / known_size;

    // Check that inferred dimension fits in int
    if (inferred_dim > std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          "infer_shape: inferred dimension exceeds int max");
    }

    resolved_shape[neg_one_idx] = static_cast<int>(inferred_dim);
  }

  return resolved_shape;
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

  array Y = n.bias ? addmm(
                         st.const_tensor_ref(*n.bias),
                         mat1,
                         mat2,
                         /*alpha=*/n.alpha,
                         /*beta=*/n.beta,
                         s)
                   : matmul(mat1, mat2, s);

  st.set_tensor(n.out, std::move(Y));
}

// ----- Linear -----
inline void
exec_linear(const LinearNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& X = st.const_tensor_ref(n.x);
  auto W = st.const_tensor_ref(n.weight);
  W = transpose(W, {1, 0}, s);

  array Y = n.bias ? addmm(
                         st.const_tensor_ref(*n.bias),
                         X,
                         W,
                         /*alpha=*/1.0f,
                         /*beta=*/1.0f,
                         s)
                   : matmul(X, W, s);

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
  const auto& x = st.const_tensor_ref(n.x);
  auto reps = resolve_ints(n.reps, st);
  st.set_tensor(n.out, tile(x, reps, s));
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

// ----- Multiply -----
inline void
exec_multiply(const MultiplyNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, multiply(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

// ----- Divide -----
inline void
exec_divide(const DivideNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, divide(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

// ----- Subtract -----
inline void
exec_subtract(const SubtractNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, subtract(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

// ----- Conv1D -----
inline void
exec_conv1d(const Conv1DNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  const auto& w = st.const_tensor_ref(n.w);
  auto out = conv1d(x, w, n.stride, n.padding, n.dilation, n.groups, s);
  st.set_tensor(n.out, std::move(out));
}

// ----- Conv2D -----
inline void
exec_conv2d(const Conv2DNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  const auto& w = st.const_tensor_ref(n.w);

  std::pair<int, int> stride = {n.stride_h, n.stride_w};
  std::pair<int, int> padding = {n.padding_h, n.padding_w};
  std::pair<int, int> dilation = {n.dilation_h, n.dilation_w};

  auto out = conv2d(x, w, stride, padding, dilation, n.groups, s);
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

// ----- Sigmoid -----
inline void
exec_sigmoid(const SigmoidNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  st.set_tensor(n.out, sigmoid(x, s));
}

// ----- Tanh -----
inline void exec_tanh(const TanhNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  st.set_tensor(n.out, tanh(x, s));
}

// ----- Squeeze -----
inline void
exec_squeeze(const SqueezeNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  auto dims_fb = n.dims;

  if (dims_fb.size() > 0) {
    // Squeeze specific dimensions
    std::vector<int> dims;
    for (auto d : dims_fb) {
      dims.push_back(d);
    }
    st.set_tensor(n.out, squeeze(x, dims, s));
  } else {
    // Squeeze all dimensions of size 1
    st.set_tensor(n.out, squeeze(x, s));
  }
}

// ----- Split -----
inline void
exec_split(const SplitNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);

  // Resolve dynamic sizes to std::vector<int>
  std::vector<int32_t> sizes_vec = resolve_ints(n.sizes, st);

  // Get results based on split mode
  auto outs_fb = n.outs;

  if (sizes_vec.size() == 1) {
    // Single value means split_size (chunk size)
    // Compute actual sizes: e.g., dim_size=10, split_size=3 -> [3, 3, 3, 1]
    int split_size = sizes_vec[0];
    int axis = n.axis < 0 ? n.axis + x.ndim() : n.axis;
    int dim_size = x.shape(axis);

    std::vector<int> indices;
    for (int pos = split_size; pos < dim_size; pos += split_size) {
      indices.push_back(pos);
    }

    auto results = split(x, to_shape(indices), n.axis, s);
    if (results.size() != outs_fb.size()) {
      throw std::runtime_error("Split: output count mismatch");
    }
    for (size_t i = 0; i < results.size(); ++i) {
      st.set_tensor(outs_fb[i], std::move(results[i]));
    }
  } else {
    // Multiple sizes: convert to cumulative indices for MLX
    // sizes=[10, 20, 30] -> indices=[10, 30] (split at positions 10 and 30)
    std::vector<int> indices;
    indices.reserve(sizes_vec.size() - 1);
    int cumsum = 0;
    for (size_t i = 0; i < sizes_vec.size() - 1; ++i) {
      cumsum += sizes_vec[i];
      indices.push_back(cumsum);
    }
    auto results = split(x, to_shape(indices), n.axis, s);
    if (results.size() != outs_fb.size()) {
      throw std::runtime_error("Split: output count mismatch");
    }
    for (size_t i = 0; i < results.size(); ++i) {
      st.set_tensor(outs_fb[i], std::move(results[i]));
    }
  }
}

// ----- Rsqrt -----
inline void
exec_rsqrt(const RsqrtNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  st.set_tensor(n.out, rsqrt(x, s));
}

// ----- Maximum -----
inline void
exec_maximum(const MaximumNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, maximum(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

// ----- Log -----
inline void exec_log(const LogNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  st.set_tensor(n.out, log(x, s));
}

// ----- Softmax -----
inline void
exec_softmax(const SoftmaxNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  st.set_tensor(n.out, softmax(x, n.axis, /*precise=*/false, s));
}

// ----- BroadcastTo -----
inline void exec_broadcast_to(
    const BroadcastToNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  auto shape_vec = resolve_ints(n.shape, st);
  auto inferred_shape = infer_shape_with_minus_one(shape_vec, x.size());
  st.set_tensor(
      n.out,
      broadcast_to(
          x,
          ::mlx::core::Shape(inferred_shape.begin(), inferred_shape.end()),
          s));
}

// ----- Pad -----
inline void exec_pad(const PadNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);

  // Convert flat pad_width to vector of pairs
  std::vector<std::pair<int, int>> pad_width_pairs;
  for (size_t i = 0; i < n.pad_width.size(); i += 2) {
    pad_width_pairs.push_back({n.pad_width[i], n.pad_width[i + 1]});
  }

  // MLX pad signature: pad(array, pad_width, pad_value, mode, stream)
  if (n.mode == "constant") {
    array pad_value(n.constant_value);
    st.set_tensor(n.out, pad(x, pad_width_pairs, pad_value, "constant", s));
  } else if (n.mode == "edge") {
    array pad_value(0.0f);
    st.set_tensor(n.out, pad(x, pad_width_pairs, pad_value, "edge", s));
  } else {
    throw std::runtime_error("Unsupported pad mode: " + n.mode);
  }
}

// ----- Where -----
inline void
exec_where(const WhereNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& condition = st.const_tensor_ref(n.condition);
  const auto& x = st.const_tensor_ref(n.x);
  const auto& y = st.const_tensor_ref(n.y);
  st.set_tensor(n.out, where(condition, x, y, s));
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

// ----- Concatenate -----
inline void exec_concatenate(
    const ConcatenateNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  auto tensors_fb = n.tensors;
  std::vector<array> tensors;
  for (auto tid : tensors_fb) {
    tensors.push_back(st.const_tensor_ref(tid));
  }
  st.set_tensor(n.out, concatenate(tensors, n.axis, s));
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
      case OpCode::MULTIPLY:
        ops::exec_multiply(std::get<MultiplyNode>(instr.node), st, s);
        break;
      case OpCode::DIVIDE:
        ops::exec_divide(std::get<DivideNode>(instr.node), st, s);
        break;
      case OpCode::SUBTRACT:
        ops::exec_subtract(std::get<SubtractNode>(instr.node), st, s);
        break;
      case OpCode::CONV1D:
        ops::exec_conv1d(std::get<Conv1DNode>(instr.node), st, s);
        break;
      case OpCode::CONV2D:
        ops::exec_conv2d(std::get<Conv2DNode>(instr.node), st, s);
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
      case OpCode::SIGMOID:
        ops::exec_sigmoid(std::get<SigmoidNode>(instr.node), st, s);
        break;
      case OpCode::TANH:
        ops::exec_tanh(std::get<TanhNode>(instr.node), st, s);
        break;
      case OpCode::SQUEEZE:
        ops::exec_squeeze(std::get<SqueezeNode>(instr.node), st, s);
        break;
      case OpCode::SPLIT:
        ops::exec_split(std::get<SplitNode>(instr.node), st, s);
        break;
      case OpCode::RSQRT:
        ops::exec_rsqrt(std::get<RsqrtNode>(instr.node), st, s);
        break;
      case OpCode::MAXIMUM:
        ops::exec_maximum(std::get<MaximumNode>(instr.node), st, s);
        break;
      case OpCode::LOG:
        ops::exec_log(std::get<LogNode>(instr.node), st, s);
        break;
      case OpCode::SOFTMAX:
        ops::exec_softmax(std::get<SoftmaxNode>(instr.node), st, s);
        break;
      case OpCode::BROADCAST_TO:
        ops::exec_broadcast_to(std::get<BroadcastToNode>(instr.node), st, s);
        break;
      case OpCode::PAD:
        ops::exec_pad(std::get<PadNode>(instr.node), st, s);
        break;
      case OpCode::WHERE:
        ops::exec_where(std::get<WhereNode>(instr.node), st, s);
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
      case OpCode::CONCATENATE:
        ops::exec_concatenate(std::get<ConcatenateNode>(instr.node), st, s);
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
