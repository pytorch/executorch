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
 * Normalize axis to be in range [0, rank) and validate.
 * @param axis The axis value (can be negative)
 * @param rank The tensor rank
 * @param op_name Name of the operation for error messages
 * @return Normalized axis in range [0, rank)
 * @throws std::out_of_range if axis is out of range
 */
inline int normalize_axis(int axis, int rank, const char* op_name) {
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank) {
    throw std::out_of_range(std::string(op_name) + ": axis out of range");
  }
  return axis;
}

/**
 * Infers dimensions with -1 in a reshape-like operation.
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
// Formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
inline array gelu_tanh_impl(const array& x, StreamOrDevice s = {}) {
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

// ----- GELU implementation (exact, using erf) -----
// Formula: 0.5 * x * (1 + erf(x / sqrt(2)))
inline array gelu_none_impl(const array& x, StreamOrDevice s = {}) {
  constexpr float inv_sqrt_2 = 0.7071067812f;
  auto dtype = x.dtype();

  auto scaled = multiply(array(inv_sqrt_2, dtype), x, s);
  auto erf_val = erf(scaled, s);
  auto one_plus_erf = add(array(1.0f, dtype), erf_val, s);
  auto out = multiply(x, one_plus_erf, s);
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

  std::optional<array> freqs_arr = std::nullopt;
  if (n.freqs) {
    freqs_arr = st.const_tensor_ref(*n.freqs);
  }

  // MLX has two overloads: rope(..., int offset, ...) and rope(..., const
  // array& offset, ...) Call the appropriate one based on is_vid
  if (n.offset.is_vid) {
    // Scalar offset from Vid
    int offset = st.const_value_ref<int32_t>(n.offset.vid);
    st.set_tensor(
        n.out,
        fast::rope(
            x,
            n.head_dim,
            n.traditional,
            n.base,
            n.scale,
            offset,
            freqs_arr,
            s));
  } else {
    // Tensor offset from Tid
    const array& offset = st.const_tensor_ref(n.offset.tid);
    st.set_tensor(
        n.out,
        fast::rope(
            x,
            n.head_dim,
            n.traditional,
            n.base,
            n.scale,
            offset,
            freqs_arr,
            s));
  }
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
    // MLX's SDPA handles bool masks natively (True=attend, False=masked)
    // For non-bool masks, ensure dtype matches Q
    if (M.dtype() != bool_ && M.dtype() != Q.dtype()) {
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
exec_add_int(const AddIntNode& n, ExecutionState& st, StreamOrDevice) {
  int32_t a = resolve_int(n.a, st);
  int32_t b = resolve_int(n.b, st);
  st.set_value(n.out, a + b);
}

// ----- Sub Scalar -----
inline void exec_subtract_int(
    const SubtractIntNode& n,
    ExecutionState& st,
    StreamOrDevice) {
  int32_t a = resolve_int(n.a, st);
  int32_t b = resolve_int(n.b, st);
  st.set_value(n.out, a - b);
}

// ----- Mul Scalar -----
inline void exec_multiply_int(
    const MultiplyIntNode& n,
    ExecutionState& st,
    StreamOrDevice) {
  int32_t a = resolve_int(n.a, st);
  int32_t b = resolve_int(n.b, st);
  st.set_value(n.out, a * b);
}

// ----- Floor Div Scalar -----
inline void exec_floor_divide_int(
    const FloorDivideIntNode& n,
    ExecutionState& st,
    StreamOrDevice) {
  int32_t a = resolve_int(n.a, st);
  int32_t b = resolve_int(n.b, st);
  // Floor division for integers (Python semantics: rounds towards negative
  // infinity)
  int32_t result = a / b;
  // Adjust for floor division when signs differ and there's a remainder
  if ((a % b != 0) && ((a < 0) != (b < 0))) {
    result -= 1;
  }
  st.set_value(n.out, result);
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
  if (n.approximate == "tanh") {
    st.set_tensor(n.out, gelu_tanh_impl(x, s));
  } else {
    // "none" or any other value uses exact GELU
    st.set_tensor(n.out, gelu_none_impl(x, s));
  }
}

// ----- ARange -----
inline void
exec_arange(const ARangeNode& n, ExecutionState& st, StreamOrDevice s) {
  // Get start, stop, step - may be literal int64 or dynamic Vid
  int start_val = resolve_int(n.start, st);
  int stop_val = resolve_int(n.stop, st);
  int step_val = resolve_int(n.step, st);

  if (n.dtype.has_value()) {
    st.set_tensor(
        n.out,
        arange(
            start_val, stop_val, step_val, resolve_dtype(n.dtype.value()), s));
  } else {
    // No dtype specified - use MLX's default (infers from inputs)
    st.set_tensor(n.out, arange(start_val, stop_val, step_val, s));
  }
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

// ----- Minimum -----

inline void
exec_minimum(const MinimumNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, minimum(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
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

  // Replace -1 with actual input dimensions (PyTorch expand semantics:
  // -1 means "keep this dimension unchanged from input").
  // Dimensions are aligned from the RIGHT (broadcast semantics).
  const auto& x_shape = x.shape();
  int offset =
      static_cast<int>(shape_vec.size()) - static_cast<int>(x_shape.size());
  for (size_t i = 0; i < shape_vec.size(); i++) {
    if (shape_vec[i] == -1) {
      int input_dim = static_cast<int>(i) - offset;
      if (input_dim >= 0 && input_dim < static_cast<int>(x_shape.size())) {
        shape_vec[i] = static_cast<int>(x_shape[input_dim]);
      }
    }
  }

  st.set_tensor(
      n.out,
      broadcast_to(
          x, ::mlx::core::Shape(shape_vec.begin(), shape_vec.end()), s));
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

  int axis = normalize_axis(resolve_int(n.axis, st), rank, "Slice");
  int start = resolve_int(n.start, st);
  int stop = resolve_int(n.stop, st);

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

// ----- AsType -----
inline void
exec_astype(const AsTypeNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, astype(st.const_tensor_ref(n.x), resolve_dtype(n.dtype), s));
}

// ----- Quantized Linear -----
inline void exec_quantized_linear(
    const QuantizedLinearNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  // scale_only means biases should be computed, not provided
  assert(
      !(n.scale_only && n.biases) &&
      "scale_only=true but biases tensor also provided");

  array X = st.const_tensor_ref(n.x);
  array Wq = st.const_tensor_ref(n.w);
  array Sc = st.const_tensor_ref(n.scales);

  std::optional<array> Qb = std::nullopt;
  if (n.biases) {
    Qb = st.const_tensor_ref(*n.biases);
  } else if (n.scale_only) {
    // Compute biases from scales: B = -scales * 2^(bits-1)
    float offset = static_cast<float>(1 << (n.bits - 1));
    Qb = multiply(Sc, array(-offset, Sc.dtype()), s);
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

  Dtype out_dtype = resolve_dtype(n.out_dtype);
  if (out_dtype != Y.dtype()) {
    Y = astype(Y, out_dtype, s);
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
  st.set_tensor(
      n.out, full(to_shape(n.shape, st), n.v, resolve_dtype(n.dtype), s));
}

// ----- FullLike -----
inline void
exec_full_like(const FullLikeNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  // Use input dtype if not specified
  auto dtype = n.dtype.has_value() ? resolve_dtype(n.dtype.value()) : x.dtype();
  st.set_tensor(n.out, full_like(x, n.v, dtype, s));
}

// ----- Slice Update -----
inline void exec_slice_update(
    const SliceUpdateNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  array& dst = st.tensor_ref(n.dst);
  const array& upd = st.const_tensor_ref(n.update);

  const int rank = static_cast<int>(dst.ndim());

  int axis = normalize_axis(resolve_int(n.axis, st), rank, "SliceUpdate");
  int start = resolve_int(n.start, st);
  int stop = resolve_int(n.stop, st);

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

// ----- Index Update -----
// Helper: finds next contiguous run in indices starting at offset
// Returns (dst_start, dst_stop, upd_start, upd_stop) for the run
// Returns (0, 0, 0, 0) when no more runs
inline std::tuple<int, int, int, int> next_contiguous_run(
    const std::vector<int32_t>& indices,
    size_t offset) {
  if (offset >= indices.size())
    return {0, 0, 0, 0};

  int dst_start = indices[offset];
  int upd_start = static_cast<int>(offset);
  size_t len = 1;
  while (offset + len < indices.size() &&
         indices[offset + len] == dst_start + static_cast<int>(len)) {
    ++len;
  }
  int dst_stop = dst_start + static_cast<int>(len);
  int upd_stop = upd_start + static_cast<int>(len);
  return {dst_start, dst_stop, upd_start, upd_stop};
}

// Copies update tensor into dst at positions specified by 1D indices along axis
// Optimizes into slice_update calls for contiguous runs
inline void exec_index_update(
    const IndexUpdateNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  array& dst = st.tensor_ref(n.dst);
  const array& upd = st.const_tensor_ref(n.update);
  const array& indices = st.const_tensor_ref(n.indices);
  if (indices.ndim() != 1) {
    throw std::invalid_argument("IndexUpdate: indices must be 1D");
  }
  const int rank = static_cast<int>(dst.ndim());
  int axis = normalize_axis(n.axis, rank, "IndexUpdate");
  const int dst_dim = static_cast<int>(dst.shape()[axis]);

  // Get indices as a vector of ints, handling negative indices
  // Note: PyTorch uses int64 for indices, so we read as int64_t
  eval(indices); // Ensure indices are materialized before accessing data
  if (indices.dtype() != ::mlx::core::int64) {
    throw std::invalid_argument(
        std::string("IndexUpdate: expected int64 indices, got ") +
        ExecutionState::dtype_str(indices.dtype()));
  }
  std::vector<int32_t> idx_vec(indices.size());
  auto idx_data = indices.data<int64_t>();
  for (size_t i = 0; i < indices.size(); ++i) {
    int64_t idx = idx_data[i];
    if (idx < 0) {
      idx += dst_dim;
    }
    if (idx < 0 || idx >= dst_dim) {
      throw std::out_of_range(
          "IndexUpdate: index " + std::to_string(idx_data[i]) +
          " out of range for axis " + std::to_string(axis) + " with size " +
          std::to_string(dst_dim));
    }
    idx_vec[i] = idx;
  }

  if (idx_vec.empty()) {
    return;
  }

  // Build base start/stop vectors for slice_update
  std::vector<int> dst_vstart(rank, 0);
  std::vector<int> dst_vstop;
  dst_vstop.reserve(rank);
  auto sh = dst.shape();
  for (int i = 0; i < rank; ++i) {
    dst_vstop.push_back(static_cast<int>(sh[i]));
  }

  std::vector<int> upd_vstart(rank, 0);
  std::vector<int> upd_vstop;
  upd_vstop.reserve(rank);
  auto upd_sh = upd.shape();
  for (int i = 0; i < rank; ++i) {
    upd_vstop.push_back(static_cast<int>(upd_sh[i]));
  }

  // Process contiguous runs
  size_t offset = 0;
  while (offset < idx_vec.size()) {
    auto [dst_start, dst_stop, upd_start, upd_stop] =
        next_contiguous_run(idx_vec, offset);

    // Set axis range for dst
    dst_vstart[axis] = dst_start;
    dst_vstop[axis] = dst_stop;

    // Set axis range for upd slice
    upd_vstart[axis] = upd_start;
    upd_vstop[axis] = upd_stop;

    // Slice update - skip slicing if using entire update tensor
    array upd_slice =
        (upd_start == 0 && upd_stop == static_cast<int>(upd_sh[axis]))
        ? upd
        : slice(upd, to_shape(upd_vstart), to_shape(upd_vstop), s);
    dst = slice_update(
        dst, upd_slice, to_shape(dst_vstart), to_shape(dst_vstop), s);

    offset = static_cast<size_t>(upd_stop);
  }
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

  Dtype out_dtype = resolve_dtype(n.out_dtype);
  if (out_dtype != Y.dtype()) {
    Y = astype(Y, out_dtype, s);
  }

  st.set_tensor(n.out, std::move(Y));
}

// ----- Comparison Ops -----
inline void exec_less(const LessNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, less(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

inline void
exec_less_equal(const LessEqualNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, less_equal(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

inline void
exec_greater(const GreaterNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, greater(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

inline void exec_greater_equal(
    const GreaterEqualNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  st.set_tensor(
      n.out,
      greater_equal(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

inline void
exec_equal(const EqualNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, equal(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

inline void
exec_not_equal(const NotEqualNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, not_equal(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

// ----- Logical Ops -----
inline void exec_logical_not(
    const LogicalNotNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  st.set_tensor(n.out, logical_not(st.const_tensor_ref(n.a), s));
}

inline void exec_logical_and(
    const LogicalAndNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  st.set_tensor(
      n.out,
      logical_and(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

inline void
exec_logical_or(const LogicalOrNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, logical_or(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

// ----- Tri -----
inline void exec_tri(const TriNode& n, ExecutionState& st, StreamOrDevice s) {
  int rows = resolve_int(n.n, st);
  int cols = resolve_int(n.m, st);
  auto dtype = resolve_dtype(n.dtype);
  st.set_tensor(n.out, tri(rows, cols, n.k, dtype, s));
}

// ----- Tril -----
inline void exec_tril(const TrilNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  st.set_tensor(n.out, tril(x, n.k, s));
}

// ----- Triu -----
inline void exec_triu(const TriuNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  st.set_tensor(n.out, triu(x, n.k, s));
}

// =============================================================================
// Math ops - Unary element-wise
// =============================================================================

// ----- Floor -----
inline void
exec_floor(const FloorNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, floor(st.const_tensor_ref(n.x), s));
}

// ----- Ceil -----
inline void exec_ceil(const CeilNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, ceil(st.const_tensor_ref(n.x), s));
}

// ----- Square -----
inline void
exec_square(const SquareNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, square(st.const_tensor_ref(n.x), s));
}

// ----- Exp -----
inline void exec_exp(const ExpNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, exp(st.const_tensor_ref(n.x), s));
}

// ----- Sin -----
inline void exec_sin(const SinNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, sin(st.const_tensor_ref(n.x), s));
}

// ----- Cos -----
inline void exec_cos(const CosNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, cos(st.const_tensor_ref(n.x), s));
}

// ----- Tan -----
inline void exec_tan(const TanNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, tan(st.const_tensor_ref(n.x), s));
}

// ----- Arcsin -----
inline void
exec_arcsin(const ArcsinNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, arcsin(st.const_tensor_ref(n.x), s));
}

// ----- Arccos -----
inline void
exec_arccos(const ArccosNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, arccos(st.const_tensor_ref(n.x), s));
}

// ----- Arctan -----
inline void
exec_arctan(const ArctanNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, arctan(st.const_tensor_ref(n.x), s));
}

// ----- Sinh -----
inline void exec_sinh(const SinhNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, sinh(st.const_tensor_ref(n.x), s));
}

// ----- Cosh -----
inline void exec_cosh(const CoshNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, cosh(st.const_tensor_ref(n.x), s));
}

// ----- Arcsinh -----
inline void
exec_arcsinh(const ArcsinhNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, arcsinh(st.const_tensor_ref(n.x), s));
}

// ----- Arccosh -----
inline void
exec_arccosh(const ArccoshNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, arccosh(st.const_tensor_ref(n.x), s));
}

// ----- Arctanh -----
inline void
exec_arctanh(const ArctanhNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, arctanh(st.const_tensor_ref(n.x), s));
}

// ----- Log2 -----
inline void exec_log2(const Log2Node& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, log2(st.const_tensor_ref(n.x), s));
}

// ----- Log10 -----
inline void
exec_log10(const Log10Node& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, log10(st.const_tensor_ref(n.x), s));
}

// ----- Log1p -----
inline void
exec_log1p(const Log1pNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, log1p(st.const_tensor_ref(n.x), s));
}

// ----- Erf -----
inline void exec_erf(const ErfNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, erf(st.const_tensor_ref(n.x), s));
}

// ----- Expm1 -----
inline void
exec_expm1(const Expm1Node& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, expm1(st.const_tensor_ref(n.x), s));
}

// ----- Round -----
inline void
exec_round(const RoundNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, round(st.const_tensor_ref(n.x), n.decimals, s));
}

// ----- Reciprocal -----
inline void
exec_reciprocal(const ReciprocalNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, reciprocal(st.const_tensor_ref(n.x), s));
}

// ----- Sqrt -----
inline void exec_sqrt(const SqrtNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, sqrt(st.const_tensor_ref(n.x), s));
}

// ----- Abs -----
inline void exec_abs(const AbsNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, abs(st.const_tensor_ref(n.x), s));
}

// ----- Neg -----
inline void exec_neg(const NegNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(n.out, negative(st.const_tensor_ref(n.x), s));
}

// =============================================================================
// Math ops - Binary element-wise
// =============================================================================

// ----- Atan2 -----
inline void
exec_atan2(const Atan2Node& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, arctan2(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

// ----- LogAddExp -----
inline void
exec_logaddexp(const LogAddExpNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, logaddexp(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

// ----- FloorDivide -----
inline void exec_floor_divide(
    const FloorDivideNode& n,
    ExecutionState& st,
    StreamOrDevice s) {
  st.set_tensor(
      n.out,
      floor_divide(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

// ----- Power -----
inline void
exec_power(const PowerNode& n, ExecutionState& st, StreamOrDevice s) {
  st.set_tensor(
      n.out, power(st.const_tensor_ref(n.a), st.const_tensor_ref(n.b), s));
}

// =============================================================================
// Math ops - Reduction
// =============================================================================

// ----- LogSumExp -----
inline void
exec_logsumexp(const LogSumExpNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  std::vector<int> axes(n.axes.begin(), n.axes.end());
  st.set_tensor(n.out, logsumexp(x, axes, n.keepdims, s));
}

// ----- Sum -----
inline void exec_sum(const SumNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  std::vector<int> axes(n.axes.begin(), n.axes.end());
  if (axes.empty()) {
    st.set_tensor(n.out, sum(x, n.keepdims, s));
  } else {
    st.set_tensor(n.out, sum(x, axes, n.keepdims, s));
  }
}

// ----- Mean -----
inline void exec_mean(const MeanNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  std::vector<int> axes(n.axes.begin(), n.axes.end());
  if (axes.empty()) {
    st.set_tensor(n.out, mean(x, n.keepdims, s));
  } else {
    st.set_tensor(n.out, mean(x, axes, n.keepdims, s));
  }
}

// ----- Var -----
inline void exec_var(const VarNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  std::vector<int> axes(n.axes.begin(), n.axes.end());
  if (axes.empty()) {
    st.set_tensor(n.out, var(x, n.keepdims, n.ddof, s));
  } else {
    st.set_tensor(n.out, var(x, axes, n.keepdims, n.ddof, s));
  }
}

// ----- Std -----
inline void exec_std(const StdNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  std::vector<int> axes(n.axes.begin(), n.axes.end());
  if (axes.empty()) {
    st.set_tensor(n.out, ::mlx::core::std(x, n.keepdims, n.ddof, s));
  } else {
    st.set_tensor(n.out, ::mlx::core::std(x, axes, n.keepdims, n.ddof, s));
  }
}

// ----- Prod -----
inline void exec_prod(const ProdNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  std::vector<int> axes(n.axes.begin(), n.axes.end());
  if (axes.empty()) {
    st.set_tensor(n.out, prod(x, n.keepdims, s));
  } else {
    st.set_tensor(n.out, prod(x, axes, n.keepdims, s));
  }
}

// ----- Max (amax) -----
inline void exec_max(const MaxNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  std::vector<int> axes(n.axes.begin(), n.axes.end());
  if (axes.empty()) {
    st.set_tensor(n.out, max(x, n.keepdims, s));
  } else {
    st.set_tensor(n.out, max(x, axes, n.keepdims, s));
  }
}

// ----- Min (amin) -----
inline void exec_min(const MinNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  std::vector<int> axes(n.axes.begin(), n.axes.end());
  if (axes.empty()) {
    st.set_tensor(n.out, min(x, n.keepdims, s));
  } else {
    st.set_tensor(n.out, min(x, axes, n.keepdims, s));
  }
}

// ----- Argmax -----
inline void
exec_argmax(const ArgmaxNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  st.set_tensor(n.out, argmax(x, n.axis, n.keepdims, s));
}

// ----- Argmin -----
inline void
exec_argmin(const ArgminNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  st.set_tensor(n.out, argmin(x, n.axis, n.keepdims, s));
}

// ----- Median -----
inline void
exec_median(const MedianNode& n, ExecutionState& st, StreamOrDevice s) {
  const auto& x = st.const_tensor_ref(n.x);
  std::vector<int> axes(n.axes.begin(), n.axes.end());
  if (axes.empty()) {
    st.set_tensor(n.out, median(x, n.keepdims, s));
  } else {
    st.set_tensor(n.out, median(x, axes, n.keepdims, s));
  }
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
      case OpCode::ADD_INT:
        ops::exec_add_int(std::get<AddIntNode>(instr.node), st, s);
        break;
      case OpCode::SUBTRACT_INT:
        ops::exec_subtract_int(std::get<SubtractIntNode>(instr.node), st, s);
        break;
      case OpCode::MULTIPLY_INT:
        ops::exec_multiply_int(std::get<MultiplyIntNode>(instr.node), st, s);
        break;
      case OpCode::FLOOR_DIVIDE_INT:
        ops::exec_floor_divide_int(
            std::get<FloorDivideIntNode>(instr.node), st, s);
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
      case OpCode::MINIMUM:
        ops::exec_minimum(std::get<MinimumNode>(instr.node), st, s);
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
      case OpCode::ASTYPE:
        ops::exec_astype(std::get<AsTypeNode>(instr.node), st, s);
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
      case OpCode::FULL_LIKE:
        ops::exec_full_like(std::get<FullLikeNode>(instr.node), st, s);
        break;
      case OpCode::ARGMAX:
        ops::exec_argmax(std::get<ArgmaxNode>(instr.node), st, s);
        break;
      case OpCode::SLICE_UPDATE:
        ops::exec_slice_update(std::get<SliceUpdateNode>(instr.node), st, s);
        break;
      case OpCode::INDEX_UPDATE:
        ops::exec_index_update(std::get<IndexUpdateNode>(instr.node), st, s);
        break;
      case OpCode::QUANTIZED_GATHER:
        ops::exec_quantized_gather(
            std::get<QuantizedGatherNode>(instr.node), st, s);
        break;
      case OpCode::LESS:
        ops::exec_less(std::get<LessNode>(instr.node), st, s);
        break;
      case OpCode::LESS_EQUAL:
        ops::exec_less_equal(std::get<LessEqualNode>(instr.node), st, s);
        break;
      case OpCode::GREATER:
        ops::exec_greater(std::get<GreaterNode>(instr.node), st, s);
        break;
      case OpCode::GREATER_EQUAL:
        ops::exec_greater_equal(std::get<GreaterEqualNode>(instr.node), st, s);
        break;
      case OpCode::EQUAL:
        ops::exec_equal(std::get<EqualNode>(instr.node), st, s);
        break;
      case OpCode::NOT_EQUAL:
        ops::exec_not_equal(std::get<NotEqualNode>(instr.node), st, s);
        break;
      case OpCode::LOGICAL_NOT:
        ops::exec_logical_not(std::get<LogicalNotNode>(instr.node), st, s);
        break;
      case OpCode::LOGICAL_AND:
        ops::exec_logical_and(std::get<LogicalAndNode>(instr.node), st, s);
        break;
      case OpCode::LOGICAL_OR:
        ops::exec_logical_or(std::get<LogicalOrNode>(instr.node), st, s);
        break;
      case OpCode::TRI:
        ops::exec_tri(std::get<TriNode>(instr.node), st, s);
        break;
      case OpCode::TRIL:
        ops::exec_tril(std::get<TrilNode>(instr.node), st, s);
        break;
      case OpCode::TRIU:
        ops::exec_triu(std::get<TriuNode>(instr.node), st, s);
        break;
      // Math ops - Unary
      case OpCode::FLOOR:
        ops::exec_floor(std::get<FloorNode>(instr.node), st, s);
        break;
      case OpCode::CEIL:
        ops::exec_ceil(std::get<CeilNode>(instr.node), st, s);
        break;
      case OpCode::SQUARE:
        ops::exec_square(std::get<SquareNode>(instr.node), st, s);
        break;
      case OpCode::EXP:
        ops::exec_exp(std::get<ExpNode>(instr.node), st, s);
        break;
      case OpCode::SIN:
        ops::exec_sin(std::get<SinNode>(instr.node), st, s);
        break;
      case OpCode::COS:
        ops::exec_cos(std::get<CosNode>(instr.node), st, s);
        break;
      case OpCode::TAN:
        ops::exec_tan(std::get<TanNode>(instr.node), st, s);
        break;
      case OpCode::ARCSIN:
        ops::exec_arcsin(std::get<ArcsinNode>(instr.node), st, s);
        break;
      case OpCode::ARCCOS:
        ops::exec_arccos(std::get<ArccosNode>(instr.node), st, s);
        break;
      case OpCode::ARCTAN:
        ops::exec_arctan(std::get<ArctanNode>(instr.node), st, s);
        break;
      case OpCode::SINH:
        ops::exec_sinh(std::get<SinhNode>(instr.node), st, s);
        break;
      case OpCode::COSH:
        ops::exec_cosh(std::get<CoshNode>(instr.node), st, s);
        break;
      case OpCode::ARCSINH:
        ops::exec_arcsinh(std::get<ArcsinhNode>(instr.node), st, s);
        break;
      case OpCode::ARCCOSH:
        ops::exec_arccosh(std::get<ArccoshNode>(instr.node), st, s);
        break;
      case OpCode::ARCTANH:
        ops::exec_arctanh(std::get<ArctanhNode>(instr.node), st, s);
        break;
      case OpCode::LOG2:
        ops::exec_log2(std::get<Log2Node>(instr.node), st, s);
        break;
      case OpCode::LOG10:
        ops::exec_log10(std::get<Log10Node>(instr.node), st, s);
        break;
      case OpCode::LOG1P:
        ops::exec_log1p(std::get<Log1pNode>(instr.node), st, s);
        break;
      case OpCode::ERF:
        ops::exec_erf(std::get<ErfNode>(instr.node), st, s);
        break;
      case OpCode::EXPM1:
        ops::exec_expm1(std::get<Expm1Node>(instr.node), st, s);
        break;
      case OpCode::ROUND:
        ops::exec_round(std::get<RoundNode>(instr.node), st, s);
        break;
      case OpCode::RECIPROCAL:
        ops::exec_reciprocal(std::get<ReciprocalNode>(instr.node), st, s);
        break;
      case OpCode::SQRT:
        ops::exec_sqrt(std::get<SqrtNode>(instr.node), st, s);
        break;
      case OpCode::ABS:
        ops::exec_abs(std::get<AbsNode>(instr.node), st, s);
        break;
      case OpCode::NEG:
        ops::exec_neg(std::get<NegNode>(instr.node), st, s);
        break;
      // Math ops - Binary
      case OpCode::ATAN2:
        ops::exec_atan2(std::get<Atan2Node>(instr.node), st, s);
        break;
      case OpCode::LOG_ADD_EXP:
        ops::exec_logaddexp(std::get<LogAddExpNode>(instr.node), st, s);
        break;
      case OpCode::FLOOR_DIVIDE:
        ops::exec_floor_divide(std::get<FloorDivideNode>(instr.node), st, s);
        break;
      case OpCode::POWER:
        ops::exec_power(std::get<PowerNode>(instr.node), st, s);
        break;
      // Math ops - Reduction
      case OpCode::LOG_SUM_EXP:
        ops::exec_logsumexp(std::get<LogSumExpNode>(instr.node), st, s);
        break;
      case OpCode::SUM:
        ops::exec_sum(std::get<SumNode>(instr.node), st, s);
        break;
      case OpCode::MEAN:
        ops::exec_mean(std::get<MeanNode>(instr.node), st, s);
        break;
      case OpCode::VAR:
        ops::exec_var(std::get<VarNode>(instr.node), st, s);
        break;
      case OpCode::STD:
        ops::exec_std(std::get<StdNode>(instr.node), st, s);
        break;
      case OpCode::PROD:
        ops::exec_prod(std::get<ProdNode>(instr.node), st, s);
        break;
      case OpCode::MAX:
        ops::exec_max(std::get<MaxNode>(instr.node), st, s);
        break;
      case OpCode::MIN:
        ops::exec_min(std::get<MinNode>(instr.node), st, s);
        break;
      case OpCode::ARGMIN:
        ops::exec_argmin(std::get<ArgminNode>(instr.node), st, s);
        break;
      case OpCode::MEDIAN:
        ops::exec_median(std::get<MedianNode>(instr.node), st, s);
        break;
      case OpCode::SENTINEL:
        break;
    }
  }
};

} // namespace mlx
} // namespace backends
} // namespace executorch
