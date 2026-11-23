// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlGraphBuilder.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <ggml.h>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlTypes.h>

namespace ptn::ggml_detail {

void GraphBuilder::lower_rms_norm(const Node& node) {
  const TensorRef input = tensor_arg(node, "input", 0);
  const std::vector<int64_t> normalized =
      int_list(require_arg(node, "normalized_shape", 1).arg);
  if (input.sizes.empty() || normalized.size() != 1 ||
      normalized[0] != input.sizes.back()) {
    throw std::runtime_error("ggml: rms_norm only normalizes the last dim");
  }
  const double eps =
      float_arg(node, "eps", 3, std::numeric_limits<float>::epsilon(), true);
  // ggml's norm kernels only read F32.
  ggml_tensor* result = ggml_rms_norm(
      instance_.context.get(), as_f32(input.tensor), static_cast<float>(eps));
  if (const std::optional<TensorRef> weight =
          optional_tensor(node, "weight", 2)) {
    result = ggml_mul(instance_.context.get(), result, as_f32(weight->tensor));
  }
  set_single_output(node, TensorRef{result, input.sizes});
}

void GraphBuilder::lower_layer_norm(const Node& node) {
  TensorRef input = tensor_arg(node, "input", 0);
  const std::vector<int64_t> normalized =
      int_list(require_arg(node, "normalized_shape", 1).arg);
  if (input.sizes.empty() || normalized.size() != 1 ||
      normalized[0] != input.sizes.back()) {
    throw std::runtime_error("ggml: layer_norm only normalizes the last dim");
  }
  require_only_first_output_used(node);
  // ggml's norm kernels only read F32. Computed in F32, as eager does for F16
  // and BF16; set() rounds back.
  input.tensor = half_as_f32(input.tensor);
  require_f32(node, input.tensor);
  const double eps = float_arg(node, "eps", 4, 1e-5, true);
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* result =
      ggml_norm(ctx, contiguous(input.tensor), static_cast<float>(eps));
  if (const std::optional<TensorRef> weight =
          optional_tensor(node, "weight", 2)) {
    result = ggml_mul(ctx, result, half_as_f32(weight->tensor));
  }
  if (const std::optional<TensorRef> bias = optional_tensor(node, "bias", 3)) {
    result = ggml_add(ctx, result, half_as_f32(bias->tensor));
  }
  set(output_ids(node).at(0), TensorRef{result, input.sizes});
}

void GraphBuilder::lower_gelu(const Node& node) {
  const TensorRef input = tensor_arg(node, "self", 0);
  require_float(node, input.tensor);
  const NamedArgument* approximate = find_arg(node, "approximate", 1);
  const bool tanh = approximate != nullptr &&
      approximate->arg.kind() == ArgKind::String &&
      approximate->arg.as_string().value == "tanh";
  ggml_context* ctx = instance_.context.get();
  if (!tanh) {
    set_single_output(
        node, TensorRef{ggml_gelu_erf(ctx, input.tensor), input.sizes});
    return;
  }
  // ggml_gelu, as llama.cpp: ggml-cpu reads it from an F16 table, a few 1e-4
  // off PyTorch's in F32, and GPU backends evaluate the formula in F32 in one
  // kernel. With exact_gelu_for_testing the CPU builds it from exact ops:
  // x * (1 + tanh(c * (x + 0.044715 x^3))) / 2 with c = sqrt(2 / pi).
  if (gpu_ || !exact_gelu_) {
    set_single_output(
        node, TensorRef{ggml_gelu(ctx, input.tensor), input.sizes});
    return;
  }
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  ggml_tensor* x = contiguous(as_f32(input.tensor));
  ggml_tensor* cube = ggml_mul(ctx, x, ggml_mul(ctx, x, x));
  ggml_tensor* inner = ggml_scale(
      ctx, ggml_add(ctx, x, ggml_scale(ctx, cube, 0.044715f)), kSqrt2OverPi);
  ggml_tensor* half_one_plus_tanh =
      ggml_scale_bias(ctx, ggml_tanh(ctx, inner), 0.5f, 0.5f);
  set_as_output_type(
      output_ids(node).at(0),
      TensorRef{ggml_mul(ctx, x, half_one_plus_tanh), input.sizes});
}

void GraphBuilder::lower_cast(const Node& node) {
  // ggml_cpy reads any strides, so a deferred permutation is consumed here.
  const ValueId input_id = tensor_arg_id(node, "self", 0);
  const TensorRef input = view(input_id);
  const std::vector<ValueId> ids = output_ids(node);
  if (ids.size() != 1) {
    throw std::runtime_error("ggml: cast must produce one output");
  }
  const TensorMeta& meta = method_.graph.value(ids.front()).tensor_meta();
  const ggml_type type = compute_type(meta);
  if (input.tensor->type == type) {
    // Already the target type: share it as materialize() would, keeping a
    // deferred view deferred.
    set_view(ids.front(), input);
    return;
  }
  // ggml-cpu converts I32 only to and from F32.
  ggml_tensor* source = input.tensor;
  if ((source->type == GGML_TYPE_I32) != (type == GGML_TYPE_I32) &&
      source->type != GGML_TYPE_F32 && type != GGML_TYPE_F32) {
    source = as_f32(source);
  }
  set(ids.front(),
      TensorRef{ggml_cast(instance_.context.get(), source, type), input.sizes});
}

void GraphBuilder::lower_binary(const Node& node, const std::string& key) {
  const TensorRef left = tensor_arg(node, "self", 0);
  const NamedArgument& rhs_arg = require_arg(node, "other", 1);
  if (float_arg(node, "alpha", 2, 1.0, true) != 1.0) {
    throw std::runtime_error("ggml: add/sub alpha must be one");
  }
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* result = nullptr;
  std::vector<int64_t> sizes = left.sizes;
  if (rhs_arg.arg.kind() == ArgKind::Tensor) {
    BinaryOperands operands = tensor_operands(
        left, tensor(rhs_arg.arg.as_tensor().id), /*force_f32=*/false);
    ggml_tensor* lhs = operands.lhs;
    ggml_tensor* right = operands.rhs;
    sizes = std::move(operands.sizes);
    if (key == "aten.add.Tensor") {
      result = ggml_add(ctx, lhs, right);
    } else if (key == "aten.mul.Tensor") {
      // Recognize the backend-independent arithmetic spellings of SiLU
      // and SwiGLU. A single ggml GLU node also lets CUDA use its generic
      // fused quantized matvec path when the operands qualify.
      const auto is_unary = [](const ggml_tensor* t, ggml_unary_op op) {
        return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
      };
      const auto matches_swiglu = [&](ggml_tensor* gate, ggml_tensor* up) {
        ggml_tensor* x = gate->src[0];
        return is_unary(gate, GGML_UNARY_OP_SILU) && x != nullptr &&
            gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(gate, up) && ggml_is_contiguous_1(x) &&
            ggml_is_contiguous_1(up);
      };
      const auto matches_silu = [&](ggml_tensor* sigmoid, ggml_tensor* x) {
        return is_unary(sigmoid, GGML_UNARY_OP_SIGMOID) &&
            sigmoid->src[0] == x && x->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(sigmoid, x);
      };
      if (matches_swiglu(lhs, right)) {
        result = ggml_swiglu_split(ctx, lhs->src[0], right);
      } else if (matches_swiglu(right, lhs)) {
        result = ggml_swiglu_split(ctx, right->src[0], lhs);
      } else if (matches_silu(lhs, right)) {
        result = ggml_silu(ctx, right);
      } else if (matches_silu(right, lhs)) {
        result = ggml_silu(ctx, lhs);
      } else {
        result = ggml_mul(ctx, lhs, right);
      }
    } else if (key == "aten.sub.Tensor") {
      result = ggml_sub(ctx, lhs, right);
    } else {
      result = ggml_div(ctx, lhs, right);
    }
  } else {
    const float scalar = static_cast<float>(float_arg(node, "other", 1, 0));
    // ggml_scale only takes contiguous F32.
    ggml_tensor* lhs = contiguous(as_f32(left.tensor));
    if (key == "aten.mul.Tensor") {
      result = ggml_scale(ctx, lhs, scalar);
    } else if (key == "aten.div.Tensor") {
      result = ggml_scale(ctx, lhs, 1.0f / scalar);
    } else if (key == "aten.add.Tensor") {
      result = ggml_scale_bias(ctx, lhs, 1.0f, scalar);
    } else {
      result = ggml_scale_bias(ctx, lhs, 1.0f, -scalar);
    }
  }
  // Bools are carried as 0/1 floats, and Bool add is a logical or.
  const std::vector<ValueId> ids = output_ids(node);
  if (key == "aten.add.Tensor" && ids.size() == 1 &&
      method_.graph.value(ids.front()).tensor_meta().dtype ==
          ScalarType::Bool) {
    result = ggml_step(ctx, result);
  }
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

// ggml-cpu's unary kernels abort on integer tensors.
void GraphBuilder::require_float(const Node& node, const ggml_tensor* tensor) {
  if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 ||
      tensor->type == GGML_TYPE_I32 || tensor->type == GGML_TYPE_I64) {
    throw std::runtime_error(
        "ggml: op '" + node.target + "' supports floating-point tensors only");
  }
}

void GraphBuilder::lower_unary(const Node& node, Unary fn) {
  const TensorRef input = tensor_arg(node, "self", 0);
  require_float(node, input.tensor);
  set_single_output(
      node, TensorRef{fn(instance_.context.get(), input.tensor), input.sizes});
}

void GraphBuilder::lower_sqrt(const Node& node) {
  const TensorRef input = tensor_arg(node, "self", 0);
  // ggml's sqrt reads F16 and BF16 as they are and computes in F32, as eager
  // does.
  if (input.tensor->type != GGML_TYPE_F16 &&
      input.tensor->type != GGML_TYPE_BF16) {
    require_f32(node, input.tensor);
  }
  set_single_output(
      node,
      TensorRef{
          ggml_sqrt(instance_.context.get(), contiguous(input.tensor)),
          input.sizes});
}

void GraphBuilder::lower_hardtanh(const Node& node) {
  TensorRef input = tensor_arg(node, "self", 0);
  require_float(node, input.tensor);
  // ggml-cpu clamps F32 and F16 only, so BF16 clamps in F32, exactly; set()
  // rounds back.
  if (input.tensor->type == GGML_TYPE_BF16) {
    input.tensor = as_f32(input.tensor);
  }
  const float min =
      static_cast<float>(float_arg(node, "min_val", 1, -1.0, true));
  const float max =
      static_cast<float>(float_arg(node, "max_val", 2, 1.0, true));
  set_single_output(
      node,
      TensorRef{
          ggml_clamp(instance_.context.get(), input.tensor, min, max),
          input.sizes});
}

// ATen's elu is x > 0 ? scale * x : alpha * scale * expm1(input_scale * x), and
// ggml's x > 0 ? x : expm1(x).
void GraphBuilder::lower_elu(const Node& node) {
  for (const auto& [name, index] :
       {std::pair{"alpha", 1}, {"scale", 2}, {"input_scale", 3}}) {
    if (float_arg(node, name, index, 1.0, true) != 1.0) {
      throw std::runtime_error(
          "ggml: elu supports only alpha, scale and input_scale 1");
    }
  }
  lower_unary(node, ggml_elu);
}

// Reduces any dimensions: they are permuted to the end, where ggml sees them
// as one row.
std::pair<ggml_tensor*, std::vector<int64_t>> GraphBuilder::reduced_rows(
    const Node& node,
    const std::string& op) {
  TensorRef input = tensor_arg(node, "self", 0);
  // Computed in F32, as eager does for F16 and BF16; set() rounds back.
  input.tensor = half_as_f32(input.tensor);
  require_f32(node, input.tensor);
  const NamedArgument* dtype = find_arg(node, "dtype", 3);
  if (dtype != nullptr && dtype->arg.kind() != ArgKind::None) {
    throw std::runtime_error("ggml: " + op + " does not support dtype");
  }
  const size_t rank = input.sizes.size();
  const NamedArgument* dim_arg = find_arg(node, "dim", 1);
  std::vector<int64_t> dims;
  if (dim_arg != nullptr && dim_arg->arg.kind() != ArgKind::None) {
    dims = int_list(dim_arg->arg);
  }
  if (dims.empty()) {
    dims.resize(rank);
    std::iota(dims.begin(), dims.end(), 0);
  }
  std::ranges::transform(dims, dims.begin(), [rank](int64_t dim) {
    return normalize_dim(dim, rank);
  });
  std::ranges::sort(dims);
  if (std::ranges::adjacent_find(dims) != dims.end()) {
    throw std::runtime_error("ggml: " + op + " repeats a dimension");
  }
  const bool keepdim = bool_arg(node, "keepdim", 2, false);
  std::vector<int64_t> order;
  std::vector<int64_t> sizes;
  int64_t row = 1;
  for (size_t i = 0; i < rank; ++i) {
    const auto dim = static_cast<int64_t>(i);
    if (!std::ranges::binary_search(dims, dim)) {
      order.push_back(dim);
      sizes.push_back(input.sizes[i]);
    } else if (keepdim) {
      sizes.push_back(1);
    }
  }
  for (const int64_t dim : dims) {
    order.push_back(dim);
    row *= input.sizes[static_cast<size_t>(dim)];
  }
  if (row == 0) {
    throw std::runtime_error("ggml: " + op + " over an empty dimension");
  }
  if (!std::ranges::is_sorted(order)) {
    input = permuted(input, order);
  }
  ggml_tensor* rows = ggml_reshape_2d(
      instance_.context.get(),
      contiguous(input.tensor),
      row,
      static_cast<int64_t>(checked_numel(input.sizes)) / row);
  return {rows, std::move(sizes)};
}

void GraphBuilder::lower_mean(const Node& node) {
  auto [rows, sizes] = reduced_rows(node, "mean");
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* result = reshape(ctx, ggml_mean(ctx, rows), sizes);
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

void GraphBuilder::lower_sum(const Node& node) {
  auto [rows, sizes] = reduced_rows(node, "sum");
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* result = reshape(ctx, ggml_sum_rows(ctx, rows), sizes);
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

// ggml_argmax of each row as I32 indices for ggml_get_rows. GPU kernels give
// -1 for a row with no value above -FLT_MAX (all -inf, -FLT_MAX or NaN), so
// there the index is clamped to 0, an element of that row, in F32, which
// holds it exactly for rows of up to 2^24 values; row_max corrects the value.
ggml_tensor* GraphBuilder::row_argmax(ggml_tensor* rows) {
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* index = ggml_argmax(ctx, rows);
  if (!gpu_) {
    return index;
  }
  if (rows->ne[0] > kExactF32Integer) {
    throw std::runtime_error(
        "ggml: GPUs reduce rows of up to 2^24 values with an argmax");
  }
  return ggml_cast(
      ctx, ggml_relu(ctx, ggml_cast(ctx, index, GGML_TYPE_F32)), GGML_TYPE_I32);
}

// Each row's largest value, [1, 1, count], gathered at row_argmax. On GPUs a
// row with no value above -FLT_MAX gathers its first element instead, so a row
// select gives it -FLT_MAX if it holds one, else -inf; NaN is for the callers.
ggml_tensor* GraphBuilder::row_max(ggml_tensor* rows) {
  ggml_context* ctx = instance_.context.get();
  const int64_t width = rows->ne[0];
  const int64_t count = rows->ne[1];
  ggml_tensor* index = ggml_reshape_2d(ctx, row_argmax(rows), 1, count);
  ggml_tensor* values =
      ggml_get_rows(ctx, ggml_reshape_3d(ctx, rows, 1, width, count), index);
  if (!gpu_) {
    return values;
  }
  constexpr float kMax = std::numeric_limits<float>::max();
  const auto any_positive = [&](ggml_tensor* t) {
    return ggml_step(ctx, ggml_sum_rows(ctx, ggml_step(ctx, t)));
  };
  // x + FLT_MAX > 0 exactly where x > -FLT_MAX; x / 2 + FLT_MAX > 0 for every
  // x but -inf and NaN.
  ggml_tensor* found = any_positive(ggml_scale_bias(ctx, rows, 1.0f, kMax));
  ggml_tensor* holds_max = any_positive(ggml_scale_bias(ctx, rows, 0.5f, kMax));
  // -FLT_MAX where the row holds it, else -FLT_MAX - FLT_MAX = -inf.
  ggml_tensor* largest = ggml_scale_bias(
      ctx, ggml_scale_bias(ctx, holds_max, kMax, -kMax), 1.0f, -kMax);
  ggml_tensor* candidates = ggml_concat(
      ctx,
      ggml_reshape_3d(ctx, values, 1, 1, count),
      ggml_reshape_3d(ctx, largest, 1, 1, count),
      /*dim=*/1);
  ggml_tensor* picks = ggml_cast(
      ctx,
      ggml_reshape_2d(ctx, ggml_scale_bias(ctx, found, -1.0f, 1.0f), 1, count),
      GGML_TYPE_I32);
  return ggml_get_rows(ctx, candidates, picks);
}

// `values`, one per row of `rows`, with NaN where the row holds one: a row
// select picks the row's sum there, NaN too.
ggml_tensor* GraphBuilder::nan_where_rows_hold_nan(
    ggml_tensor* rows,
    ggml_tensor* values) {
  ggml_context* ctx = instance_.context.get();
  const int64_t count = rows->ne[1];
  ggml_tensor* candidates = ggml_concat(
      ctx,
      ggml_reshape_3d(ctx, values, 1, 1, count),
      ggml_reshape_3d(ctx, ggml_sum_rows(ctx, rows), 1, 1, count),
      /*dim=*/1);
  ggml_tensor* has_nan = ggml_step(ctx, ggml_sum_rows(ctx, nan_mask(rows)));
  ggml_tensor* picks =
      ggml_cast(ctx, ggml_reshape_2d(ctx, has_nan, 1, count), GGML_TYPE_I32);
  return ggml_get_rows(ctx, candidates, picks);
}

// Each row's largest element. Unlike max pooling, which starts from -FLT_MAX,
// this keeps an all -inf row at -inf. argmax skips NaN, which eager returns
// for a row holding one.
void GraphBuilder::lower_amax(const Node& node) {
  auto [rows, sizes] = reduced_rows(node, "amax");
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* result =
      reshape(ctx, nan_where_rows_hold_nan(rows, row_max(rows)), sizes);
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

// ggml normalizes rows, so another dim is permuted to the end and back.
void GraphBuilder::lower_softmax(const Node& node) {
  TensorRef input = tensor_arg(node, "self", 0);
  // Computed in F32, as eager does for F16 and BF16; set() rounds back.
  input.tensor = half_as_f32(input.tensor);
  require_f32(node, input.tensor);
  const size_t rank = input.sizes.size();
  if (rank == 0) {
    throw std::runtime_error("ggml: softmax of a scalar is not supported");
  }
  const int64_t dim = normalize_dim(int_arg(node, "dim", 1, -1), rank);
  std::vector<int64_t> order;
  for (int64_t d = 0; d < static_cast<int64_t>(rank); ++d) {
    if (d != dim) {
      order.push_back(d);
    }
  }
  order.push_back(dim);
  const bool moved = dim != static_cast<int64_t>(rank) - 1;
  const TensorRef last = moved ? permuted(input, order) : input;
  ggml_context* ctx = instance_.context.get();
  const int64_t width = last.sizes.back();
  if (width == 0) {
    throw std::runtime_error("ggml: softmax over an empty dimension");
  }
  ggml_tensor* rows = ggml_reshape_2d(
      ctx,
      contiguous(last.tensor),
      width,
      static_cast<int64_t>(checked_numel(last.sizes)) / width);
  const ValueId id = output_ids(node).at(0);
  ggml_tensor* result = reshape(ctx, ggml_soft_max(ctx, rows), last.sizes);
  if (!moved) {
    set(id, TensorRef{result, input.sizes});
    return;
  }
  std::vector<int64_t> inverse(rank);
  for (size_t i = 0; i < rank; ++i) {
    inverse[static_cast<size_t>(order[i])] = static_cast<int64_t>(i);
  }
  set_view(id, permuted(TensorRef{result, last.sizes}, inverse));
}

namespace {

// The dtype eager compares two tensors in, as far as it matters here: two
// floating types in the wider one, F16 with BF16 in F32; a 0-d tensor takes a
// dimensioned one's dtype unless its category is higher. Two integer types
// compare exactly in F32 whatever their widths, so the left one stands in.
ScalarType compare_dtype(
    ScalarType a,
    bool a_dimensioned,
    ScalarType b,
    bool b_dimensioned) {
  const auto category = [](ScalarType t) {
    return t == ScalarType::Bool ? 0 : is_integer(t) ? 1 : 2;
  };
  if (a_dimensioned != b_dimensioned) {
    const ScalarType dimensioned = a_dimensioned ? a : b;
    const ScalarType zero_dim = a_dimensioned ? b : a;
    return category(zero_dim) > category(dimensioned) ? zero_dim : dimensioned;
  }
  if (a == b) {
    return a;
  }
  if (category(a) != category(b)) {
    return category(a) > category(b) ? a : b;
  }
  return category(a) == 2 ? ScalarType::Float : a;
}

} // namespace

// Comparisons produce 0/1 floats: a <= b is 1 - step(a - b) and a < b is
// step(b - a), with a and b in F32. a != b is step(step(a - b) + step(b - a)
// + n), where n is 1 where an operand is NaN, since equal infinities also
// differ by NaN. Each operand is first rounded to its own dtype (a lowering
// may have left an F16 result in F32), then to the dtype eager compares in
// when that is F16 or BF16; a scalar takes a tensor's dtype within its
// category. Two integers compare exactly, which F32 matches within +-2^24, so
// integer operands must be ones F32 holds exactly (see
// require_exact_integers). Every comparison also defers integers computed
// from those by adds and subtracts, such as the positions an attention mask
// compares, to the host, which bounds each integer of that computation within
// +-2^24 from constants and shapes, or else on every run. F32 then keeps the
// sign of a - b and whether it is 0, which is all a comparison reads.
void GraphBuilder::lower_compare(const Node& node, Comparison comparison) {
  const ValueId left_id = tensor_arg_id(node, "self", 0);
  TensorRef left = tensor_arg(node, "self", 0);
  const NamedArgument& other = require_arg(node, "other", 1);
  ggml_context* ctx = instance_.context.get();
  const std::string op = comparison == Comparison::Le ? "le"
      : comparison == Comparison::Lt                  ? "lt"
                                                      : "ne";
  const ScalarType left_dtype =
      method_.graph.value(left_id).tensor_meta().dtype;
  constexpr bool defer_to_host = true;
  const auto round_to = [&](ggml_tensor* t, ScalarType dtype) {
    if (dtype != ScalarType::Half && dtype != ScalarType::BFloat16) {
      return t;
    }
    const ggml_type half =
        dtype == ScalarType::Half ? GGML_TYPE_F16 : GGML_TYPE_BF16;
    return t->type == half ? t : ggml_cast(ctx, as_f32(t), half);
  };
  ggml_tensor* difference = nullptr;
  const bool equality = comparison == Comparison::Ne;
  // 1 where an operand is NaN, for ne.
  ggml_tensor* nan_operand = nullptr;
  std::vector<int64_t> sizes = left.sizes;
  if (other.arg.kind() == ArgKind::Tensor) {
    const ValueId right_id = other.arg.as_tensor().id;
    TensorRef right = tensor(right_id);
    const ScalarType right_dtype =
        method_.graph.value(right_id).tensor_meta().dtype;
    const ScalarType dtype = compare_dtype(
        left_dtype, !left.sizes.empty(), right_dtype, !right.sizes.empty());
    if (is_integer(dtype)) {
      require_exact_integers(left_id, op + "'s self", defer_to_host);
      require_exact_integers(right_id, op + "'s other", defer_to_host);
    }
    left.tensor = round_to(round_to(left.tensor, left_dtype), dtype);
    right.tensor = round_to(round_to(right.tensor, right_dtype), dtype);
    if (const std::optional<int64_t> extent = key_extent_of(node)) {
      // An attention mask over a key prefix: only the per-key operand
      // spans the key axis.
      left = last_axis_prefix(left, *extent);
      right = last_axis_prefix(right, *extent);
    }
    BinaryOperands operands = tensor_operands(left, right, /*force_f32=*/true);
    difference = ggml_sub(ctx, operands.lhs, operands.rhs);
    if (equality) {
      nan_operand =
          ggml_add(ctx, nan_mask(operands.lhs), nan_mask(operands.rhs));
    }
    sizes = std::move(operands.sizes);
  } else {
    auto scalar = static_cast<float>(float_arg(node, "other", 1, 0));
    if (is_integer(left_dtype) && other.arg.kind() == ArgKind::Int) {
      require_exact_integers(left_id, op + "'s self", defer_to_host);
      const int64_t value = int_value(other.arg);
      if (value < -kExactF32Integer || value > kExactF32Integer) {
        throw std::out_of_range(
            "ggml: " + op +
            "'s other is outside +-2^24, which F32 holds "
            "exactly");
      }
    } else if (left_dtype == ScalarType::Half) {
      scalar = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scalar));
    } else if (left_dtype == ScalarType::BFloat16) {
      scalar = ggml_bf16_to_fp32(ggml_fp32_to_bf16(scalar));
    }
    ggml_tensor* self = contiguous(as_f32(round_to(left.tensor, left_dtype)));
    difference = ggml_scale_bias(ctx, self, 1.0f, -scalar);
    if (equality) {
      // A NaN scalar makes every difference NaN.
      nan_operand = nan_mask(std::isnan(scalar) ? difference : self);
    }
  }
  ggml_tensor* result = nullptr;
  switch (comparison) {
    case Comparison::Le:
      result = ggml_scale_bias(ctx, ggml_step(ctx, difference), -1.0f, 1.0f);
      break;
    case Comparison::Lt:
      result = ggml_step(ctx, ggml_neg(ctx, difference));
      break;
    case Comparison::Ne:
      result = ggml_step(
          ctx,
          ggml_add(
              ctx,
              ggml_add(
                  ctx,
                  ggml_step(ctx, difference),
                  ggml_step(ctx, ggml_neg(ctx, difference))),
              nan_operand));
      break;
  }
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

// Sets `id` to `result` in the value's declared type. Bools stay 0/1 floats.
void GraphBuilder::set_as_output_type(ValueId id, TensorRef result) {
  const TensorMeta& meta = method_.graph.value(id).tensor_meta();
  if (meta.dtype != ScalarType::Bool) {
    const ggml_type type = compute_type(meta);
    if (result.tensor->type != type) {
      result.tensor = ggml_cast(instance_.context.get(), result.tensor, type);
    }
  }
  set(id, std::move(result));
}

// 1 where x is NaN, else 0, in F32: |x| + 1 is NaN only for a NaN x.
ggml_tensor* GraphBuilder::nan_mask(ggml_tensor* x) {
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* plus_one =
      ggml_scale_bias(ctx, ggml_abs(ctx, contiguous(as_f32(x))), 1.0f, 1.0f);
  return ggml_scale_bias(ctx, ggml_step(ctx, plus_one), -1.0f, 1.0f);
}

// The result is F32, which holds integers exactly up to 2^24, and ggml
// derives the length from the F32 bounds.
void GraphBuilder::lower_arange(const Node& node) {
  const double start = float_arg(node, "start", 0, 0.0);
  const double end = float_arg(node, "end", 1, 0.0);
  const double step = float_arg(node, "step", 2, 1.0, true);
  if (step <= 0.0 || end <= start) {
    throw std::runtime_error(
        "ggml: arange needs a positive step and end above start");
  }
  constexpr double kExactF32 = 1 << 24;
  if (std::abs(start) > kExactF32 || std::abs(end) > kExactF32) {
    throw std::runtime_error("ggml: arange bounds must be within 2^24");
  }
  ggml_tensor* result = ggml_arange(
      instance_.context.get(),
      static_cast<float>(start),
      static_cast<float>(end),
      static_cast<float>(step));
  if (result->ne[0] != static_cast<int64_t>(std::ceil((end - start) / step))) {
    throw std::runtime_error("ggml: arange length changes in F32");
  }
  set_single_output(node, TensorRef{result, {result->ne[0]}});
}

// softplus(x) = x where beta * x > threshold, else log1p(exp(beta * x)) / beta.
// ggml_softplus is the beta 1, threshold 20 case; its value past 20 is
// log1p(exp(y)) rounded to F32, so other betas and thresholds scale its input
// and output and pick x by row gather, as where does.
void GraphBuilder::lower_softplus(const Node& node) {
  const TensorRef input = tensor_arg(node, "self", 0);
  const double beta = float_arg(node, "beta", 1, 1.0);
  const double threshold = float_arg(node, "threshold", 2, 20.0);
  if (beta == 0.0) {
    throw std::runtime_error("ggml: softplus needs a nonzero beta");
  }
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* x = contiguous(as_f32(input.tensor));
  if (beta == 1.0 && threshold == 20.0) {
    set_as_output_type(
        output_ids(node).at(0), TensorRef{ggml_softplus(ctx, x), input.sizes});
    return;
  }
  ggml_tensor* y = ggml_scale(ctx, x, static_cast<float>(beta));
  ggml_tensor* soft =
      ggml_scale(ctx, ggml_softplus(ctx, y), static_cast<float>(1.0 / beta));
  const int64_t count = ggml_nelements(x);
  ggml_tensor* candidates = ggml_concat(
      ctx,
      ggml_reshape_3d(ctx, soft, 1, 1, count),
      ggml_reshape_3d(ctx, x, 1, 1, count),
      /*dim=*/1);
  ggml_tensor* past = ggml_step(
      ctx, ggml_scale_bias(ctx, y, 1.0f, -static_cast<float>(threshold)));
  ggml_tensor* picks =
      ggml_cast(ctx, ggml_reshape_2d(ctx, past, 1, count), GGML_TYPE_I32);
  set_as_output_type(
      output_ids(node).at(0),
      TensorRef{
          reshape(ctx, ggml_get_rows(ctx, candidates, picks), input.sizes),
          input.sizes});
}

} // namespace ptn::ggml_detail
