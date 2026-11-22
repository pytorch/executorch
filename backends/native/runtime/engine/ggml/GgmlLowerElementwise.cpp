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
  set_single_output(
      node,
      TensorRef{
          tanh ? ggml_gelu(ctx, input.tensor)
               : ggml_gelu_erf(ctx, input.tensor),
          input.sizes});
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

// Reduces a run of trailing dimensions, which ggml sees as one row.
std::pair<ggml_tensor*, std::vector<int64_t>> GraphBuilder::trailing_rows(
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
  const size_t first = rank - dims.size();
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] != static_cast<int64_t>(first + i)) {
      throw std::runtime_error(
          "ggml: " + op + " supports trailing dimensions only");
    }
  }
  const bool keepdim = bool_arg(node, "keepdim", 2, false);
  int64_t row = 1;
  std::vector<int64_t> sizes;
  for (size_t i = 0; i < rank; ++i) {
    if (i < first) {
      sizes.push_back(input.sizes[i]);
    } else {
      row *= input.sizes[i];
      if (keepdim) {
        sizes.push_back(1);
      }
    }
  }
  if (row == 0) {
    throw std::runtime_error("ggml: " + op + " over an empty dimension");
  }
  ggml_tensor* rows = ggml_reshape_2d(
      instance_.context.get(),
      contiguous(input.tensor),
      row,
      static_cast<int64_t>(checked_numel(input.sizes)) / row);
  return {rows, std::move(sizes)};
}

void GraphBuilder::lower_mean(const Node& node) {
  auto [rows, sizes] = trailing_rows(node, "mean");
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* result = reshape(ctx, ggml_mean(ctx, rows), sizes);
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

void GraphBuilder::lower_sum(const Node& node) {
  auto [rows, sizes] = trailing_rows(node, "sum");
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* result = reshape(ctx, ggml_sum_rows(ctx, rows), sizes);
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

// Gathers each row's argmax element. Unlike max pooling, which starts from
// -FLT_MAX, this keeps an all -inf row at -inf. NaN does not propagate.
void GraphBuilder::lower_amax(const Node& node) {
  auto [rows, sizes] = trailing_rows(node, "amax");
  ggml_context* ctx = instance_.context.get();
  const int64_t width = rows->ne[0];
  const int64_t count = rows->ne[1];
  ggml_tensor* index = ggml_reshape_2d(ctx, ggml_argmax(ctx, rows), 1, count);
  ggml_tensor* values =
      ggml_get_rows(ctx, ggml_reshape_3d(ctx, rows, 1, width, count), index);
  ggml_tensor* result = reshape(ctx, values, sizes);
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

// ggml normalizes each row, so only the last dimension.
void GraphBuilder::lower_softmax(const Node& node) {
  TensorRef input = tensor_arg(node, "self", 0);
  // Computed in F32, as eager does for F16 and BF16; set() rounds back.
  input.tensor = half_as_f32(input.tensor);
  require_f32(node, input.tensor);
  const size_t rank = input.sizes.size();
  if (rank == 0 ||
      normalize_dim(int_arg(node, "dim", 1, -1), rank) !=
          static_cast<int64_t>(rank) - 1) {
    throw std::runtime_error("ggml: softmax supports the last dimension only");
  }
  ggml_context* ctx = instance_.context.get();
  set_single_output(
      node,
      TensorRef{ggml_soft_max(ctx, contiguous(input.tensor)), input.sizes});
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
// step(b - a), with a and b in F32. Each operand is first rounded to its own
// dtype (a lowering may have left an F16 result in F32), then to the dtype
// eager compares in when that is F16 or BF16; a scalar takes a tensor's dtype
// within its category. Two integers compare exactly, which F32 matches within
// +-2^24; le and lt check the integers they can trace: attention masks compare
// positions the graph computes, which le and lt accepted before any check.
void GraphBuilder::lower_compare(const Node& node, Comparison comparison) {
  const ValueId left_id = tensor_arg_id(node, "self", 0);
  TensorRef left = tensor_arg(node, "self", 0);
  const NamedArgument& other = require_arg(node, "other", 1);
  ggml_context* ctx = instance_.context.get();
  const std::string op = comparison == Comparison::Le ? "le" : "lt";
  const ScalarType left_dtype =
      method_.graph.value(left_id).tensor_meta().dtype;
  constexpr bool traced_only = true;
  const auto round_to = [&](ggml_tensor* t, ScalarType dtype) {
    if (dtype != ScalarType::Half && dtype != ScalarType::BFloat16) {
      return t;
    }
    const ggml_type half =
        dtype == ScalarType::Half ? GGML_TYPE_F16 : GGML_TYPE_BF16;
    return t->type == half ? t : ggml_cast(ctx, as_f32(t), half);
  };
  ggml_tensor* difference = nullptr;
  std::vector<int64_t> sizes = left.sizes;
  if (other.arg.kind() == ArgKind::Tensor) {
    const ValueId right_id = other.arg.as_tensor().id;
    TensorRef right = tensor(right_id);
    const ScalarType right_dtype =
        method_.graph.value(right_id).tensor_meta().dtype;
    const ScalarType dtype = compare_dtype(
        left_dtype, !left.sizes.empty(), right_dtype, !right.sizes.empty());
    if (is_integer(dtype)) {
      require_exact_integers(left_id, op + "'s self", traced_only);
      require_exact_integers(right_id, op + "'s other", traced_only);
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
    sizes = std::move(operands.sizes);
  } else {
    auto scalar = static_cast<float>(float_arg(node, "other", 1, 0));
    if (is_integer(left_dtype) && other.arg.kind() == ArgKind::Int) {
      require_exact_integers(left_id, op + "'s self", traced_only);
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
    difference = ggml_scale_bias(
        ctx,
        contiguous(as_f32(round_to(left.tensor, left_dtype))),
        1.0f,
        -scalar);
  }
  ggml_tensor* result = nullptr;
  switch (comparison) {
    case Comparison::Le:
      result = ggml_scale_bias(ctx, ggml_step(ctx, difference), -1.0f, 1.0f);
      break;
    case Comparison::Lt:
      result = ggml_step(ctx, ggml_neg(ctx, difference));
      break;
  }
  set_single_output(node, TensorRef{result, std::move(sizes)});
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

} // namespace ptn::ggml_detail
