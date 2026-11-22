// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlGraphBuilder.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>

#include <ggml.h>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlTypes.h>
#include <executorch/backends/native/runtime/graph/Value.h>

namespace ptn::ggml_detail {

// cuBLAS multiplies F16 operands in F16 unless asked for F32, a few 1e-2 off
// eager's F32 accumulation in long sums. The engine asks only on CUDA, and only
// when the cuda_f32_accumulation_for_testing option is set.
ggml_tensor* GraphBuilder::mul_mat(ggml_tensor* a, ggml_tensor* b) {
  ggml_tensor* result = ggml_mul_mat(instance_.context.get(), a, b);
  if (f32_accumulation_ &&
      (a->type == GGML_TYPE_F16 || b->type == GGML_TYPE_F16)) {
    ggml_prec_set_acc(result, GGML_PREC_F32);
  }
  return result;
}

void GraphBuilder::lower_linear(const Node& node) {
  const TensorRef input = tensor_arg(node, "input", 0);
  const TensorRef weight = tensor_arg(node, "weight", 1);
  if (weight.sizes.size() != 2 || input.sizes.empty() ||
      input.sizes.back() != weight.sizes[1]) {
    throw std::runtime_error("ggml: invalid linear shapes");
  }
  // ggml-cpu multiplies quantized or mixed-type weights only by F32
  // activations. mul_mat returns F32 whatever its operand types.
  ggml_tensor* activation = input.tensor->type == GGML_TYPE_F32 ||
          input.tensor->type == weight.tensor->type
      ? input.tensor
      : as_f32(input.tensor);
  ggml_tensor* result = mul_mat(weight.tensor, activation);
  if (const std::optional<TensorRef> bias = optional_tensor(node, "bias", 2)) {
    if (bias->sizes != std::vector<int64_t>{weight.sizes[0]}) {
      throw std::runtime_error(
          "ggml: linear bias must hold one value per output feature");
    }
    result = ggml_add(instance_.context.get(), result, as_f32(bias->tensor));
  }
  std::vector<int64_t> sizes = input.sizes;
  sizes.back() = weight.sizes[0];
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

namespace {

// The entries of aten.index.Tensor's `indices` list; invalid ones are None.
std::vector<ValueId> index_entries(const Node& node) {
  const NamedArgument* arg = find_arg(node, "indices", 1);
  if (arg == nullptr) {
    return {};
  }
  if (arg->arg.kind() == ArgKind::TensorList) {
    return arg->arg.as_tensor_list().ids;
  }
  if (arg->arg.kind() == ArgKind::OptionalTensorList) {
    return arg->arg.as_optional_tensor_list().ids;
  }
  return {};
}

// The single index tensor of an aten.index.Tensor, which must index dim 0.
ValueId row_index_id(const Node& node) {
  const std::vector<ValueId> ids = index_entries(node);
  if (std::ranges::count_if(ids, [](ValueId id) { return valid(id); }) != 1) {
    throw std::runtime_error("ggml: index supports a single index tensor");
  }
  if (ids.size() != 1) {
    throw std::runtime_error("ggml: index supports indexing dim 0 only");
  }
  return ids.front();
}

// Whether an aten.index.Tensor is x[:, rows] of a [1, T, D] x, as Hugging Face
// selects logits_to_keep rows.
bool leading_one_index(const Node& node, const std::vector<int64_t>& sizes) {
  const std::vector<ValueId> ids = index_entries(node);
  return ids.size() == 2 && !valid(ids.front()) && valid(ids.back()) &&
      sizes.size() == 3 && sizes.front() == 1;
}

const Value* tensor_input(
    const Method& method,
    const Node& node,
    std::string_view name,
    size_t positional) {
  const NamedArgument* arg = find_arg(node, name, positional);
  if (arg == nullptr || arg->arg.kind() != ArgKind::Tensor ||
      !valid(arg->arg.as_tensor().id)) {
    return nullptr;
  }
  const Value& value = method.graph.value(arg->arg.as_tensor().id);
  return value.is_tensor() ? &value : nullptr;
}

} // namespace

const Node* expert_matmul(const Method& method, const Node& index) {
  if (!index.is_call() || op_key(index.target) != "aten.index.Tensor") {
    return nullptr;
  }
  const NamedArgument* self = find_arg(index, "self", 0);
  const std::vector<ValueId> ids = index_entries(index);
  if (self == nullptr || self->arg.kind() != ArgKind::Tensor) {
    return nullptr;
  }
  const ValueId weight_id = self->arg.as_tensor().id;
  const std::vector<ValueId> outputs = output_ids(index);
  if (!valid(weight_id) || ids.size() != 1 || !valid(ids.front()) ||
      outputs.size() != 1 ||
      std::ranges::find(method.graph.output_ids, outputs.front()) !=
          method.graph.output_ids.end()) {
    return nullptr;
  }
  const Value& weight = method.graph.value(weight_id);
  const Value& expert_ids = method.graph.value(ids.front());
  const Value& selected = method.graph.value(outputs.front());
  const bool constant =
      std::ranges::any_of(method.data_bindings, [&](const DataBinding& b) {
        return b.value_id == weight_id && b.has_data;
      });
  if (!constant || !weight.is_tensor() || weight.tensor_meta().ndim() != 3 ||
      !expert_ids.is_tensor() || expert_ids.tensor_meta().ndim() != 1 ||
      !is_integer(expert_ids.tensor_meta().dtype) ||
      selected.consumer_ids.size() != 1) {
    return nullptr;
  }
  const Node& bmm = method.graph.node(selected.consumer_ids.front());
  const Value* left = tensor_input(method, bmm, "self", 0);
  const Value* right = tensor_input(method, bmm, "mat2", 1);
  if (!bmm.is_call() || op_key(bmm.target) != "aten.bmm.default" ||
      left != &selected || right == nullptr || right == &selected) {
    return nullptr;
  }
  const std::vector<int64_t>& sizes = right->tensor_meta().sizes;
  return sizes.size() == 3 && sizes[1] == weight.tensor_meta().sizes[2] &&
          sizes[2] == 1
      ? &bmm
      : nullptr;
}

const Node* expert_gather(const Method& method, const Node& matmul) {
  const NamedArgument* self = find_arg(matmul, "self", 0);
  if (self == nullptr || self->arg.kind() != ArgKind::Tensor ||
      !valid(self->arg.as_tensor().id)) {
    return nullptr;
  }
  const NodeId producer =
      method.graph.value(self->arg.as_tensor().id).producer_id;
  if (!valid(producer)) {
    return nullptr;
  }
  const Node& index = method.graph.node(producer);
  return expert_matmul(method, index) == &matmul ? &index : nullptr;
}

// y[s] = W[ids[s]] @ x[s] as ggml_mul_mat_id over W's [I, O, E] layout, so the
// gathered [S, O, I] weights are never materialized.
void GraphBuilder::lower_expert_matmul(const Node& bmm, const Node& index) {
  const TensorRef weight = tensor_arg(index, "self", 0);
  const ValueId ids_id = row_index_id(index);
  const TensorRef ids = tensor(ids_id);
  const TensorRef input = tensor_arg(bmm, "mat2", 1);
  const int64_t experts = weight.sizes[0];
  const int64_t out = weight.sizes[1];
  const int64_t in = weight.sizes[2];
  const int64_t count = ids.sizes.at(0);
  if (input.sizes != std::vector<int64_t>{count, in, 1}) {
    throw std::runtime_error(
        "ggml: expert bmm needs activations [S, I, 1] for S expert ids");
  }
  check_row_indices(ids_id, experts);
  ggml_context* ctx = instance_.context.get();
  // ggml-cpu's mul_mat_id reads F32 activations only.
  ggml_tensor* result = ggml_mul_mat_id(
      ctx,
      weight.tensor,
      reshape(ctx, as_f32(input.tensor), {count, 1, in}),
      reshape(ctx, as_i32(ids.tensor), {count, 1}));
  std::vector<int64_t> sizes = {count, out, 1};
  set_single_output(bmm, TensorRef{reshape(ctx, result, sizes), sizes});
}

// Whole slices along dim 0 of a tensor of any rank, gathered as rows of its
// [N, rest] view. x[:, rows] of a [1, T, D] x gathers rows of the [T, D]
// matrix and keeps the leading 1.
void GraphBuilder::lower_row_gather(const Node& node) {
  const bool embedding = op_key(node.target) == "aten.embedding.default";
  const ValueId weight_id = tensor_arg_id(node, "weight", 0);
  TensorRef weight = tensor(weight_id);
  const bool leading_one = !embedding && leading_one_index(node, weight.sizes);
  if (leading_one) {
    weight.sizes.erase(weight.sizes.begin());
  }
  const ValueId indices_id = embedding ? tensor_arg_id(node, "indices", 1)
      : leading_one                    ? index_entries(node).back()
                                       : row_index_id(node);
  const ScalarType dtype = method_.graph.value(indices_id).tensor_meta().dtype;
  if (weight.sizes.empty() || (embedding && weight.sizes.size() != 2) ||
      (dtype != ScalarType::Int && dtype != ScalarType::Long)) {
    throw std::runtime_error(
        "ggml: row gather needs a tensor and integer indices");
  }
  TensorRef indices = tensor(indices_id);
  check_row_indices(indices_id, weight.sizes[0]);
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* flat =
      reshape(ctx, as_i32(indices.tensor), {ggml_nelements(indices.tensor)});
  const std::vector<int64_t> slice(
      weight.sizes.begin() + 1, weight.sizes.end());
  const int64_t width = static_cast<int64_t>(checked_numel(slice));
  std::vector<int64_t> sizes = indices.sizes;
  sizes.insert(sizes.end(), slice.begin(), slice.end());
  ggml_tensor* table = weight.sizes.size() == 2
      ? weight.tensor
      : reshape(ctx, weight.tensor, {weight.sizes[0], width});
  ggml_tensor* rows = ggml_get_rows(ctx, table, flat);
  const auto found = row_scales_.find(weight_id);
  if (found != row_scales_.end()) {
    const int64_t groups = found->second->ne[0];
    const int64_t count = ggml_nelements(flat);
    rows = ggml_mul(
        ctx,
        ggml_reshape_3d(ctx, rows, width / groups, groups, count),
        ggml_reshape_3d(
            ctx, ggml_get_rows(ctx, found->second, flat), 1, groups, count));
  }
  if (leading_one) {
    sizes.insert(sizes.begin(), 1);
  }
  ggml_tensor* result = reshape(ctx, rows, sizes);
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

void GraphBuilder::lower_matmul(const Node& node) {
  const TensorRef left = tensor_arg(node, "self", 0);
  const TensorRef right = view_arg(node, "other", 1);
  if (left.sizes.size() < 2 || right.sizes.size() < 2) {
    throw std::runtime_error("ggml: matmul requires rank >= 2");
  }
  if (left.sizes.back() != right.sizes[right.sizes.size() - 2]) {
    throw std::runtime_error("ggml: matmul contracted dims differ");
  }
  // mul_mat broadcasts only its first operand, here the right batch.
  const size_t left_batch = left.sizes.size() - 2;
  const size_t right_batch = right.sizes.size() - 2;
  bool broadcastable = right_batch <= left_batch;
  for (size_t i = 1; broadcastable && i <= right_batch; ++i) {
    const int64_t dim = right.sizes[right_batch - i];
    broadcastable = dim == 1 || dim == left.sizes[left_batch - i];
  }
  if (!broadcastable) {
    throw std::runtime_error("ggml: matmul cannot broadcast the left batch");
  }
  ggml_context* ctx = instance_.context.get();
  // mul_mat(a, b) contracts ne[0] of both, so [M,K] @ [K,N] is
  // mul_mat(right^T, left). reshape() copies right^T only when it is strided
  // and also resets the strides of size-1 dims, which ggml_is_contiguous
  // ignores but mul_mat's transposed assert reads.
  std::vector<int64_t> right_t_sizes = right.sizes;
  std::swap(right_t_sizes.back(), right_t_sizes[right_batch]);
  ggml_tensor* right_t =
      reshape(ctx, ggml_transpose(ctx, right.tensor), right_t_sizes);
  ggml_tensor* result = mul_mat(right_t, left.tensor);
  std::vector<int64_t> sizes = left.sizes;
  sizes.back() = right.sizes.back();
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

DynamicQuant GraphBuilder::dynamic_quant_arg(
    const Node& node,
    std::string_view name,
    size_t positional) const {
  const Argument& arg = require_arg(node, name, positional).arg;
  const auto* quant = arg.kind() == ArgKind::Tensor
      ? std::get_if<DynamicQuant>(&value(arg.as_tensor().id))
      : nullptr;
  if (quant == nullptr) {
    throw std::runtime_error(
        "ggml: '" + node.target +
        "' is only supported inside a dynamic activation quant chain");
  }
  return *quant;
}

void GraphBuilder::set_dynamic_quant(
    const Node& node,
    const DynamicQuant& quant) {
  for (const ValueId id : output_ids(node)) {
    set(id, quant);
  }
}

// ggml backends quantize the activations of a quantized matmul themselves,
// in their own block format, so torchao's per-token int8 round trip is
// dropped rather than emulated.
void GraphBuilder::lower_choose_qparams(const Node& node) {
  const TensorRef input = tensor_arg(node, "input", 0);
  const Argument& target_dtype = require_arg(node, "target_dtype", 3).arg;
  if (target_dtype.kind() != ArgKind::ScalarType ||
      target_dtype.as_scalar_type().value != ScalarType::Char) {
    throw std::runtime_error(
        "ggml: dynamic activation quantization must target int8");
  }
  std::vector<int64_t> per_token(input.sizes.size(), 1);
  per_token.back() = input.sizes.back();
  if (int_list(require_arg(node, "block_size", 2).arg) != per_token) {
    throw std::runtime_error(
        "ggml: only per-token dynamic activation quantization is supported");
  }
  set_dynamic_quant(node, DynamicQuant{input});
}

void GraphBuilder::lower_quantize_affine(const Node& node) {
  const DynamicQuant quant = dynamic_quant_arg(node, "scale", 2);
  if (tensor_arg(node, "input", 0).tensor != quant.input.tensor) {
    throw std::runtime_error(
        "ggml: quantize_affine input differs from its qparams source");
  }
  set_dynamic_quant(node, quant);
}

void GraphBuilder::lower_dequantize_affine(const Node& node) {
  const DynamicQuant quant = dynamic_quant_arg(node, "input", 0);
  if (dynamic_quant_arg(node, "scale", 2).input.tensor != quant.input.tensor) {
    throw std::runtime_error(
        "ggml: dequantize_affine scale differs from its input chain");
  }
  const std::vector<ValueId> ids = output_ids(node);
  if (ids.size() != 1) {
    throw std::runtime_error("ggml: dequantize_affine must produce one tensor");
  }
  if (std::ranges::find(method_.graph.output_ids, ids.front()) !=
      method_.graph.output_ids.end()) {
    throw std::runtime_error(
        "ggml: dynamically quantized activations must not be graph outputs");
  }
  for (const NodeId consumer : method_.graph.value(ids.front()).consumer_ids) {
    const Node& linear = method_.graph.node(consumer);
    const NamedArgument* input = find_arg(linear, "input", 0);
    if (op_key(linear.target) != "aten.linear.default" || input == nullptr ||
        input->arg.kind() != ArgKind::Tensor ||
        input->arg.as_tensor().id != ids.front() ||
        !ggml_is_quantized(tensor_arg(linear, "weight", 1).tensor->type)) {
      throw std::runtime_error(
          "ggml: dynamically quantized activations must feed quantized "
          "linear inputs");
    }
  }
  set(ids.front(), quant.input);
}

} // namespace ptn::ggml_detail
