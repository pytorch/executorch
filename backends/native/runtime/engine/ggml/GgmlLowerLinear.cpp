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

namespace ptn::ggml_detail {

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
  ggml_tensor* result =
      ggml_mul_mat(instance_.context.get(), weight.tensor, activation);
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

void GraphBuilder::lower_embedding(const Node& node) {
  const TensorRef weight = tensor_arg(node, "weight", 0);
  TensorRef indices = tensor_arg(node, "indices", 1);
  check_row_indices(
      require_arg(node, "indices", 1).arg.as_tensor().id, weight.sizes.at(0));
  if (indices.tensor->type != GGML_TYPE_I32) {
    indices.tensor =
        ggml_cast(instance_.context.get(), indices.tensor, GGML_TYPE_I32);
  }
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* flat =
      reshape(ctx, indices.tensor, {ggml_nelements(indices.tensor)});
  std::vector<int64_t> sizes = indices.sizes;
  sizes.push_back(weight.sizes.at(1));
  ggml_tensor* result =
      reshape(ctx, ggml_get_rows(ctx, weight.tensor, flat), sizes);
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
  ggml_tensor* result = ggml_mul_mat(ctx, right_t, left.tensor);
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
