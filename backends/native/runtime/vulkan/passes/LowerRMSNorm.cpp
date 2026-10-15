// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/LowerRMSNorm.h>

#include <algorithm>
#include <cmath>
#include <string_view>
#include <vector>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/ScalarType.h>
#include <executorch/backends/native/runtime/graph/Value.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

namespace ptn::vulkan {
namespace {

constexpr std::string_view kAtenRMSNorm = "torch.ops.aten.rms_norm.default";
constexpr const char* kVulkanRMSNorm = "torch.ops.et_vk.rms_norm.default";

const Argument*
get_arg(const Node& node, std::string_view name, size_t position) {
  const auto found = std::ranges::find_if(
      node.inputs,
      [name](const NamedArgument& input) { return input.name == name; });
  if (found != node.inputs.end()) {
    return &found->arg;
  }
  return position < node.inputs.size() ? &node.inputs[position].arg : nullptr;
}

bool is_constant(const Value& value) {
  return value.role == ValueRole::Parameter ||
      value.role == ValueRole::ConstantTensor;
}

} // namespace

size_t lower_rms_norms(Graph& graph) {
  size_t count = 0;
  for (const NodeId node_id : graph.schedule) {
    Node& node = graph.node(node_id);
    if (!node.is_call() || node.target != kAtenRMSNorm ||
        node.outputs.size() != 1 || !valid(node.outputs[0].value_id)) {
      continue;
    }

    const Argument* input = get_arg(node, "input", 0);
    const Argument* normalized_shape = get_arg(node, "normalized_shape", 1);
    const Argument* weight = get_arg(node, "weight", 2);
    const Argument* eps = get_arg(node, "eps", 3);
    if (input == nullptr || input->kind() != ArgKind::Tensor ||
        normalized_shape == nullptr ||
        normalized_shape->kind() != ArgKind::IntList || weight == nullptr ||
        weight->kind() != ArgKind::Tensor || eps == nullptr ||
        eps->kind() != ArgKind::Float || valid(eps->as_float().id) ||
        !std::isfinite(eps->as_float().value) || eps->as_float().value < 0) {
      continue;
    }

    const ValueId input_id = input->as_tensor().id;
    const ValueId weight_id = weight->as_tensor().id;
    const ValueId output_id = node.outputs[0].value_id;
    if (!valid(input_id) || !valid(weight_id)) {
      continue;
    }
    const Value& input_value = graph.value(input_id);
    const Value& weight_value = graph.value(weight_id);
    const Value& output_value = graph.value(output_id);
    if (!input_value.is_tensor() || !weight_value.is_tensor() ||
        !output_value.is_tensor() || !is_constant(weight_value)) {
      continue;
    }

    const TensorMeta& input_meta = input_value.tensor_meta();
    const TensorMeta& weight_meta = weight_value.tensor_meta();
    const TensorMeta& output_meta = output_value.tensor_meta();
    const IntListArg& shape = normalized_shape->as_int_list();
    if (input_meta.sizes.empty() || !shape.ids.empty() ||
        shape.values != std::vector<int64_t>{input_meta.sizes.back()} ||
        weight_meta.sizes != shape.values ||
        (input_meta.dtype != ScalarType::Float &&
         input_meta.dtype != ScalarType::Half) ||
        weight_meta.dtype != input_meta.dtype ||
        output_meta.dtype != input_meta.dtype ||
        output_meta.sizes != input_meta.sizes) {
      continue;
    }

    const Argument input_arg = *input;
    const Argument weight_arg = *weight;
    const Argument eps_arg = *eps;
    node.target = kVulkanRMSNorm;
    node.inputs = {
        {.name = "input", .arg = input_arg},
        {.name = "weight", .arg = weight_arg},
        {.name = "epsilon", .arg = eps_arg},
    };
    ++count;
  }
  if (count > 0) {
    graph.rebuild_def_use();
    validate_graph(graph);
  }
  return count;
}

} // namespace ptn::vulkan
