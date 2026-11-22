// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/FlattenLinearBatch.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/TensorMeta.h>
#include <executorch/backends/native/runtime/graph/Value.h>

namespace ptn::vulkan {
namespace {

Node view_copy(
    std::string name,
    ValueId source,
    ValueId result,
    std::vector<int64_t> sizes) {
  return Node{
      .name = std::move(name),
      .target = "torch.ops.aten.view_copy.default",
      .inputs =
          {
              {.name = "self", .arg = TensorArg{source}},
              {.name = "size", .arg = IntListArg{.values = std::move(sizes)}},
          },
      .outputs = {{.value_id = result}},
  };
}

// `meta` with its first two dims merged.
TensorMeta flattened(const TensorMeta& meta, const std::string& name) {
  int64_t batch = 0;
  if (__builtin_mul_overflow(meta.sizes[0], meta.sizes[1], &batch)) {
    throw std::runtime_error(
        "vulkan: linear '" + name + "' has batch dims whose product overflows");
  }
  return TensorMeta{
      .dtype = meta.dtype, .sizes = {batch, meta.sizes[2], meta.sizes[3]}};
}

} // namespace

size_t flatten_linear_batches(Graph& graph) {
  size_t count = 0;
  const std::vector<NodeId> schedule = graph.schedule;
  for (const NodeId node_id : schedule) {
    const Node& node = graph.node(node_id);
    if (!node.is_call() || node.target != "torch.ops.aten.linear.default" ||
        node.inputs.empty() || node.inputs[0].arg.kind() != ArgKind::Tensor ||
        node.outputs.size() != 1 || !valid(node.outputs[0].value_id)) {
      continue;
    }
    const ValueId input_id = node.inputs[0].arg.as_tensor().id;
    const ValueId output_id = node.outputs[0].value_id;
    const TensorMeta input_meta = graph.value(input_id).tensor_meta();
    const TensorMeta output_meta = graph.value(output_id).tensor_meta();
    if (input_meta.sizes.size() != 4) {
      continue;
    }
    if (output_meta.sizes.size() != 4) {
      throw std::runtime_error(
          "vulkan: linear '" + node.name + "' of a rank-4 input has a rank-" +
          std::to_string(output_meta.sizes.size()) + " output");
    }
    if (!input_meta.lower_bounds.empty()) {
      throw std::runtime_error(
          "vulkan: linear '" + node.name +
          "' of a dynamically shaped rank-4 input is not supported");
    }
    const std::string name = node.name;
    const TensorMeta flat_input = flattened(input_meta, name);
    const TensorMeta flat_output = flattened(output_meta, name);
    const ValueId flat_input_id =
        graph.append_value(Value(name + "_flat_input", flat_input));
    const ValueId flat_output_id =
        graph.append_value(Value(name + "_flat_output", flat_output));
    graph.insert_node_before(
        node_id,
        view_copy(
            name + "_flatten", input_id, flat_input_id, flat_input.sizes));
    Node& linear = graph.node(node_id);
    linear.inputs[0].arg = TensorArg{flat_input_id};
    linear.outputs[0].value_id = flat_output_id;
    graph.insert_node_after(
        node_id,
        view_copy(
            name + "_unflatten", flat_output_id, output_id, output_meta.sizes));
    ++count;
  }
  if (count > 0) {
    graph.rebuild_def_use();
  }
  return count;
}

} // namespace ptn::vulkan
