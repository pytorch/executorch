// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/MaterializeViewCopies.h>

#include <algorithm>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/Node.h>

namespace ptn::vulkan {
namespace {

ValueId alias_root(const Graph& graph, ValueId value_id) {
  for (size_t depth = 0; valid(value_id) && depth <= graph.values.size();
       ++depth) {
    const ValueId source_id = graph.value(value_id).alias_id;
    if (!valid(source_id)) {
      return value_id;
    }
    value_id = source_id;
  }
  return kInvalid;
}

bool references_alias_group(
    const Graph& graph,
    const Argument& argument,
    ValueId root_id) {
  if (argument.kind() == ArgKind::Tensor) {
    return alias_root(graph, argument.as_tensor().id) == root_id;
  }
  if (argument.kind() == ArgKind::TensorList) {
    return std::ranges::any_of(argument.as_tensor_list().ids, [&](ValueId id) {
      return alias_root(graph, id) == root_id;
    });
  }
  if (argument.kind() == ArgKind::OptionalTensorList) {
    return std::ranges::any_of(
        argument.as_optional_tensor_list().ids,
        [&](ValueId id) { return alias_root(graph, id) == root_id; });
  }
  return false;
}

bool has_mutated_alias_use(const Graph& graph, ValueId value_id) {
  const ValueId root_id = alias_root(graph, value_id);
  return std::ranges::any_of(graph.schedule, [&](NodeId node_id) {
    return std::ranges::any_of(
        graph.node(node_id).inputs, [&](const NamedArgument& input) {
          return input.mutated &&
              references_alias_group(graph, input.arg, root_id);
        });
  });
}

} // namespace

// cppcheck-suppress unusedFunction
size_t materialize_view_copies(
    Graph& graph,
    const std::vector<ValueId>& value_ids) {
  size_t count = 0;
  for (const NodeId node_id : graph.schedule) {
    Node& node = graph.node(node_id);
    if (!node.is_call() || node.target != "torch.ops.aten.view.default" ||
        node.outputs.size() != 1 ||
        node.outputs[0].kind != OutputValueKind::Tensor ||
        !valid(node.outputs[0].value_id)) {
      continue;
    }
    const ValueId output_id = node.outputs[0].value_id;
    if (std::ranges::find(value_ids, output_id) == value_ids.end() ||
        has_mutated_alias_use(graph, output_id)) {
      continue;
    }
    node.target = "torch.ops.aten.view_copy.default";
    graph.value(output_id).alias_id = kInvalid;
    ++count;
  }
  return count;
}

} // namespace ptn::vulkan
