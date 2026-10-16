// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/MaterializeViewCopies.h>

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <utility>

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

// cppcheck-suppress unusedFunction
std::vector<NodeId> materialize_dynamic_view_copies(Graph& graph) {
  constexpr std::array<std::pair<std::string_view, std::string_view>, 5>
      kCopies{{
          {"torch.ops.aten.alias.default", "torch.ops.aten.clone.default"},
          {"torch.ops.aten.permute.default",
           "torch.ops.aten.permute_copy.default"},
          {"torch.ops.aten.select.int", "torch.ops.aten.select_copy.int"},
          {"torch.ops.aten.squeeze.dims", "torch.ops.aten.squeeze_copy.dims"},
          {"torch.ops.aten.unsqueeze.default",
           "torch.ops.aten.unsqueeze_copy.default"},
      }};
  std::vector<NodeId> rewritten;
  for (const NodeId node_id : graph.schedule) {
    Node& node = graph.node(node_id);
    if (!node.is_call() || node.outputs.size() != 1 ||
        node.outputs[0].kind != OutputValueKind::Tensor ||
        !valid(node.outputs[0].value_id)) {
      continue;
    }
    const auto copy = std::ranges::find_if(
        kCopies, [&](const auto& entry) { return entry.first == node.target; });
    const ValueId output_id = node.outputs[0].value_id;
    const ValueId source_id = graph.value(output_id).alias_id;
    if (copy == kCopies.end() || !valid(source_id) ||
        graph.value(source_id).tensor_meta().lower_bounds.empty() ||
        has_mutated_alias_use(graph, output_id)) {
      continue;
    }
    // clone takes a memory_format that alias does not have.
    if (node.target == "torch.ops.aten.alias.default") {
      node.inputs.push_back({.name = "memory_format", .arg = NoneArg{}});
    }
    // The copy kernels' resize functions index sizes with the dims as given.
    const int64_t rank =
        static_cast<int64_t>(
            graph.value(source_id).tensor_meta().sizes.size()) +
        (node.target == "torch.ops.aten.unsqueeze.default" ? 1 : 0);
    const auto non_negative = [rank](int64_t dim) {
      return dim < 0 ? dim + rank : dim;
    };
    for (NamedArgument& input : node.inputs) {
      if (input.name != "dim" && input.name != "dims") {
        continue;
      }
      if (input.arg.kind() == ArgKind::Int) {
        const IntArg& dim = input.arg.as_int();
        input.arg = IntArg{.value = non_negative(dim.value), .id = dim.id};
      } else if (input.arg.kind() == ArgKind::IntList) {
        IntListArg dims = input.arg.as_int_list();
        std::ranges::transform(dims.values, dims.values.begin(), non_negative);
        input.arg = std::move(dims);
      }
    }
    node.target = std::string(copy->second);
    graph.value(output_id).alias_id = kInvalid;
    rewritten.push_back(node_id);
  }
  return rewritten;
}

} // namespace ptn::vulkan
