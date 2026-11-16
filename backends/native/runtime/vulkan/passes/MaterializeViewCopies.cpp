// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/MaterializeViewCopies.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/utils/Aliases.h>

namespace ptn::vulkan {
namespace {

bool materialize_view_copy(Graph& graph, Node& node) {
  constexpr std::array<std::pair<std::string_view, std::string_view>, 8>
      kCopies{{
          {"torch.ops.aten.view.default", "torch.ops.aten.view_copy.default"},
          {"torch.ops.aten.alias.default", "torch.ops.aten.clone.default"},
          {"torch.ops.aten.permute.default",
           "torch.ops.aten.permute_copy.default"},
          // ET-VK has no transpose_copy; the equivalent permute replaces it.
          {"torch.ops.aten.transpose.int",
           "torch.ops.aten.permute_copy.default"},
          {"torch.ops.aten.select.int", "torch.ops.aten.select_copy.int"},
          {"torch.ops.aten.squeeze.dim", "torch.ops.aten.squeeze_copy.dims"},
          {"torch.ops.aten.squeeze.dims", "torch.ops.aten.squeeze_copy.dims"},
          {"torch.ops.aten.unsqueeze.default",
           "torch.ops.aten.unsqueeze_copy.default"},
      }};
  if (!node.is_call() || node.outputs.size() != 1 ||
      node.outputs[0].kind != OutputValueKind::Tensor ||
      !valid(node.outputs[0].value_id)) {
    return false;
  }
  const auto copy = std::ranges::find_if(
      kCopies, [&](const auto& entry) { return entry.first == node.target; });
  const ValueId output_id = node.outputs[0].value_id;
  const ValueId source_id = graph.value(output_id).alias_id;
  if (copy == kCopies.end() || !valid(source_id) ||
      has_mutated_alias_use(graph, output_id)) {
    return false;
  }
  // clone takes a memory_format that alias does not have.
  if (node.target == "torch.ops.aten.alias.default") {
    node.inputs.push_back({.name = "memory_format", .arg = NoneArg{}});
  }
  if (node.target == "torch.ops.aten.squeeze.dim") {
    for (NamedArgument& input : node.inputs) {
      if (input.name == "dim") {
        const IntArg& dim = input.arg.as_int();
        input.arg = IntListArg{
            .values = {dim.value},
            .ids = valid(dim.id) ? std::vector<ValueId>{dim.id}
                                 : std::vector<ValueId>{}};
      }
    }
  }
  // The copy kernels' resize functions index sizes with the dims as given.
  const int64_t rank =
      static_cast<int64_t>(graph.value(source_id).tensor_meta().sizes.size()) +
      (node.target == "torch.ops.aten.unsqueeze.default" ? 1 : 0);
  const auto non_negative = [rank](int64_t dim) {
    return dim < 0 ? dim + rank : dim;
  };
  if (node.target == "torch.ops.aten.transpose.int") {
    std::vector<int64_t> dims(static_cast<size_t>(rank));
    std::iota(dims.begin(), dims.end(), int64_t{0});
    const auto dim_at = [&](std::string_view name) {
      const auto input =
          std::ranges::find(node.inputs, name, &NamedArgument::name);
      return static_cast<size_t>(non_negative(input->arg.as_int().value));
    };
    std::swap(dims.at(dim_at("dim0")), dims.at(dim_at("dim1")));
    std::erase_if(node.inputs, [](const NamedArgument& input) {
      return input.name == "dim0" || input.name == "dim1";
    });
    node.inputs.push_back(
        {.name = "dims", .arg = IntListArg{.values = std::move(dims)}});
  }
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
  return true;
}

} // namespace

// cppcheck-suppress unusedFunction
size_t materialize_view_copies(
    Graph& graph,
    const std::vector<ValueId>& value_ids) {
  size_t count = 0;
  for (const NodeId node_id : graph.schedule) {
    Node& node = graph.node(node_id);
    if (node.outputs.size() == 1 &&
        std::ranges::find(value_ids, node.outputs[0].value_id) !=
            value_ids.end() &&
        materialize_view_copy(graph, node)) {
      ++count;
    }
  }
  return count;
}

// cppcheck-suppress unusedFunction
std::vector<NodeId> materialize_dynamic_view_copies(Graph& graph) {
  std::vector<NodeId> rewritten;
  for (const NodeId node_id : graph.schedule) {
    Node& node = graph.node(node_id);
    if (!node.is_call() || node.target == "torch.ops.aten.view.default" ||
        node.outputs.size() != 1 ||
        node.outputs[0].kind != OutputValueKind::Tensor ||
        !valid(node.outputs[0].value_id)) {
      continue;
    }
    const ValueId source_id = graph.value(node.outputs[0].value_id).alias_id;
    if (valid(source_id) &&
        !graph.value(source_id).tensor_meta().lower_bounds.empty() &&
        materialize_view_copy(graph, node)) {
      rewritten.push_back(node_id);
    }
  }
  return rewritten;
}

} // namespace ptn::vulkan
