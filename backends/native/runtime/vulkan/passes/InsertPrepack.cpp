// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/InsertPrepack.h>

#include <algorithm>
#include <array>
#include <string_view>
#include <vector>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

namespace ptn::vulkan {
namespace {

bool needs_prepack(const Node& node) {
  constexpr std::array<std::string_view, 3> kOps{
      "torch.ops.aten.add.Tensor",
      "torch.ops.aten.index.Tensor",
      "torch.ops.aten.mul.Tensor",
  };
  return node.is_call() && std::ranges::find(kOps, node.target) != kOps.end();
}

Node make_prepack_node(ValueId input, ValueId output, const std::string& name) {
  return Node{
      .name = name,
      .target = "torch.ops.et_vk.prepack.default",
      .inputs = {{.name = "self", .arg = TensorArg{input}}},
      .outputs = {{.value_id = output}},
  };
}

} // namespace

// cppcheck-suppress unusedFunction
size_t insert_prepack_nodes(Method& method) {
  Graph& graph = method.graph;
  size_t count = 0;
  for (const DataBinding& binding : method.data_bindings) {
    if (!binding.has_data || binding.mutated || !valid(binding.value_id)) {
      continue;
    }

    const Value& source = graph.value(binding.value_id);
    if (!source.is_tensor()) {
      continue;
    }
    std::vector<NodeId> consumers;
    for (const NodeId consumer_id : source.consumer_ids) {
      if (needs_prepack(graph.node(consumer_id))) {
        consumers.push_back(consumer_id);
      }
    }
    if (consumers.empty()) {
      continue;
    }

    const std::string source_name = source.name;
    Value prepacked(source_name + "_vulkan_prepacked", source.tensor_meta());
    prepacked.role = ValueRole::ConstantTensor;
    const ValueId prepacked_id = graph.append_value(std::move(prepacked));
    graph.insert_node_before(
        consumers.front(),
        make_prepack_node(
            binding.value_id, prepacked_id, source_name + "_vulkan_prepack"));
    for (const NodeId consumer_id : consumers) {
      graph.replace_input(consumer_id, binding.value_id, prepacked_id);
    }
    ++count;
  }
  if (count > 0) {
    stable_topological_sort(graph);
    validate_graph(graph);
  }
  return count;
}

} // namespace ptn::vulkan
