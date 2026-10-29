// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/utils/Aliases.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Node.h>

namespace ptn {
namespace {

constexpr std::string_view kView = "torch.ops.aten.view.default";
constexpr std::array<std::string_view, 3> kInPlaceActivations{
    "torch.ops.aten.relu_.default",
    "torch.ops.aten.gelu_.default",
    "torch.ops.aten.sigmoid_.default"};
constexpr std::array<std::pair<std::string_view, std::string_view>, 3>
    kViewCopies{
        {{kView, "torch.ops.aten.view_copy.default"},
         {"torch.ops.aten.permute.default",
          "torch.ops.aten.permute_copy.default"},
         {"torch.ops.aten.transpose.int",
          "torch.ops.aten.transpose_copy.int"}}};

[[noreturn]] void fail(const std::string& reason) {
  throw std::runtime_error("functionalize_aliases: " + reason);
}

bool is_in_place_activation(std::string_view target) {
  return std::ranges::find(kInPlaceActivations, target) !=
      kInPlaceActivations.end();
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

// Values with the same key, dtype and sizes read the same elements in the same
// logical order: a view reshapes its operand in row-major order, and an
// in-place activation keeps its operand's indexing. Any other alias starts a
// new key. Keys follow producer operands, not alias_id, which may name a more
// distant ancestor.
std::vector<ValueId> indexing_keys(const Graph& graph) {
  std::vector<ValueId> keys(graph.values.size());
  for (ValueId id = 0; std::cmp_less(id, graph.values.size()); ++id) {
    ValueId key = id;
    while (valid(graph.value(key).alias_id) &&
           valid(graph.value(key).producer_id)) {
      const auto& producer = graph.node(graph.value(key).producer_id);
      if ((producer.target != kView &&
           !is_in_place_activation(producer.target)) ||
          producer.inputs.empty() ||
          producer.inputs[0].arg.kind() != ArgKind::Tensor) {
        break;
      }
      key = producer.inputs[0].arg.as_tensor().id;
    }
    keys.at(id) = key;
  }
  return keys;
}

bool same_indexing(
    const Graph& graph,
    const std::vector<ValueId>& keys,
    ValueId left,
    ValueId right) {
  const auto& left_meta = graph.value(left).tensor_meta();
  const auto& right_meta = graph.value(right).tensor_meta();
  return keys.at(left) == keys.at(right) &&
      left_meta.dtype == right_meta.dtype &&
      left_meta.sizes == right_meta.sizes;
}

// as_strided reads storage, so its operand must index its root's storage in
// logical order.
bool contiguous_reshape(
    const Node& node,
    const Graph& graph,
    const std::vector<ValueId>& roots,
    const std::vector<ValueId>& keys) {
  if (node.inputs.size() != 4 ||
      node.inputs[1].arg.kind() != ArgKind::IntList ||
      node.inputs[2].arg.kind() != ArgKind::IntList ||
      (node.inputs[3].arg.kind() != ArgKind::None &&
       (node.inputs[3].arg.kind() != ArgKind::Int ||
        node.inputs[3].arg.as_int().value != 0))) {
    return false;
  }
  const auto source = node.inputs[0].arg.as_tensor().id;
  if (keys.at(source) != roots.at(source)) {
    return false;
  }
  const auto& meta = graph.value(roots.at(source)).tensor_meta();
  int32_t previous = -1;
  for (const auto dim : meta.dim_order_hint) {
    if (meta.sizes[dim] > 1) {
      if (dim < previous) {
        return false;
      }
      previous = dim;
    }
  }
  const auto& sizes = node.inputs[1].arg.as_int_list().values;
  const auto& strides = node.inputs[2].arg.as_int_list().values;
  if (sizes != graph.value(node.output_value_ids()[0]).tensor_meta().sizes ||
      sizes.size() != strides.size()) {
    return false;
  }
  int64_t extent = 1;
  for (size_t reverse = sizes.size(); reverse > 0; --reverse) {
    const auto dim = reverse - 1;
    if (sizes[dim] <= 0 || extent > INT64_MAX / sizes[dim] ||
        (sizes[dim] > 1 && strides[dim] != extent)) {
      return false;
    }
    extent *= sizes[dim];
  }
  return extent == meta.numel();
}

ValueId latest_value(const std::vector<ValueId>& latest, ValueId id) {
  return valid(id) && valid(latest.at(id)) ? latest[id] : id;
}

void read_latest_values(Node& node, const std::vector<ValueId>& latest) {
  const auto remap = [&](auto list) {
    std::transform(
        list.ids.begin(), list.ids.end(), list.ids.begin(), [&](ValueId id) {
          return latest_value(latest, id);
        });
    return list;
  };
  for (auto& input : node.inputs) {
    const auto kind = input.arg.kind();
    if (kind == ArgKind::Tensor) {
      input.arg = TensorArg{latest_value(latest, input.arg.as_tensor().id)};
    } else if (kind == ArgKind::TensorList) {
      input.arg = remap(input.arg.as_tensor_list());
    } else if (kind == ArgKind::OptionalTensorList) {
      input.arg = remap(input.arg.as_optional_tensor_list());
    }
  }
}

// Def-use is left as deserialized until the pass finishes, so liveness is
// judged against the original readers of each alias.
void redirect_mutated_aliases(
    const Graph& graph,
    NodeId id,
    ValueId output,
    const std::vector<ValueId>& roots,
    const std::vector<ValueId>& keys,
    const std::vector<size_t>& position,
    std::vector<ValueId>& latest) {
  for (ValueId alias = 0; std::cmp_less(alias, graph.values.size()); ++alias) {
    const auto& value = graph.value(alias);
    if (roots.at(alias) != roots.at(output) || !valid(value.producer_id) ||
        position.at(value.producer_id) >= position.at(id)) {
      continue;
    }
    const auto& consumers = value.consumer_ids;
    const bool live = std::ranges::any_of(
                          consumers,
                          [&](const auto consumer) {
                            return position.at(consumer) > position.at(id);
                          }) ||
        std::ranges::find(graph.output_ids, alias) != graph.output_ids.end();
    if (!live) {
      continue;
    }
    if (!same_indexing(graph, keys, alias, output)) {
      fail("mutation through a live view is not supported");
    }
    latest.at(alias) = output;
  }
}

void functionalize_mutation(
    Graph& graph,
    NodeId id,
    const std::vector<ValueId>& roots,
    const std::vector<ValueId>& keys,
    const std::vector<size_t>& position,
    std::vector<ValueId>& latest) {
  auto& node = graph.node(id);
  const auto source = node.inputs[0].arg.as_tensor().id;
  const auto output = node.output_value_ids()[0];
  if (!is_in_place_activation(node.target) || !node.inputs[0].mutated ||
      std::any_of(
          node.inputs.begin() + 1,
          node.inputs.end(),
          [](const auto& input) { return input.mutated; }) ||
      graph.value(roots.at(source)).role != ValueRole::Intermediate ||
      !same_indexing(graph, keys, source, output)) {
    fail("unsupported Native mutation");
  }
  redirect_mutated_aliases(graph, id, output, roots, keys, position, latest);
  node.inputs[0].mutated = false;
  node.target.erase(node.target.find("_.default"), 1);
}

void materialize_view(
    Node& node,
    const Graph& graph,
    const std::vector<ValueId>& roots,
    const std::vector<ValueId>& keys) {
  if (node.target == "torch.ops.aten.as_strided.default") {
    if (!contiguous_reshape(node, graph, roots, keys)) {
      fail("Native as_strided requires a contiguous full-span reshape");
    }
    node.target = "torch.ops.aten.as_strided_copy.default";
    return;
  }
  const auto copy = std::ranges::find_if(
      kViewCopies, [&](const auto& view) { return node.target == view.first; });
  if (copy == kViewCopies.end()) {
    fail("unsupported Native view " + node.target);
  }
  node.target = std::string(copy->second);
}

} // namespace

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

void functionalize_aliases(Graph& graph) {
  std::vector<size_t> position(graph.nodes.size());
  for (size_t index = 0; index < graph.schedule.size(); ++index) {
    position.at(graph.schedule[index]) = index;
  }
  std::vector<ValueId> roots(graph.values.size());
  for (ValueId id = 0; std::cmp_less(id, graph.values.size()); ++id) {
    roots.at(id) = alias_root(graph, id);
  }
  const auto keys = indexing_keys(graph);
  std::vector<ValueId> latest(graph.values.size(), kInvalid);
  for (const auto id : graph.schedule) {
    auto& node = graph.node(id);
    read_latest_values(node, latest);
    if (!node.is_call()) {
      continue;
    }
    const bool mutating = std::ranges::any_of(
        node.inputs, [](const auto& input) { return input.mutated; });
    const auto outputs = node.output_value_ids();
    const bool aliases = std::ranges::any_of(outputs, [&](const auto output) {
      return valid(graph.value(output).alias_id);
    });
    if (!mutating && !aliases) {
      continue;
    }
    if (node.inputs.empty() || node.inputs[0].arg.kind() != ArgKind::Tensor ||
        outputs.size() != 1 || !graph.value(outputs[0]).is_tensor()) {
      fail("unsupported Native alias or mutation");
    }
    if (roots.at(node.inputs[0].arg.as_tensor().id) != roots.at(outputs[0])) {
      fail("Native alias does not match its operand");
    }
    if (mutating) {
      functionalize_mutation(graph, id, roots, keys, position, latest);
    } else {
      materialize_view(node, graph, roots, keys);
    }
  }
  std::transform(
      graph.output_ids.begin(),
      graph.output_ids.end(),
      graph.output_ids.begin(),
      [&](ValueId id) { return latest_value(latest, id); });
  for (auto& value : graph.values) {
    value.alias_id = kInvalid;
  }
  graph.rebuild_def_use();
}

} // namespace ptn
