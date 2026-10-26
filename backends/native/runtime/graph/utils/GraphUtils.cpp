// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/utils/GraphUtils.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ptn {
namespace {

void validate_id(ValueId id, size_t size, const char* what) {
  if (!in_bounds(id, size)) {
    throw std::runtime_error(std::string("validate_graph: invalid ") + what);
  }
}

void validate_aliases(const Graph& graph) {
  std::vector<uint8_t> state(graph.values.size(), 0);
  for (size_t start = 0; start < graph.values.size(); ++start) {
    ValueId current = static_cast<ValueId>(start);
    std::vector<ValueId> path;
    while (valid(current)) {
      validate_id(current, graph.values.size(), "alias id");
      if (state[static_cast<size_t>(current)] != 0) {
        break;
      }
      state[static_cast<size_t>(current)] = 1;
      path.push_back(current);
      current = graph.value(current).alias_id;
    }
    if (valid(current)) {
      validate_id(current, graph.values.size(), "alias id");
      if (state[static_cast<size_t>(current)] == 1) {
        throw std::runtime_error("validate_graph: alias cycle");
      }
    }
    for (ValueId id : path) {
      state[static_cast<size_t>(id)] = 2;
    }
  }
}

void validate_subgraph_arguments(const Graph& graph, const Node& node) {
  if (std::ranges::any_of(node.inputs, [&graph](const NamedArgument& input) {
        return input.arg.kind() == ArgKind::Graph &&
            !in_bounds(
                   input.arg.as_graph().subgraph_id, graph.subgraphs.size());
      })) {
    throw std::runtime_error("validate_graph: invalid subgraph id");
  }
}

void validate_one_graph(const Graph& graph) {
  std::vector<int64_t> positions(graph.nodes.size(), -1);
  for (size_t position = 0; position < graph.schedule.size(); ++position) {
    const NodeId id = graph.schedule[position];
    if (!in_bounds(id, graph.nodes.size())) {
      throw std::runtime_error("validate_graph: invalid scheduled node id");
    }
    if (positions[static_cast<size_t>(id)] >= 0) {
      throw std::runtime_error("validate_graph: duplicate scheduled node id");
    }
    positions[static_cast<size_t>(id)] = static_cast<int64_t>(position);
  }

  std::vector<NodeId> producers(graph.values.size(), kInvalid);
  std::vector<std::vector<NodeId>> consumers(graph.values.size());
  for (NodeId node_id : graph.schedule) {
    validate_subgraph_arguments(graph, graph.node(node_id));
    for (ValueId output_id : graph.node(node_id).output_value_ids()) {
      validate_id(output_id, graph.values.size(), "node output id");
      NodeId& producer = producers[static_cast<size_t>(output_id)];
      if (valid(producer)) {
        throw std::runtime_error(
            "validate_graph: value has multiple producers");
      }
      producer = node_id;
    }
  }

  for (NodeId node_id : graph.schedule) {
    std::vector<uint8_t> seen(graph.values.size(), 0);
    for (ValueId input_id : graph.node(node_id).input_value_ids()) {
      validate_id(input_id, graph.values.size(), "node input id");
      const NodeId producer = producers[static_cast<size_t>(input_id)];
      if (!valid(producer)) {
        throw std::runtime_error(
            "validate_graph: input has no active producer");
      }
      if (positions[static_cast<size_t>(producer)] >=
          positions[static_cast<size_t>(node_id)]) {
        throw std::runtime_error("validate_graph: schedule is not topological");
      }
      if (seen[static_cast<size_t>(input_id)] == 0) {
        consumers[static_cast<size_t>(input_id)].push_back(node_id);
        seen[static_cast<size_t>(input_id)] = 1;
      }
    }
  }

  std::vector<uint8_t> is_graph_input(graph.values.size(), 0);
  for (ValueId id : graph.input_ids) {
    validate_id(id, graph.values.size(), "graph input id");
    if (is_graph_input[static_cast<size_t>(id)] != 0) {
      throw std::runtime_error("validate_graph: duplicate graph input id");
    }
    is_graph_input[static_cast<size_t>(id)] = 1;
    const NodeId producer = producers[static_cast<size_t>(id)];
    if (!valid(producer) || !graph.node(producer).is_placeholder()) {
      throw std::runtime_error(
          "validate_graph: graph input is not produced by a placeholder");
    }
  }

  std::vector<ValueId> scheduled_inputs;
  std::vector<ValueId> returned_outputs;
  for (NodeId node_id : graph.schedule) {
    const Node& node = graph.node(node_id);
    if (node.is_placeholder()) {
      const std::vector<ValueId> ids = node.output_value_ids();
      std::ranges::copy_if(
          ids,
          std::back_inserter(scheduled_inputs),
          [&is_graph_input](ValueId id) {
            return is_graph_input[static_cast<size_t>(id)] != 0;
          });
    } else if (node.is_output()) {
      const std::vector<ValueId> ids = node.input_value_ids();
      returned_outputs.insert(returned_outputs.end(), ids.begin(), ids.end());
    }
  }
  if (scheduled_inputs != graph.input_ids) {
    throw std::runtime_error(
        "validate_graph: graph inputs disagree with placeholder order");
  }
  if (returned_outputs != graph.output_ids) {
    throw std::runtime_error(
        "validate_graph: graph outputs disagree with output-node order");
  }
  for (ValueId id : graph.output_ids) {
    validate_id(id, graph.values.size(), "graph output id");
    if (!valid(producers[static_cast<size_t>(id)])) {
      throw std::runtime_error("validate_graph: graph output has no producer");
    }
  }
  for (size_t i = 0; i < graph.values.size(); ++i) {
    if (graph.values[i].producer_id != producers[i] ||
        graph.values[i].consumer_ids != consumers[i]) {
      throw std::runtime_error("validate_graph: stale def-use information");
    }
  }
  validate_aliases(graph);
  if (std::ranges::any_of(graph.values, [&graph](const Value& value) {
        return valid(value.alias_id) &&
            (!value.is_tensor() || !graph.value(value.alias_id).is_tensor());
      })) {
    throw std::runtime_error(
        "validate_graph: alias must connect tensor values");
  }
}

} // namespace

// cppcheck-suppress unusedFunction
void stable_topological_sort(Graph& graph) {
  std::vector<int64_t> positions(graph.nodes.size(), -1);
  for (size_t i = 0; i < graph.schedule.size(); ++i) {
    const NodeId id = graph.schedule[i];
    if (!in_bounds(id, graph.nodes.size())) {
      throw std::runtime_error(
          "stable_topological_sort: invalid scheduled node id");
    }
    if (positions[static_cast<size_t>(id)] >= 0) {
      throw std::runtime_error(
          "stable_topological_sort: duplicate scheduled node id");
    }
    positions[static_cast<size_t>(id)] = static_cast<int64_t>(i);
  }

  std::vector<NodeId> producers(graph.values.size(), kInvalid);
  for (NodeId node_id : graph.schedule) {
    for (ValueId output_id : graph.node(node_id).output_value_ids()) {
      validate_id(output_id, graph.values.size(), "node output id");
      NodeId& producer = producers[static_cast<size_t>(output_id)];
      if (valid(producer)) {
        throw std::runtime_error(
            "stable_topological_sort: value has multiple producers");
      }
      producer = node_id;
    }
  }

  std::vector<size_t> indegrees(graph.nodes.size(), 0);
  std::vector<std::vector<NodeId>> successors(graph.nodes.size());
  for (NodeId node_id : graph.schedule) {
    std::vector<uint8_t> seen(graph.nodes.size(), 0);
    for (ValueId input_id : graph.node(node_id).input_value_ids()) {
      validate_id(input_id, graph.values.size(), "node input id");
      const NodeId producer = producers[static_cast<size_t>(input_id)];
      if (!valid(producer)) {
        throw std::runtime_error(
            "stable_topological_sort: input has no active producer");
      }
      if (seen[static_cast<size_t>(producer)] == 0) {
        successors[static_cast<size_t>(producer)].push_back(node_id);
        ++indegrees[static_cast<size_t>(node_id)];
        seen[static_cast<size_t>(producer)] = 1;
      }
    }
  }

  using Ready = std::pair<size_t, NodeId>;
  std::priority_queue<Ready, std::vector<Ready>, std::greater<Ready>> ready;
  for (NodeId node_id : graph.schedule) {
    if (indegrees[static_cast<size_t>(node_id)] == 0) {
      ready.emplace(
          static_cast<size_t>(positions[static_cast<size_t>(node_id)]),
          node_id);
    }
  }

  std::vector<NodeId> sorted;
  sorted.reserve(graph.schedule.size());
  while (!ready.empty()) {
    const NodeId node_id = ready.top().second;
    ready.pop();
    sorted.push_back(node_id);
    for (NodeId successor : successors[static_cast<size_t>(node_id)]) {
      size_t& indegree = indegrees[static_cast<size_t>(successor)];
      --indegree;
      if (indegree == 0) {
        ready.emplace(
            static_cast<size_t>(positions[static_cast<size_t>(successor)]),
            successor);
      }
    }
  }
  if (sorted.size() != graph.schedule.size()) {
    throw std::runtime_error("stable_topological_sort: graph contains a cycle");
  }
  graph.schedule = std::move(sorted);
  graph.rebuild_def_use();
}

// cppcheck-suppress unusedFunction
void validate_graph(const Graph& graph) {
  validate_one_graph(graph);
  for (const Graph& subgraph : graph.subgraphs) {
    validate_graph(subgraph);
  }
}

} // namespace ptn
