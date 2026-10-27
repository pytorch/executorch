// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/Graph.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ptn {

namespace {

template <typename Vec>
size_t checked_index(const Vec& vec, int32_t id, const char* what) {
  if (!in_bounds(id, vec.size())) {
    throw std::runtime_error(std::string(what) + ": invalid id");
  }
  return static_cast<size_t>(id);
}

// False for kInvalid — a legitimately absent operand. Throws when `id` is set
// but out of range: skipping it would leave def-use half-wired with no signal.
bool resolves(ValueId id, size_t size, const char* what) {
  if (!valid(id)) {
    return false;
  }
  if (!in_bounds(id, size)) {
    throw std::runtime_error(
        std::string("Graph::rebuild_def_use: ") + what + " id " +
        std::to_string(id) + " does not address the value list");
  }
  return true;
}

template <typename Id>
Id next_id(size_t size, const char* what) {
  if (size > static_cast<size_t>(std::numeric_limits<Id>::max())) {
    throw std::runtime_error(std::string(what) + ": id space exhausted");
  }
  return static_cast<Id>(size);
}

size_t replace_argument(Argument& argument, ValueId from, ValueId to) {
  switch (argument.kind()) {
    case ArgKind::Tensor:
      if (argument.as_tensor().id == from) {
        argument = TensorArg{to};
        return 1;
      }
      return 0;
    case ArgKind::Int: {
      const IntArg old = argument.as_int();
      if (old.id == from) {
        argument = IntArg{old.value, to};
        return 1;
      }
      return 0;
    }
    case ArgKind::Float: {
      const FloatArg old = argument.as_float();
      if (old.id == from) {
        argument = FloatArg{old.value, to};
        return 1;
      }
      return 0;
    }
    case ArgKind::Bool: {
      const BoolArg old = argument.as_bool();
      if (old.id == from) {
        argument = BoolArg{old.value, to};
        return 1;
      }
      return 0;
    }
    case ArgKind::IntList: {
      IntListArg replacement = argument.as_int_list();
      size_t count = 0;
      for (ValueId& id : replacement.ids) {
        if (id == from) {
          id = to;
          ++count;
        }
      }
      if (count != 0) {
        argument = std::move(replacement);
      }
      return count;
    }
    case ArgKind::TensorList: {
      TensorListArg replacement = argument.as_tensor_list();
      size_t count = 0;
      for (ValueId& id : replacement.ids) {
        if (id == from) {
          id = to;
          ++count;
        }
      }
      if (count != 0) {
        argument = std::move(replacement);
      }
      return count;
    }
    case ArgKind::OptionalTensorList: {
      OptionalTensorListArg replacement = argument.as_optional_tensor_list();
      size_t count = 0;
      for (ValueId& id : replacement.ids) {
        if (id == from) {
          id = to;
          ++count;
        }
      }
      if (count != 0) {
        argument = std::move(replacement);
      }
      return count;
    }
    case ArgKind::None:
    case ArgKind::String:
    case ArgKind::ScalarType:
    case ArgKind::FloatList:
    case ArgKind::BoolList:
    case ArgKind::Graph:
      return 0;
  }
  throw std::runtime_error("Graph::replace_input: unknown argument kind");
}

size_t schedule_position(const Graph& graph, NodeId node_id) {
  const auto it =
      std::find(graph.schedule.begin(), graph.schedule.end(), node_id);
  if (it == graph.schedule.end()) {
    throw std::runtime_error("graph mutation: node is not scheduled");
  }
  return static_cast<size_t>(it - graph.schedule.begin());
}

NodeId insert_node(Graph& graph, size_t position, Node node) {
  const NodeId id = next_id<NodeId>(graph.nodes.size(), "Graph::insert_node");
  graph.nodes.push_back(std::move(node));
  graph.schedule.insert(graph.schedule.begin() + position, id);
  graph.rebuild_def_use();
  return id;
}

} // namespace

Node& Graph::node(NodeId id) {
  return nodes[checked_index(nodes, id, "Graph::node")];
}
const Node& Graph::node(NodeId id) const {
  return nodes[checked_index(nodes, id, "Graph::node")];
}

Value& Graph::value(ValueId id) {
  return values[checked_index(values, id, "Graph::value")];
}
const Value& Graph::value(ValueId id) const {
  return values[checked_index(values, id, "Graph::value")];
}

Graph& Graph::subgraph(GraphId id) {
  return subgraphs[checked_index(subgraphs, id, "Graph::subgraph")];
}
const Graph& Graph::subgraph(GraphId id) const {
  return subgraphs[checked_index(subgraphs, id, "Graph::subgraph")];
}

ValueId Graph::append_value(Value value) {
  const ValueId id = next_id<ValueId>(values.size(), "Graph::append_value");
  values.push_back(std::move(value));
  return id;
}

NodeId Graph::insert_node_before(NodeId before, Node new_node) {
  return insert_node(
      *this, schedule_position(*this, before), std::move(new_node));
}

NodeId Graph::insert_node_after(NodeId after, Node new_node) {
  return insert_node(
      *this, schedule_position(*this, after) + 1, std::move(new_node));
}

size_t Graph::replace_input(NodeId node_id, ValueId from, ValueId to) {
  schedule_position(*this, node_id);
  value(from);
  value(to);
  size_t count = 0;
  for (NamedArgument& input : node(node_id).inputs) {
    count += replace_argument(input.arg, from, to);
  }
  rebuild_def_use();
  return count;
}

size_t Graph::replace_all_uses(ValueId from, ValueId to, NodeId excluded_node) {
  value(from);
  value(to);
  if (std::find(input_ids.begin(), input_ids.end(), from) != input_ids.end()) {
    throw std::runtime_error(
        "Graph::replace_all_uses: cannot replace a graph input");
  }
  size_t count = 0;
  for (NodeId node_id : schedule) {
    if (node_id == excluded_node) {
      continue;
    }
    for (NamedArgument& input : node(node_id).inputs) {
      count += replace_argument(input.arg, from, to);
    }
  }
  for (ValueId id = 0; id < static_cast<ValueId>(values.size()); ++id) {
    if (id != to && value(id).alias_id == from) {
      value(id).alias_id = to;
      ++count;
    }
  }
  std::ranges::replace(output_ids, from, to);
  rebuild_def_use();
  return count;
}

void Graph::erase_node(NodeId node_id) {
  Node& erased_node = node(node_id);
  const size_t position = schedule_position(*this, node_id);
  if (erased_node.is_placeholder() || erased_node.is_output()) {
    throw std::runtime_error(
        "Graph::erase_node: cannot erase a graph boundary node");
  }
  rebuild_def_use();
  for (ValueId output_id : erased_node.output_value_ids()) {
    if (!value(output_id).consumer_ids.empty() ||
        std::find(output_ids.begin(), output_ids.end(), output_id) !=
            output_ids.end()) {
      throw std::runtime_error("Graph::erase_node: node has a live output");
    }
  }
  schedule.erase(schedule.begin() + position);
  rebuild_def_use();
}

void Graph::initialize_schedule() {
  if (!schedule.empty()) {
    throw std::runtime_error(
        "Graph::initialize_schedule: schedule is already set; identity order "
        "is the execution order only before any mutation");
  }
  schedule.resize(nodes.size());
  std::iota(schedule.begin(), schedule.end(), NodeId{0});
}

void Graph::rebuild_def_use() {
  for (Value& v : values) {
    v.producer_id = kInvalid;
    v.consumer_ids.clear();
  }
  for (NodeId ni : schedule) {
    const Node& n = node(ni);
    for (ValueId output_id : n.output_value_ids()) {
      if (resolves(output_id, values.size(), "output")) {
        values[output_id].producer_id = ni;
      }
    }
    for (ValueId in : n.input_value_ids()) {
      if (!resolves(in, values.size(), "input")) {
        continue;
      }
      // Nodes are walked in schedule order, so a repeated operand appends `ni`
      // consecutively; checking the tail is enough to keep this a set.
      std::vector<NodeId>& consumers = values[in].consumer_ids;
      if (consumers.empty() || consumers.back() != ni) {
        consumers.push_back(ni);
      }
    }
  }
}

} // namespace ptn
