// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/Graph.h>

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
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
        std::to_string(id) + " does not address the value arena");
  }
  return true;
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

void Graph::initialize_schedule() {
  schedule.resize(nodes.size());
  std::iota(schedule.begin(), schedule.end(), NodeId{0});
}

void Graph::rebuild_def_use() {
  for (Value& v : values) {
    v.producer_id = kInvalid;
    v.consumer_ids.clear();
  }
  for (size_t i = 0; i < nodes.size(); ++i) {
    const NodeId ni = static_cast<NodeId>(i);
    const Node& n = nodes[i];
    for (const Output& out : n.outputs) {
      if (out.kind == OutputValueKind::TensorList) {
        for (ValueId e : out.elem_ids) {
          if (resolves(e, values.size(), "output element")) {
            values[e].producer_id = ni;
          }
        }
      } else if (resolves(out.value_id, values.size(), "output")) {
        values[out.value_id].producer_id = ni;
      }
    }
    for (ValueId in : n.input_value_ids()) {
      if (!resolves(in, values.size(), "input")) {
        continue;
      }
      // Nodes are walked in arena order, so a repeated operand appends `ni`
      // consecutively; checking the tail is enough to keep this a set.
      std::vector<NodeId>& consumers = values[in].consumer_ids;
      if (consumers.empty() || consumers.back() != ni) {
        consumers.push_back(ni);
      }
    }
  }
}

} // namespace ptn
