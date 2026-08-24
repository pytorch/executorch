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
size_t checked_index(const Vec& vec, int32_t ref, const char* what) {
  if (!in_bounds(ref, vec.size())) {
    throw std::runtime_error(std::string(what) + ": invalid ref");
  }
  return static_cast<size_t>(ref);
}

// True when `ref` names a value in the arena. False for kInvalid, which is a
// legitimately absent operand. Throws when `ref` is set but out of range: that
// can only be a corrupt graph, and skipping it would leave def-use half-wired
// with no signal to the caller.
bool resolves(ValueRef ref, size_t size, const char* what) {
  if (!valid(ref)) {
    return false;
  }
  if (!in_bounds(ref, size)) {
    throw std::runtime_error(
        std::string("Graph::rebuild_def_use: ") + what + " ref " +
        std::to_string(ref) + " does not address the value arena");
  }
  return true;
}

// Format a ref list as "[%0, %1]".
std::string join_refs(const std::vector<ValueRef>& refs) {
  std::string s = "[";
  for (size_t i = 0; i < refs.size(); ++i) {
    if (i) {
      s += ", ";
    }
    s += "%" + std::to_string(refs[i]);
  }
  return s + "]";
}

} // namespace

Node& Graph::node(NodeRef ref) {
  return nodes[checked_index(nodes, ref, "Graph::node")];
}
const Node& Graph::node(NodeRef ref) const {
  return nodes[checked_index(nodes, ref, "Graph::node")];
}

Value& Graph::value(ValueRef ref) {
  return values[checked_index(values, ref, "Graph::value")];
}
const Value& Graph::value(ValueRef ref) const {
  return values[checked_index(values, ref, "Graph::value")];
}

Graph& Graph::subgraph(GraphRef ref) {
  return subgraphs[checked_index(subgraphs, ref, "Graph::subgraph")];
}
const Graph& Graph::subgraph(GraphRef ref) const {
  return subgraphs[checked_index(subgraphs, ref, "Graph::subgraph")];
}

void Graph::reset_schedule() {
  schedule.resize(nodes.size());
  std::iota(schedule.begin(), schedule.end(), NodeRef{0});
}

void Graph::rebuild_def_use() {
  for (Value& v : values) {
    v.producer_ref = kInvalid;
    v.consumer_refs.clear();
  }
  for (size_t i = 0; i < nodes.size(); ++i) {
    const NodeRef ni = static_cast<NodeRef>(i);
    const Node& n = nodes[i];
    for (const Output& out : n.outputs) {
      if (out.kind == OutputValueKind::TensorList) {
        for (ValueRef r : out.elem_refs) {
          if (resolves(r, values.size(), "output element")) {
            values[r].producer_ref = ni;
          }
        }
      } else if (resolves(out.value_ref, values.size(), "output")) {
        values[out.value_ref].producer_ref = ni;
      }
    }
    for (ValueRef r : n.input_value_refs()) {
      if (!resolves(r, values.size(), "input")) {
        continue;
      }
      // Nodes are walked in arena order, so a repeated operand appends `ni`
      // consecutively; checking the tail is enough to keep this a set.
      std::vector<NodeRef>& consumers = values[r].consumer_refs;
      if (consumers.empty() || consumers.back() != ni) {
        consumers.push_back(ni);
      }
    }
  }
}

std::string Graph::to_string() const {
  std::string s = "inputs: " + join_refs(input_refs) + "\n";
  if (!schedule.empty()) {
    for (NodeRef ref : schedule) {
      s += "  " + node(ref).to_string() + "\n";
    }
  } else {
    for (const Node& n : nodes) {
      s += "  " + n.to_string() + "\n";
    }
  }
  s += "outputs: " + join_refs(output_refs) + "\n";
  if (!subgraphs.empty()) {
    s += "(" + std::to_string(subgraphs.size()) + " subgraphs)\n";
  }
  return s;
}

} // namespace ptn
