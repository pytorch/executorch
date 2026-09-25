// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>

#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/Value.h>

namespace ptn {

// A pure function body: the index lists that own the Nodes and Values the Id
// handles point into, plus the ordered graph I/O and the subgraph storage for
// higher-order-op branch bodies. Mirrors the schema Graph; stateful
// method-level bindings (constants / output specs / mutable buffers) live on
// Method, not here.
//
// Nodes and Values are separate id spaces: a node produces none (Output), one,
// or several values (topk, split), so the two are not 1:1, though a
// single-output node does share its SSA name with the value it produces. The
// dataflow DAG lives in the Values, each carrying its producer node and its
// consumers; `schedule` is not that graph, only the linear order a runtime
// walks the nodes in.
//
// GraphId indexes `subgraphs` of the *enclosing* Graph, matching the schema
// recursion and the per-Graph SSA namespace, so a subgraph is self-contained
// with its parent.
//
// Placeholder and Output nodes are real entries in `nodes` (schema OpKind), so
// def-use is uniform: a graph input value's producer is its placeholder node,
// not kInvalid; graph inputs are identified by membership in `input_ids`.
//
// Node storage and node *order* are decoupled. `nodes` is append-only — a new
// node lands at the end, out of dataflow position — so NodeIds stay stable.
// `schedule` carries the execution order instead: reorder or insert there,
// which moves no storage and invalidates no id. Nodes are never erased, since
// dropping one would shift every later NodeId.
struct Graph {
  std::vector<Node> nodes; // node list, incl. placeholder / output nodes
  std::vector<NodeId> schedule; // execution / topological order over `nodes`
  std::vector<Value> values; // SSA-value list; ValueId indexes this
  std::vector<ValueId> input_ids; // graph input values, in order
  std::vector<ValueId> output_ids; // graph output values, in order
  std::vector<Graph> subgraphs; // HOP branch bodies; GraphId indexes this

  // Bounds-checked id resolution; each throws std::runtime_error on an invalid
  // (out-of-range or kInvalid) id.
  Node& node(NodeId id);
  const Node& node(NodeId id) const;
  Value& value(ValueId id);
  const Value& value(ValueId id) const;
  Graph& subgraph(GraphId id);
  const Graph& subgraph(GraphId id) const;

  // Set `schedule` to the identity order, which is the execution order exactly
  // when the nodes are already in dataflow position — as they are straight off
  // the wire, before any mutation. A graph is not executable until this runs:
  // `schedule` starts empty, and an engine's work list is seeded from it.
  //
  // Throws std::runtime_error if `schedule` is already set. Re-running would
  // overwrite an order that a mutation made authoritative, so this is a
  // load-time step, not a reset.
  void initialize_schedule();

  // Recompute every Value's producer / consumers from the nodes, clearing the
  // existing wiring first: each node produces its output values and consumes
  // its input_value_ids(). Independent of `schedule` — it walks `nodes`
  // directly. Does NOT recurse into subgraphs, which have their own SSA
  // namespaces; call it per graph.
  //
  // Throws std::runtime_error on an id that is set but does not address
  // `values`, which can only mean a corrupt graph. kInvalid is left alone: an
  // absent operand is legal.
  void rebuild_def_use();
};

} // namespace ptn
