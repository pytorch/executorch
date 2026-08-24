// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>
#include <vector>

#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/Value.h>

namespace ptn {

// A pure function body: the index arena that owns the Nodes and Values the Ref
// handles point into, plus the ordered graph I/O and the subgraph arena for
// higher-order-op branch bodies. Mirrors the schema Graph; stateful
// method-level bindings (constants / output specs / mutable buffers) live on
// Method (deferred), not here.
//
// GraphRef indexes `subgraphs` of the *enclosing* Graph (per-graph arena),
// matching the schema recursion and the per-Graph SSA namespace. A subgraph is
// thus self-contained with its parent.
//
// Placeholder and Output nodes are real entries in `nodes` (schema OpKind), so
// def-use is uniform: a graph input value's producer is its placeholder node,
// not kInvalid; graph inputs are identified by membership in `input_refs`.
//
// Node storage and node *order* are decoupled. `nodes` is an append-only arena
// (a new node lands at the end, out of dataflow position) so NodeRefs stay
// stable — the index-arena invariant. `schedule` gives the topological /
// execution order over it: the order a runtime walks the nodes. At load the
// arena order equals the wire's topological order and `schedule` is the
// identity [0, 1, ..., n-1]; across mutation the arena order is no longer
// topological, so `schedule` is authoritative — reorder / insert there (moving
// NodeRefs, invalidating no ref) rather than moving storage. Deletion is still
// deferred; it will need a tombstone plus a compacting pass, since dropping a
// node outright would shift every later NodeRef.
struct Graph {
  std::vector<Node> nodes; // node arena, incl. placeholder / output nodes
  std::vector<NodeRef> schedule; // execution / topological order over `nodes`
  std::vector<Value> values; // SSA-value arena; ValueRef indexes this
  std::vector<ValueRef> input_refs; // graph input values, in order
  std::vector<ValueRef> output_refs; // graph output values, in order
  std::vector<Graph> subgraphs; // HOP branch/body bodies; GraphRef indexes this

  // Bounds-checked ref resolution; each throws std::runtime_error on an invalid
  // (out-of-range or kInvalid) ref.
  Node& node(NodeRef ref);
  const Node& node(NodeRef ref) const;
  Value& value(ValueRef ref);
  const Value& value(ValueRef ref) const;
  Graph& subgraph(GraphRef ref);
  const Graph& subgraph(GraphRef ref) const;

  // Reset `schedule` to the identity order [0, 1, ..., nodes.size() - 1] (arena
  // order). The deserializer calls this after appending the nodes in wire
  // order.
  void reset_schedule();

  // Recompute every Value's producer / consumers from the nodes (clears the
  // existing wiring first). Each node is the producer of its output values and
  // a consumer of its input_value_refs(). Order-independent (walks the arena),
  // so it does not depend on `schedule`. Does NOT recurse into subgraphs — each
  // has its own SSA namespace, so call it per graph.
  //
  // consumer_refs is a set of consuming nodes, not a bag of uses: a node that
  // reads a value twice (`add(x, x)`) is listed once. Throws
  // std::runtime_error on a ref that is set but does not address the value
  // arena, which can only mean a corrupt graph; kInvalid is left alone, since
  // an absent operand is legal.
  void rebuild_def_use();

  // Multi-line debug dump: inputs, one line per node in `schedule` order (arena
  // order if `schedule` is empty), outputs, subgraph count.
  std::string to_string() const;
};

} // namespace ptn
