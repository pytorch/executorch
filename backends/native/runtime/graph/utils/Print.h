// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Graph.h>
#include <executorch/backends/native/runtime/graph/Node.h>
#include <executorch/backends/native/runtime/graph/Scalar.h>
#include <executorch/backends/native/runtime/graph/TensorMeta.h>

namespace ptn {

// Debug renderings of the in-memory IR. Free functions in their own target so
// nothing on an execution path links the formatting code; members would tie
// <string> and these format choices to every consumer of the IR headers. The
// output is for humans -- nothing parses it back, and it is not versioned.

// e.g. "Float[16,16]", "Float[1..8,16]" (bounded dynamic), "Float[0..?,16]"
// (unbounded).
std::string to_string(const TensorMeta& meta);

// The live alternative only: "true", "-3", "1.5e-08".
std::string to_string(const Scalar& scalar);

// Compact one-line form. A symbolic operand (one carrying a valid id) renders
// as "%<id>"; a literal renders as its value.
std::string to_string(const Argument& arg);

// Single line, e.g. "a = aten.add.Tensor(x, y, alpha=1) -> %3".
std::string to_string(const Node& node);

// Multi-line: inputs, one line per node in `schedule` order (declaration order
// if `schedule` is empty), outputs, subgraph count.
std::string to_string(const Graph& graph);

} // namespace ptn
