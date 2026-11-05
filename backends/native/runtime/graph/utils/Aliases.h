// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <executorch/backends/native/runtime/graph/Graph.h>

namespace ptn {

// The value that owns value_id's storage, or kInvalid on an alias cycle.
ValueId alias_root(const Graph& graph, ValueId value_id);

// Whether an active node mutates any value in value_id's alias group.
bool has_mutated_alias_use(const Graph& graph, ValueId value_id);

// For engines that execute only functional, unaliased values. Rewrites
// view/permute/transpose aliases and contiguous full-span as_strided reshapes
// to copies, and relu_/gelu_/sigmoid_ of intermediates to their functional
// forms. A mutation is functionalized only when every live alias of its operand
// reads the result with identical logical indexing. Throws std::runtime_error
// on any other alias or mutation.
void functionalize_aliases(Graph& graph);

} // namespace ptn
