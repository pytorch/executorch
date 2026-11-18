// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>

#include <executorch/backends/native/runtime/graph/Graph.h>

namespace ptn {

// The input named `name`, else the unnamed input at `position`. A named input
// at `position` belongs to another parameter, so it is never returned.
const Argument*
find_arg(const Node& node, std::string_view name, size_t position);

// The value of a tensor argument; nullopt for a null, non-tensor or None one.
std::optional<ValueId> tensor_id(const Argument* arg);

// An absent optional argument takes its schema default, `value`.
bool is_bool_or_absent(const Argument* arg, bool value);
// A float literal's value, or `value` if absent; nullopt for anything else.
std::optional<double> float_or_absent(const Argument* arg, double value);
// Whether `node` has at most `params.size()` inputs, each unnamed or named in
// `params`. find_arg reads an input of any other name as absent.
bool inputs_fit_schema(
    const Node& node,
    std::span<const std::string_view> params);

// Orders active nodes by their data dependencies. Ties retain schedule order.
void stable_topological_sort(Graph& graph);

// Checks ids, def-use, aliases, active-node uniqueness, and topological order.
// Recurses into subgraphs. Throws std::runtime_error on the first violation.
void validate_graph(const Graph& graph);

} // namespace ptn
