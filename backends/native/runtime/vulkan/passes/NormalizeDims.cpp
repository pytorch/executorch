// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/passes/NormalizeDims.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <executorch/backends/native/runtime/graph/Argument.h>
#include <executorch/backends/native/runtime/graph/Ids.h>
#include <executorch/backends/native/runtime/graph/Node.h>

namespace ptn::vulkan {
namespace {

constexpr std::array<std::string_view, 7>
    kDimArgs{"dim", "dims", "dim0", "dim1", "dim2", "start_dim", "end_dim"};

std::runtime_error
dim_error(const Node& node, std::string_view arg, const std::string& problem) {
  return std::runtime_error(
      "vulkan: node '" + node.name + "' (" + node.target +
      "): " + std::string(arg) + " " + problem);
}

std::optional<int64_t> reference_rank(const Graph& graph, const Node& node) {
  for (const NamedArgument& input : node.inputs) {
    ValueId id = kInvalid;
    if (input.arg.kind() == ArgKind::Tensor) {
      id = input.arg.as_tensor().id;
    } else if (
        input.arg.kind() == ArgKind::TensorList &&
        !input.arg.as_tensor_list().ids.empty()) {
      id = input.arg.as_tensor_list().ids.front();
    }
    if (valid(id) && graph.value(id).is_tensor()) {
      return static_cast<int64_t>(graph.value(id).tensor_meta().ndim());
    }
  }
  return std::nullopt;
}

} // namespace

size_t make_dims_non_negative(Graph& graph) {
  size_t count = 0;
  for (const NodeId node_id : graph.schedule) {
    Node& node = graph.node(node_id);
    if (!node.is_call() || !node.target.starts_with("torch.ops.aten.")) {
      continue;
    }
    std::optional<int64_t> rank;
    const auto non_negative = [&](std::string_view arg, int64_t dim) {
      if (!rank) {
        rank = reference_rank(graph, node);
        if (!rank) {
          throw dim_error(node, arg, "has no tensor input to index");
        }
        *rank += node.target.starts_with("torch.ops.aten.unsqueeze") ? 1 : 0;
        // ATen wraps a scalar's dims as a rank-one tensor's: 0 or -1.
        *rank = std::max<int64_t>(*rank, 1);
      }
      if (dim < -*rank || dim >= *rank) {
        throw dim_error(
            node,
            arg,
            std::to_string(dim) + " is out of range for rank " +
                std::to_string(*rank));
      }
      count += dim < 0 ? 1 : 0;
      return dim < 0 ? dim + *rank : dim;
    };
    for (NamedArgument& input : node.inputs) {
      if (std::ranges::find(kDimArgs, input.name) == kDimArgs.end()) {
        continue;
      }
      if (input.arg.kind() == ArgKind::Int) {
        if (valid(input.arg.as_int().id)) {
          throw dim_error(node, input.name, "is symbolic");
        }
        input.arg =
            IntArg{.value = non_negative(input.name, input.arg.as_int().value)};
      } else if (input.arg.kind() == ArgKind::IntList) {
        IntListArg dims = input.arg.as_int_list();
        if (!dims.ids.empty()) {
          throw dim_error(node, input.name, "is symbolic");
        }
        for (int64_t& dim : dims.values) {
          dim = non_negative(input.name, dim);
        }
        input.arg = std::move(dims);
      }
    }
  }
  return count;
}

} // namespace ptn::vulkan
