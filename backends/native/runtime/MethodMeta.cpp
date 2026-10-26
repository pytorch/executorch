// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/MethodMeta.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <executorch/backends/native/runtime/graph/Ids.h>

namespace ptn {
namespace {

TensorInfo tensor_info(const Graph& graph, ValueId id, const char* position) {
  if (!in_bounds(id, graph.values.size())) {
    throw std::runtime_error(
        std::string("native metadata: invalid ") + position + " value id");
  }
  const Value& value = graph.values.at(static_cast<size_t>(id));
  if (!value.is_tensor()) {
    throw std::runtime_error(
        std::string("native metadata: ") + position + " '" + value.name +
        "' is not a tensor");
  }
  const TensorMeta& meta = value.tensor_meta();
  if (!meta.dim_order_hint.empty() &&
      meta.dim_order_hint.size() != meta.sizes.size()) {
    throw std::runtime_error(
        "native metadata: tensor dim order has the wrong rank");
  }
  std::vector<uint8_t> dim_order;
  dim_order.reserve(meta.sizes.size());
  for (size_t i = 0; i < meta.sizes.size(); ++i) {
    const int32_t dim_index = meta.dim_order_hint.empty()
        ? static_cast<int32_t>(i)
        : meta.dim_order_hint[i];
    if (dim_index < 0 || dim_index > std::numeric_limits<uint8_t>::max()) {
      throw std::runtime_error(
          "native metadata: tensor dim order is not representable");
    }
    dim_order.push_back(static_cast<uint8_t>(dim_index));
  }
  return TensorInfo(meta.dtype, meta.sizes, std::move(dim_order));
}

} // namespace

MethodMeta MethodMeta::from_method(const Method& method) {
  MethodMeta out;
  out.name_ = method.name;
  out.inputs_.reserve(method.graph.input_ids.size());
  for (const ValueId id : method.graph.input_ids) {
    out.inputs_.push_back(tensor_info(method.graph, id, "input"));
  }

  if (method.output_specs.size() != method.graph.output_ids.size()) {
    throw std::runtime_error(
        "native metadata: output specification count is inconsistent");
  }
  out.outputs_.reserve(method.graph.output_ids.size());
  for (size_t i = 0; i < method.graph.output_ids.size(); ++i) {
    if (method.output_specs.at(i).kind == OutputKind::UserOutput) {
      out.outputs_.push_back(
          tensor_info(method.graph, method.graph.output_ids.at(i), "output"));
    }
  }
  return out;
}

} // namespace ptn
