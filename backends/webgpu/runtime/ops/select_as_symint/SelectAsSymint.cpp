/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>

#include <algorithm>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

// et_vk.select_as_symint: out SymInt = x[index] along dim; read at execute.
void select_as_symint_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int x_id = args.at(0);
  const int dim_id = args.at(1);
  const int index_id = args.at(2);
  const int out_id = args.at(3);

  if (graph.get_value_type(out_id) != WebGPUGraph::ValueType::SymInt) {
    throw std::runtime_error("select_as_symint: output is not a SymInt");
  }
  const std::vector<int>& inputs = graph.input_ids();
  if (std::find(inputs.begin(), inputs.end(), x_id) == inputs.end()) {
    throw std::runtime_error(
        "select_as_symint: source tensor is not a graph input");
  }
  graph.add_symint_source(
      out_id,
      x_id,
      static_cast<int>(graph.get_int(dim_id)),
      static_cast<int>(graph.get_int(index_id)));
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.select_as_symint.default, select_as_symint_impl);
}

} // namespace executorch::backends::webgpu
