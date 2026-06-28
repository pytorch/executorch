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
#include <functional>
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

// An operand is a live SymInt or a static Int constant.
int32_t read_scalar(WebGPUGraph& graph, int id) {
  if (graph.get_value_type(id) == WebGPUGraph::ValueType::SymInt) {
    return graph.read_symint(id);
  }
  return static_cast<int32_t>(graph.get_int(id));
}

// SymInt arithmetic; mirrors Vulkan SymIntOps.cpp, recomputed on resize.
void register_sym_binary(
    WebGPUGraph& graph,
    const std::vector<int>& args,
    std::function<int32_t(int32_t, int32_t)> op) {
  if (args.size() < 3) {
    throw std::runtime_error("sym binary op: expected [a, b, out] args");
  }
  const int a = args.at(0);
  const int b = args.at(1);
  const int out = args.at(2);
  if (graph.get_value_type(out) != WebGPUGraph::ValueType::SymInt) {
    return; // folded to a static Int -> nothing live to compute
  }
  auto recompute = [a, b, out, op](WebGPUGraph& g) {
    g.set_symint(out, op(read_scalar(g, a), read_scalar(g, b)));
  };
  recompute(graph); // seed the build-time value
  if (graph.get_value_type(a) == WebGPUGraph::ValueType::SymInt) {
    graph.add_resize_hook(a, recompute);
  }
  if (graph.get_value_type(b) == WebGPUGraph::ValueType::SymInt) {
    graph.add_resize_hook(b, recompute);
  }
}

void sym_add_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  register_sym_binary(graph, args, [](int32_t x, int32_t y) { return x + y; });
}

void sym_sub_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  register_sym_binary(graph, args, [](int32_t x, int32_t y) { return x - y; });
}

void sym_mul_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  register_sym_binary(graph, args, [](int32_t x, int32_t y) { return x * y; });
}

void sym_floordiv_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  register_sym_binary(graph, args, [](int32_t x, int32_t y) {
    int32_t q = x / y;
    if ((x % y != 0) && ((x < 0) != (y < 0))) {
      q--; // round toward negative infinity (Python floor division)
    }
    return q;
  });
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.select_as_symint.default, select_as_symint_impl);
  WEBGPU_REGISTER_OP(add, sym_add_impl);
  WEBGPU_REGISTER_OP(sub, sym_sub_impl);
  WEBGPU_REGISTER_OP(mul, sym_mul_impl);
  WEBGPU_REGISTER_OP(floordiv, sym_floordiv_impl);
}

} // namespace executorch::backends::webgpu
