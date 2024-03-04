/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Arithmetic.h>

#include <ATen/native/vulkan/impl/Common.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

namespace at {
namespace native {
namespace vulkan {

#define DEFINE_ARITHMETIC_FN(function, op_type)                               \
  ValueRef function(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_arithmetic_node(                                               \
        graph,                                                                \
        args[0],                                                              \
        args[1],                                                              \
        args[2],                                                              \
        arithmetic::OpType::op_type,                                          \
        args[3]);                                                             \
  }

DEFINE_ARITHMETIC_FN(add, ADD);
DEFINE_ARITHMETIC_FN(sub, SUB);
DEFINE_ARITHMETIC_FN(mul, MUL);
DEFINE_ARITHMETIC_FN(div, DIV);
DEFINE_ARITHMETIC_FN(floor_div, FLOOR_DIV);
DEFINE_ARITHMETIC_FN(pow, POW);

ValueRef add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const float alpha,
    const arithmetic::OpType optype,
    const int64_t shared_object_idx) {
  std::vector<int64_t> in1_sizes = graph.get_val_sizes(in1);
  api::ScalarType in1_dtype = graph.get_val_dtype(in1);

  ValueRef out = graph.add_tensor(in1_sizes, in1_dtype, shared_object_idx);
  add_arithmetic_node(graph, in1, in2, out, alpha, optype);
  return out;
}

// TODO(T181006464): Move to Utils when we remove ArithmeticPrepack.
ValueRef prepack_if_tensor_ref(ComputeGraph& graph, const ValueRef v) {
  if (graph.get_val(v).isTensor()) {
    return v;
  } else {
    TensorRef& tRef = graph.get_val(v).toTensorRef();
    ValueRef vTen = graph.add_tensor(tRef.sizes, tRef.dtype);
    graph.prepack_nodes().emplace_back(new ArithmeticPrepack(v, vTen));
    return vTen;
  }
}

void add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const ValueRef out,
    const float alpha,
    const arithmetic::OpType optype) {
  ValueRef arg1 = prepack_if_tensor_ref(graph, in1);
  ValueRef arg2 = prepack_if_tensor_ref(graph, in2);

  graph.execute_nodes().emplace_back(
      new ArithmeticNode(arg1, arg2, out, alpha, optype));
}

ArithmeticPrepack::ArithmeticPrepack(const ValueRef tref, const ValueRef packed)
    : PrepackNode(tref, packed) {}

void ArithmeticPrepack::encode(ComputeGraph* graph) const {
  TensorRef tref = graph->get_val(tref_).toTensorRef();
  vTensor packed = graph->get_val(packed_).toTensor();

  api::StorageBuffer staging(
      graph->context(), packed.dtype(), packed.gpu_nbytes());

  size_t numel = api::utils::multiply_integers(tref.sizes);
  size_t nbytes = numel * api::element_size(tref.dtype);
  copy_ptr_to_staging(tref.data, staging, nbytes);

  encode_copy_to_vtensor(graph->context(), staging, packed);
}

ArithmeticNode::ArithmeticNode(
    const ValueRef in1,
    const ValueRef in2,
    const ValueRef out,
    const float alpha,
    const arithmetic::OpType optype)
    : ExecuteNode({in1, in2}, {out}), alpha_(alpha), optype_(optype) {}

void ArithmeticNode::encode(ComputeGraph* graph) const {
  vTensor& in1 = graph->get_val(inputs_[0]).toTensor();
  vTensor& in2 = graph->get_val(inputs_[1]).toTensor();
  vTensor& out = graph->get_val(outputs_[0]).toTensor();

  api::ShaderInfo kernel = arithmetic::get_shader(optype_);
  arithmetic::record_op(graph->context(), kernel, in1, in2, out, alpha_);
}

} // namespace vulkan
} // namespace native
} // namespace at
