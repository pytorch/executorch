/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/native/vulkan/impl/Common.h>

#include <executorch/backends/vulkan/runtime/graph/ops/Arithmetic.h>
#include <executorch/backends/vulkan/runtime/graph/ops/Staging.h>

namespace at {
namespace native {
namespace vulkan {

void add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef t1,
    const ValueRef t2,
    const ValueRef out,
    const float alpha,
    const arithmetic::OpType optype) {
  // Prepacking first arg (if needed)
  ValueRef arg1 = t1;
  if (graph.get_val(t1).isTensorRef()) {
    TensorRef& t1_asref = graph.get_val(t1).toTensorRef();
    ValueRef t1_vten = graph.add_tensor(t1_asref.sizes, t1_asref.dtype);
    graph.prepack_nodes().emplace_back(new ArithmeticPrepack(t1, t1_vten));
    arg1 = t1_vten;
  }
  VK_CHECK_COND(graph.get_val(arg1).isTensor());
  // Prepacking second arg (if needed)
  ValueRef arg2 = t2;
  if (graph.get_val(t2).isTensorRef()) {
    TensorRef& t2_asref = graph.get_val(t2).toTensorRef();
    ValueRef t2_vten = graph.add_tensor(t2_asref.sizes, t2_asref.dtype);
    graph.prepack_nodes().emplace_back(new ArithmeticPrepack(t2, t2_vten));
    arg2 = t2_vten;
  }
  VK_CHECK_COND(graph.get_val(arg2).isTensor());

  graph.execute_nodes().emplace_back(
      new ArithmeticNode(arg1, arg2, out, alpha, optype));
}

ValueRef add_arithmetic_node(
    ComputeGraph& graph,
    const ValueRef t1,
    const ValueRef t2,
    const float alpha,
    const arithmetic::OpType optype,
    const int64_t shared_object_idx) {
  std::vector<int64_t> t1_sizes = graph.get_val_sizes(t1);
  api::ScalarType t1_dtype = graph.get_val_dtype(t1);

  ValueRef out = graph.add_tensor(t1_sizes, t1_dtype, shared_object_idx);
  add_arithmetic_node(graph, t1, t2, out, alpha, optype);
  return out;
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
    const ValueRef t1,
    const ValueRef t2,
    const ValueRef out,
    const float alpha,
    const arithmetic::OpType optype)
    : ExecuteNode({t1, t2}, {out}), alpha_(alpha), optype_(optype) {}

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
