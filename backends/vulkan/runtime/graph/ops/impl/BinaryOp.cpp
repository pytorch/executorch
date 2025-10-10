/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_binary_op_args(
    ComputeGraph& graph,
    const ValueRef self,
    const ValueRef other,
    const ValueRef out) {
  VK_CHECK_COND(graph.packed_dim_of(self) == graph.packed_dim_of(other));
  VK_CHECK_COND(graph.packed_dim_of(self) == graph.packed_dim_of(out));

  const std::vector<int64_t> self_sizes = graph.sizes_of(self);
  const std::vector<int64_t> other_sizes = graph.sizes_of(other);
  const std::vector<int64_t> out_sizes = graph.sizes_of(out);

  std::vector<int64_t> broadcasted_sizes =
      calculate_broadcasted_output_size(self_sizes, other_sizes);
  VK_CHECK_COND(out_sizes == broadcasted_sizes);
}

void resize_binary_op_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);

  // TODO(T183442143): Verify tensors are broadcastable.
  const ValueRef self = args.at(1).refs.at(0);
  const ValueRef other = args.at(1).refs.at(1);

  const std::vector<int64_t> self_sizes = graph->sizes_of(self);
  const std::vector<int64_t> other_sizes = graph->sizes_of(other);
  const std::vector<int64_t> new_out_sizes =
      calculate_broadcasted_output_size(self_sizes, other_sizes);

  graph->virtual_resize(out, new_out_sizes);
}

int remove_clamp_from_name(std::string& op) {
  if (op.find("clamp_0_with_") != std::string::npos) {
    op.erase(op.find("clamp_0_with_"), 13);

    // Clamp input 0
    return 1;
  }
  if (op.find("clamp_1_with_") != std::string::npos) {
    op.erase(op.find("clamp_1_with_"), 13);

    // Clamp input 1
    return 2;
  }
  if (op.find("_with_clamp") != std::string::npos) {
    op.erase(op.find("_with_clamp"), 11);

    // Clamp output
    return 3;
  }

  // No clamp
  return 0;
}

void add_binary_op_texture_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const ValueRef alpha,
    const ValueRef out,
    const std::string& op_name,
    const float min,
    const float max) {
  ValueRef arg1 = prepack_standard_like(graph, in1, out, true);
  ValueRef arg2 = prepack_standard_like(graph, in2, out, true);

  check_binary_op_args(graph, arg1, arg2, out);

  float alpha_val = 1.0f;
  // String is checked since floor_div passes in an unused string argument in
  // place of alpha
  if (is_valid(alpha) && !graph.val_is_string(alpha)) {
    alpha_val = graph.extract_scalar<float>(alpha);
  }

  const struct BinaryOpsParams {
    const utils::ivec2 broadcast_params;
    const float alpha_val;
  } binary_ops_params{create_broadcast_params(graph, arg1, arg2), alpha_val};

  std::string kernel_name("binary_");
  kernel_name.reserve(kShaderNameReserve);

  std::string op = op_name;
  int clamp_type = remove_clamp_from_name(op);
  kernel_name += op;
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(in1));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{arg1, arg2}, vkapi::kRead}},
      // Shader params buffers
      {},
      // Push Constants
      {{graph.sizes_pc_of(out),
        graph.sizes_pc_of(arg1),
        graph.sizes_pc_of(arg2),
        PushConstantDataInfo(&binary_ops_params, sizeof(binary_ops_params))}},
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(arg1),
       graph.hashed_layout_of(arg2),
       clamp_type,
       min,
       max},
      // Resize Args
      {},
      // Resizing Logic
      resize_binary_op_node));
}

void add_binary_op_buffer_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const ValueRef alpha,
    const ValueRef out,
    const std::string& op_name,
    const float min,
    const float max) {
  // check_binary_op_args(*t_in1, *t_in2, *t_out);

  float alpha_val = 1.0f;
  // String is checked since floor_div passes in an unused string argument in
  // place of alpha
  if (is_valid(alpha) && !graph.val_is_string(alpha)) {
    alpha_val = graph.extract_scalar<float>(alpha);
  }

  std::string kernel_name("binary_");
  kernel_name.reserve(kShaderNameReserve);
  std::string op = op_name;
  int clamp_type = remove_clamp_from_name(op);
  kernel_name += op;
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));

  add_dtype_suffix(kernel_name, graph.dtype_of(in1));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{in1, in2}, vkapi::kRead}},
      // Shader params buffers
      {graph.buffer_meta_ubo(out),
       graph.buffer_meta_ubo(in1),
       graph.buffer_meta_ubo(in2)},
      // Push Constants
      {{
          PushConstantDataInfo(&alpha_val, sizeof(float)),
      }},
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(in1),
       graph.hashed_layout_of(in2),
       min,
       max},
      // Resize Args
      {},
      // Resizing Logic
      resize_binary_op_node));
}

void add_binary_op_node(
    ComputeGraph& graph,
    const ValueRef in1,
    const ValueRef in2,
    const ValueRef alpha,
    const ValueRef out,
    const std::string& op_name,
    const float min = std::numeric_limits<float>::infinity(),
    const float max = -std::numeric_limits<float>::infinity()) {
  if (graph.is_buffer_storage(out)) {
    add_binary_op_buffer_node(graph, in1, in2, alpha, out, op_name, min, max);
  } else {
    add_binary_op_texture_node(graph, in1, in2, alpha, out, op_name, min, max);
  }
}

#define DEFINE_BINARY_OP_WITH_ALPHA_FN(op_name)                          \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_binary_op_node(                                           \
        graph, args[0], args[1], args[2], args[3], #op_name);            \
  }

#define DEFINE_BINARY_OP_FN(op_name)                                     \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_binary_op_node(                                           \
        graph, args[0], args[1], kDummyValueRef, args[2], #op_name);     \
  }

float get_val_or_inf_(ComputeGraph& graph, const ValueRef& val, bool max) {
  if (!graph.val_is_none(val)) {
    return graph.extract_scalar<float>(val);
  }
  return max ? std::numeric_limits<float>::infinity()
             : -std::numeric_limits<float>::infinity();
}

#define DEFINE_BINARY_OP_WITH_ALPHA_FN_CLAMPED(op_name)                  \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_binary_op_node(                                           \
        graph,                                                           \
        args[0],                                                         \
        args[1],                                                         \
        args[2],                                                         \
        args[5],                                                         \
        #op_name,                                                        \
        get_val_or_inf_(graph, args[3], false),                          \
        get_val_or_inf_(graph, args[4], true));                          \
  }

#define DEFINE_BINARY_OP_FN_CLAMPED(op_name)                             \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_binary_op_node(                                           \
        graph,                                                           \
        args[0],                                                         \
        args[1],                                                         \
        kDummyValueRef,                                                  \
        args[4],                                                         \
        #op_name,                                                        \
        get_val_or_inf_(graph, args[2], false),                          \
        get_val_or_inf_(graph, args[3], true));                          \
  }

DEFINE_BINARY_OP_WITH_ALPHA_FN(add);
DEFINE_BINARY_OP_WITH_ALPHA_FN(sub);

// Floor div does not have an alpha, but a string argument (which is unused) is
// passed in at the same location as the alpha argument in other op.
DEFINE_BINARY_OP_WITH_ALPHA_FN(floor_divide);

DEFINE_BINARY_OP_FN(mul);
DEFINE_BINARY_OP_FN(div);
DEFINE_BINARY_OP_FN(pow);
DEFINE_BINARY_OP_FN(minimum);
DEFINE_BINARY_OP_FN(eq);
DEFINE_BINARY_OP_FN(lt);
DEFINE_BINARY_OP_FN(le);
DEFINE_BINARY_OP_FN(gt);
DEFINE_BINARY_OP_FN(ge);

DEFINE_BINARY_OP_FN_CLAMPED(add_with_clamp);
DEFINE_BINARY_OP_FN_CLAMPED(sub_with_clamp);
DEFINE_BINARY_OP_FN_CLAMPED(mul_with_clamp);
DEFINE_BINARY_OP_FN_CLAMPED(div_with_clamp);

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.add.Tensor, add);
  VK_REGISTER_OP(aten.sub.Tensor, sub);
  VK_REGISTER_OP(aten.mul.Tensor, mul);
  VK_REGISTER_OP(aten.div.Tensor, div);
  VK_REGISTER_OP(aten.div.Tensor_mode, floor_divide);
  VK_REGISTER_OP(aten.pow.Tensor_Tensor, pow);
  VK_REGISTER_OP(aten.minimum.default, minimum);
  VK_REGISTER_OP(aten.eq.Tensor, eq);
  VK_REGISTER_OP(aten.lt.Tensor, lt);
  VK_REGISTER_OP(aten.le.Tensor, le);
  VK_REGISTER_OP(aten.gt.Tensor, gt);
  VK_REGISTER_OP(aten.ge.Tensor, ge);

  VK_REGISTER_OP(et_vk.binary_add_with_clamp.default, add_with_clamp);
  VK_REGISTER_OP(et_vk.binary_sub_with_clamp.default, sub_with_clamp);
  VK_REGISTER_OP(et_vk.binary_mul_with_clamp.default, mul_with_clamp);
  VK_REGISTER_OP(et_vk.binary_div_with_clamp.default, div_with_clamp);
}

} // namespace vkcompute
