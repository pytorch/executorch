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

constexpr float kDummyFloat = -1.0f;
const std::string kClampShaderName = "clamp";

void resize_unary_op_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);

  const std::vector<int64_t> self_sizes = graph->sizes_of(self);
  graph->virtual_resize(out, self_sizes);
}

void add_unary_op_node(
    ComputeGraph& graph,
    const ValueRef in,
    const float min,
    const float max,
    const ValueRef out,
    const std::string& op_name,
    const ValueRef other = kDummyValueRef) {
  std::string kernel_name(op_name);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));

  std::vector<ArgGroup> args = {{out, vkapi::kWrite}, {in, vkapi::kRead}};
  std::vector<ArgGroup> args_with_binary_op = {
      {out, vkapi::kWrite}, {{in, other}, vkapi::kRead}};

  const utils::vec2 min_max = {min, max};
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      other == kDummyValueRef ? args : args_with_binary_op,
      // Shader params buffers
      {},
      // Push Constants
      {
          graph.is_buffer_storage(out) ? graph.numel_pc_of(out)
                                       : graph.logical_limits_pc_of(out),
          PushConstantDataInfo(&min_max, sizeof(min_max)),
      },
      // pcs,
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_unary_op_node));
}

float get_val_or_inf(ComputeGraph& graph, const ValueRef& val, bool max) {
  if (!graph.val_is_none(val)) {
    return graph.extract_scalar<float>(val);
  }
  return max ? std::numeric_limits<float>::infinity()
             : -std::numeric_limits<float>::infinity();
}

#define DEFINE_ACTIVATION_FN(op_name)                                    \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_unary_op_node(                                            \
        graph, args[0], kDummyFloat, kDummyFloat, args[1], #op_name);    \
  }

#define DEFINE_CLAMP_FN(op_name)                                         \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_unary_op_node(                                            \
        graph,                                                           \
        args[0],                                                         \
        get_val_or_inf(graph, args[1], /*max = */ false),                \
        get_val_or_inf(graph, args[2], /*max = */ true),                 \
        args[3],                                                         \
        kClampShaderName);                                               \
  }

#define DEFINE_CLAMP_BINARY_OP_FN(op_name)                               \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_unary_op_node(                                            \
        graph,                                                           \
        args[0],                                                         \
        get_val_or_inf(graph, args[1], /*max = */ false),                \
        get_val_or_inf(graph, args[2], /*max = */ true),                 \
        args[args.size() - 1],                                           \
        #op_name,                                                        \
        args[3]);                                                        \
  }

#define DEFINE_RELU_FN(op_name)                                          \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_unary_op_node(                                            \
        graph,                                                           \
        args[0],                                                         \
        0,                                                               \
        std::numeric_limits<float>::infinity(),                          \
        args[1],                                                         \
        kClampShaderName);                                               \
  }

#define DEFINE_RELU6_FN(op_name)                                               \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) {       \
    return add_unary_op_node(graph, args[0], 0, 6, args[1], kClampShaderName); \
  }

#define DEFINE_HARDSHRINK_FN(op_name)                                    \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_unary_op_node(                                            \
        graph,                                                           \
        args[0],                                                         \
        get_val_or_inf(graph, args[1], /*max = */ false),                \
        -get_val_or_inf(graph, args[1], /*max = */ true),                \
        args[2],                                                         \
        "hardshrink");                                                   \
  }

#define DEFINE_LEAKY_RELU_FN(op_name)                                    \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_unary_op_node(                                            \
        graph,                                                           \
        args[0],                                                         \
        get_val_or_inf(graph, args[1], /*neg slope*/ false),             \
        kDummyFloat,                                                     \
        args[2],                                                         \
        "leaky_relu");                                                   \
  }

void gelu(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // args[1] is the `approximate` string
  // https://fburl.com/code/9omngmyo
  // currently only `approximate = "tanh"` is supported
  return add_unary_op_node(
      graph, args[0], kDummyFloat, kDummyFloat, args[2], "gelu");
}

DEFINE_ACTIVATION_FN(abs);
DEFINE_ACTIVATION_FN(cos);
DEFINE_ACTIVATION_FN(exp);
DEFINE_ACTIVATION_FN(neg);
DEFINE_ACTIVATION_FN(sigmoid);
DEFINE_ACTIVATION_FN(sin);
DEFINE_ACTIVATION_FN(sqrt);
DEFINE_ACTIVATION_FN(rsqrt);
DEFINE_ACTIVATION_FN(tanh);
DEFINE_CLAMP_FN(clamp);
DEFINE_CLAMP_FN(hardtanh);
DEFINE_RELU_FN(relu);
DEFINE_RELU6_FN(relu6);
DEFINE_HARDSHRINK_FN(hardshrink);
DEFINE_ACTIVATION_FN(hardswish);
DEFINE_ACTIVATION_FN(hardsigmoid);
DEFINE_LEAKY_RELU_FN(leaky_relu);
DEFINE_ACTIVATION_FN(round);

DEFINE_CLAMP_BINARY_OP_FN(clamp_add);
DEFINE_CLAMP_BINARY_OP_FN(clamp_sub);
DEFINE_CLAMP_BINARY_OP_FN(clamp_mul);
DEFINE_CLAMP_BINARY_OP_FN(clamp_div);

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.abs.default, abs);
  VK_REGISTER_OP(aten.clamp.default, clamp);
  VK_REGISTER_OP(aten.cos.default, cos);
  VK_REGISTER_OP(aten.exp.default, exp);
  VK_REGISTER_OP(aten.gelu.default, gelu);
  VK_REGISTER_OP(aten.hardtanh.default, hardtanh);
  VK_REGISTER_OP(aten.neg.default, neg);
  VK_REGISTER_OP(aten.relu.default, relu);
  VK_REGISTER_OP(aten.relu6.default, relu6);
  VK_REGISTER_OP(aten.sigmoid.default, sigmoid);
  VK_REGISTER_OP(aten.sin.default, sin);
  VK_REGISTER_OP(aten.sqrt.default, sqrt);
  VK_REGISTER_OP(aten.rsqrt.default, rsqrt);
  VK_REGISTER_OP(aten.tanh.default, tanh);
  VK_REGISTER_OP(aten.hardshrink.default, hardshrink);
  VK_REGISTER_OP(aten.hardswish.default, hardswish);
  VK_REGISTER_OP(aten.hardsigmoid.default, hardsigmoid);
  VK_REGISTER_OP(aten.leaky_relu.default, leaky_relu);
  VK_REGISTER_OP(aten.round.default, round);

  VK_REGISTER_OP(et_vk.clamp_with_binary_add.default, clamp_add);
  VK_REGISTER_OP(et_vk.clamp_with_binary_sub.default, clamp_sub);
  VK_REGISTER_OP(et_vk.clamp_with_binary_mul.default, clamp_mul);
  VK_REGISTER_OP(et_vk.clamp_with_binary_div.default, clamp_div);
}

} // namespace vkcompute
