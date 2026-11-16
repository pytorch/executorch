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
#include <executorch/backends/vulkan/runtime/graph/ops/impl/unary/UnaryOp.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

constexpr float kDummyFloat = -1.0f;

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

void check_unary_op_inputs(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out) {
  VK_CHECK_COND(graph.storage_type_of(in) == graph.storage_type_of(out));
  VK_CHECK_COND(graph.dtype_of(in) == graph.dtype_of(out));
  // The shader reads and writes the same texel index.
  if (graph.is_buffer_storage(out)) {
    VK_CHECK_COND(graph.hashed_layout_of(in) == graph.hashed_layout_of(out));
    return;
  }
  VK_CHECK_COND(graph.has_standard_axis_map(in));
  VK_CHECK_COND(graph.has_standard_axis_map(out));
  VK_CHECK_COND(graph.packed_dim_of(in) == graph.packed_dim_of(out));
}

void add_unary_op_node(
    ComputeGraph& graph,
    const ValueRef in,
    const float min,
    const float max,
    const ValueRef out,
    const UnaryOpType op_type) {
  check_unary_op_inputs(graph, in, out);
  std::string kernel_name("unary_op");
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));

  const utils::vec2 min_max = {min, max};
  vkapi::ParamsBindList param_ubos = {graph.meta_ubo(out)};
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_vec4_gwg,
      default_pick_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Shader params buffers
      param_ubos,
      // Push Constants
      {
          PushConstantDataInfo(&min_max, sizeof(min_max)),
      },
      // Specialization Constants
      {static_cast<int32_t>(op_type), graph.hashed_layout_of(out)},
      // Resize Args
      {},
      // Resizing Logic
      resize_unary_op_node));
}

void add_dynamic_clamp_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef min,
    const ValueRef max,
    const ValueRef out) {
  check_unary_op_inputs(graph, in, out);
  std::string kernel_name("unary_op_dynamic");
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));

  const bool output_is_int = graph.dtype_of(out) == vkapi::kInt;
  const utils::ivec2 bounds_are_int = {
      output_is_int || graph.val_is_symint(min) ? 1 : 0,
      output_is_int || graph.val_is_symint(max) ? 1 : 0};
  const vkapi::BufferBindInfo min_param = bounds_are_int[0]
      ? graph.get_or_create_int_param_buffer(
            min, std::numeric_limits<int32_t>::min())
      : graph.create_params_buffer(graph.extract_scalar_or<float>(
            min, -std::numeric_limits<float>::infinity()));
  const vkapi::BufferBindInfo max_param = bounds_are_int[1]
      ? graph.get_or_create_int_param_buffer(
            max, std::numeric_limits<int32_t>::max())
      : graph.create_params_buffer(graph.extract_scalar_or<float>(
            max, std::numeric_limits<float>::infinity()));

  vkapi::ParamsBindList param_ubos = {graph.meta_ubo(out)};
  param_ubos.append(min_param);
  param_ubos.append(max_param);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_vec4_gwg,
      default_pick_lwg,
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      param_ubos,
      {PushConstantDataInfo(&bounds_are_int, sizeof(bounds_are_int))},
      {static_cast<int32_t>(UnaryOpType::kClamp), graph.hashed_layout_of(out)},
      {},
      resize_unary_op_node));
}

float get_val_or_inf(ComputeGraph& graph, const ValueRef& val, bool max) {
  if (!graph.val_is_none(val)) {
    return graph.extract_scalar<float>(val);
  }
  return max ? std::numeric_limits<float>::infinity()
             : -std::numeric_limits<float>::infinity();
}

#define DEFINE_ACTIVATION_FN(op_name, op_type)                           \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_unary_op_node(                                            \
        graph,                                                           \
        args[0],                                                         \
        kDummyFloat,                                                     \
        kDummyFloat,                                                     \
        args[1],                                                         \
        UnaryOpType::op_type);                                           \
  }

#define DEFINE_CLAMP_FN(op_name)                                         \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    if (graph.val_is_symint(args[1]) || graph.val_is_symint(args[2])) {  \
      return add_dynamic_clamp_node(                                     \
          graph, args[0], args[1], args[2], args[3]);                    \
    }                                                                    \
    return add_unary_op_node(                                            \
        graph,                                                           \
        args[0],                                                         \
        get_val_or_inf(graph, args[1], /*max = */ false),                \
        get_val_or_inf(graph, args[2], /*max = */ true),                 \
        args[3],                                                         \
        UnaryOpType::kClamp);                                            \
  }

#define DEFINE_RELU6_FN(op_name)                                         \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_unary_op_node(                                            \
        graph, args[0], 0, 6, args[1], UnaryOpType::kClamp);             \
  }

#define DEFINE_HARDSHRINK_FN(op_name)                                    \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_unary_op_node(                                            \
        graph,                                                           \
        args[0],                                                         \
        get_val_or_inf(graph, args[1], /*max = */ false),                \
        -get_val_or_inf(graph, args[1], /*max = */ true),                \
        args[2],                                                         \
        UnaryOpType::kHardshrink);                                       \
  }

#define DEFINE_LEAKY_RELU_FN(op_name)                                    \
  void op_name(ComputeGraph& graph, const std::vector<ValueRef>& args) { \
    return add_unary_op_node(                                            \
        graph,                                                           \
        args[0],                                                         \
        get_val_or_inf(graph, args[1], /*neg slope*/ false),             \
        kDummyFloat,                                                     \
        args[2],                                                         \
        UnaryOpType::kLeakyRelu);                                        \
  }

void gelu(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const std::string approximate = graph.extract_string(args[1]);
  VK_CHECK_COND(approximate == "none" || approximate == "tanh");
  return add_unary_op_node(
      graph,
      args[0],
      kDummyFloat,
      kDummyFloat,
      args[2],
      approximate == "tanh" ? UnaryOpType::kGelu : UnaryOpType::kGeluErf);
}

DEFINE_ACTIVATION_FN(abs, kAbs);
DEFINE_ACTIVATION_FN(cos, kCos);
DEFINE_ACTIVATION_FN(exp, kExp);
DEFINE_ACTIVATION_FN(neg, kNeg);
DEFINE_ACTIVATION_FN(sigmoid, kSigmoid);
DEFINE_ACTIVATION_FN(sin, kSin);
DEFINE_ACTIVATION_FN(sqrt, kSqrt);
DEFINE_ACTIVATION_FN(rsqrt, kRsqrt);
DEFINE_ACTIVATION_FN(tanh, kTanh);
DEFINE_ACTIVATION_FN(tan, kTan);
DEFINE_CLAMP_FN(clamp);
DEFINE_CLAMP_FN(hardtanh);
DEFINE_ACTIVATION_FN(relu, kRelu);
DEFINE_RELU6_FN(relu6);
DEFINE_HARDSHRINK_FN(hardshrink);
DEFINE_ACTIVATION_FN(hardswish, kHardswish);
DEFINE_ACTIVATION_FN(hardsigmoid, kHardsigmoid);
DEFINE_LEAKY_RELU_FN(leaky_relu);
DEFINE_ACTIVATION_FN(log10, kLog10);
DEFINE_ACTIVATION_FN(round, kRound);
DEFINE_ACTIVATION_FN(bitwise_not, kBitwiseNot);

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
  VK_REGISTER_OP(aten.tan.default, tan);
  VK_REGISTER_OP(aten.hardshrink.default, hardshrink);
  VK_REGISTER_OP(aten.hardswish.default, hardswish);
  VK_REGISTER_OP(aten.hardsigmoid.default, hardsigmoid);
  VK_REGISTER_OP(aten.leaky_relu.default, leaky_relu);
  VK_REGISTER_OP(aten.log10.default, log10);
  VK_REGISTER_OP(aten.round.default, round);
  VK_REGISTER_OP(aten.bitwise_not.default, bitwise_not);
  VK_REGISTER_OP(aten.logical_not.default, bitwise_not);
}

} // namespace vkcompute
