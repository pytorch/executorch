/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/StagingUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_conv2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  vTensor& out = graph->get_val(args[0].refs[0]).toTensor();
  vTensor& self = graph->get_val(args[1].refs[0]).toTensor();

  size_t ndim = self.sizes().size();
  std::vector<int64_t> new_out_sizes(ndim);

  // Batch, Channel
  if (ndim == 4) {
    new_out_sizes.at(ndim - 4) = self.sizes().at(ndim - 4);
  }
  const auto weight_sizes = graph->get_val(extra_args[0]).toTensorRef().sizes;
  new_out_sizes.at(ndim - 3) = weight_sizes.at(ndim - 4);

  // Height, Width
  const auto new_out_sizes_hw = calc_out_sizes_hw(
      *graph,
      self.sizes(),
      extra_args[0],
      /*kernel_size_only = */ false,
      extra_args[1],
      extra_args[2],
      extra_args[3]);
  new_out_sizes.at(ndim - 2) = new_out_sizes_hw.at(0);
  new_out_sizes.at(ndim - 1) = new_out_sizes_hw.at(1);

  out.virtual_resize(new_out_sizes);
}

ValueRef prepack_biases(ComputeGraph& graph, const ValueRef vref) {
  if (graph.get_val(vref).isNone()) {
    VK_THROW("aten.convolution.default: Null bias is not supported yet!");
  }

  ValueRef v = graph.add_tensor_like(vref, api::kTexture2D, api::kWidthPacked);
  vTensor& t = graph.get_val(v).toTensor();

  api::ShaderInfo shader = get_nchw_to_image_shader(t);

  api::utils::uvec3 global_size = t.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      global_size,
      local_size,
      vref,
      v,
      {t.gpu_sizes_ubo(), t.cpu_sizes_ubo()}));

  return v;
}

api::ShaderInfo get_conv2d_shader(const vTensor& t_out, bool prepack_weights) {
  std::stringstream kernel_name;
  kernel_name << "conv2d";
  if (prepack_weights) {
    kernel_name << "_prepack_weights";
  }
  apply_dtype_suffix(kernel_name, t_out);

  return VK_KERNEL_FROM_STR(kernel_name.str());
}

ValueRef prepack_weights(ComputeGraph& graph, const ValueRef vref) {
  const auto original_sizes = graph.get_val(vref).toTensorRef().sizes;

  int64_t batch_padded =
      api::utils::align_up(api::utils::val_at(-4, original_sizes), INT64_C(4));
  int64_t channels_padded =
      api::utils::align_up(api::utils::val_at(-3, original_sizes), INT64_C(4));
  int64_t height = api::utils::val_at(-2, original_sizes);
  int64_t width = api::utils::val_at(-1, original_sizes);

  const auto final_sizes = std::vector<int64_t>{
      4, batch_padded * height / 4, channels_padded * width};

  ValueRef v = graph.add_tensor(
      final_sizes,
      graph.get_val(vref).toTensorRef().dtype,
      api::kTexture2D,
      api::kChannelsPacked);
  vTensor& t = graph.get_val(v).toTensor();

  api::utils::uvec3 global_size = t.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  api::ShaderInfo shader = get_conv2d_shader(t, /*prepack_weights = */ true);

  const auto padded_sizes = std::vector<int64_t>{batch_padded, channels_padded};

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      global_size,
      local_size,
      vref,
      v,
      {t.gpu_sizes_ubo(),
       graph.create_params_buffer(
           api::utils::make_ivec4(original_sizes, /*reverse = */ true)),
       graph.create_params_buffer(
           api::utils::make_ivec2(padded_sizes, /*reverse = */ true))}));

  return v;
}

void check_conv2d_args(const vTensor& in, const vTensor& out) {
  if (in.sizes().at(0) > 1) {
    VK_THROW(
        "aten.convolution.default: input batch size > 1 is not supported yet!");
  }
  VK_CHECK_COND(check_memory_layout_is(in, api::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(out, api::kChannelsPacked));
}

struct Conv2dParams final {
  api::utils::ivec2 overlay_region;
  int in_group_size;
};

Conv2dParams create_conv2d_params(
    ComputeGraph& graph,
    const ValueRef weight,
    const KernelParams& p) {
  const auto overlay_region = api::utils::make_ivec2({
      p.kernel_size.data[0] +
          (p.kernel_size.data[0] - 1) * (p.dilation.data[0] - 1),
      p.kernel_size.data[1] +
          (p.kernel_size.data[1] - 1) * (p.dilation.data[1] - 1),
  });
  const auto weight_sizes = graph.get_val(weight).toTensorRef().sizes;
  const int32_t in_group_size = api::utils::safe_downcast<int32_t>(
      api::utils::align_up(weight_sizes.at(1), INT64_C(4)));
  return {overlay_region, in_group_size};
}

void check_conv2d_params(const KernelParams& p) {
  if ((p.padding.data[0] > 0 && p.kernel_size.data[0] > 1 &&
       p.dilation.data[0] > 1) ||
      (p.padding.data[1] > 0 && p.kernel_size.data[1] > 1 &&
       p.dilation.data[1] > 1)) {
    VK_THROW(
        "aten.convolution.default: padding > 0 while dilation, kernel_size > 1 is not supported yet!");
  }
}

void add_conv2d_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out) {
  ValueRef arg_in = prepack_if_tensor_ref(graph, in);
  ValueRef arg_weight = prepack_weights(graph, weight);
  ValueRef arg_bias = prepack_biases(graph, bias);

  vTensor& t_in = graph.get_val(arg_in).toTensor();
  vTensor& t_out = graph.get_val(out).toTensor();

  check_conv2d_args(t_in, t_out);

  api::utils::uvec3 global_size = t_out.virtual_extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  KernelParams kernel_params = create_kernel_params(
      graph,
      weight,
      /*kernel_size_only = */ false,
      stride,
      padding,
      dilation);
  Conv2dParams extra_params =
      create_conv2d_params(graph, weight, kernel_params);

  check_conv2d_params(kernel_params);

  api::ShaderInfo shader =
      get_conv2d_shader(t_out, /*prepack_weights = */ false);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      shader,
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, api::MemoryAccessType::WRITE},
       {{arg_in, arg_weight, arg_bias}, api::MemoryAccessType::READ}},
      // Shader params buffers
      {
          t_out.extents_ubo(),
          t_in.extents_ubo(),
          graph.create_params_buffer(kernel_params),
          graph.create_params_buffer(extra_params),
      },
      // Resizing
      resize_conv2d_node,
      {weight, stride, padding, dilation}));
}

void conv2d(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const bool transposed = graph.get_val(args[6]).toBool();
  if (transposed) {
    VK_THROW("aten.convolution.default: transpose is not supported yet!");
  }
  const int64_t groups = graph.get_val(args[8]).toInt();
  if (groups > 1) {
    VK_THROW("aten.convolution.default: groups > 1 is not supported yet!");
  }
  return add_conv2d_node(
      graph, args[0], args[1], args[2], args[3], args[4], args[5], args[9]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.convolution.default, conv2d);
}

} // namespace vkcompute
