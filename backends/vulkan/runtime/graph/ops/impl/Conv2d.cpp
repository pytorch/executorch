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

struct Conv2dParams final {
  api::utils::ivec2 overlay_region;
  int in_group_size;
};

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

  const auto kernel_size =
      api::utils::make_ivec2({weight_sizes.at(3), weight_sizes.at(2)});
  const auto stride = reverse(*graph, extra_args[1]);
  const auto padding = reverse(*graph, extra_args[2]);
  const auto dilation = reverse(*graph, extra_args[3]);

  // Height, Width
  new_out_sizes.at(ndim - 2) = calc_out_size(
      self.sizes().at(ndim - 2),
      kernel_size.data[1],
      stride.data[1],
      padding.data[1],
      dilation.data[1]);
  new_out_sizes.at(ndim - 1) = calc_out_size(
      self.sizes().at(ndim - 1),
      kernel_size.data[0],
      stride.data[0],
      padding.data[0],
      dilation.data[0]);

  VK_CHECK_COND(new_out_sizes.at(ndim - 2) >= 1);
  VK_CHECK_COND(new_out_sizes.at(ndim - 1) >= 1);

  out.virtual_resize(new_out_sizes);
}

ValueRef prepack_biases(ComputeGraph& graph, const ValueRef vref) {
  if (graph.get_val(vref).isNone()) {
    VK_THROW("aten.convolution.default: Null bias is not supported yet!");
  }

  TensorRef& tref = graph.get_val(vref).toTensorRef();
  ValueRef v = graph.add_tensor(
      tref.sizes,
      tref.dtype,
      api::StorageType::TEXTURE_2D,
      api::GPUMemoryLayout::TENSOR_WIDTH_PACKED);
  vTensor t = graph.get_val(v).toTensor();

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

ValueRef prepack_weights(ComputeGraph& graph, const ValueRef vref) {
  TensorRef& tref = graph.get_val(vref).toTensorRef();

  int64_t batch_padded =
      api::utils::align_up(api::utils::val_at(-4, tref.sizes), INT64_C(4));
  int64_t channels_padded =
      api::utils::align_up(api::utils::val_at(-3, tref.sizes), INT64_C(4));
  int64_t height = api::utils::val_at(-2, tref.sizes);
  int64_t width = api::utils::val_at(-1, tref.sizes);

  const auto final_sizes = std::vector<int64_t>{
      4, batch_padded * height / 4, channels_padded * width};

  ValueRef v = graph.add_tensor(
      final_sizes,
      tref.dtype,
      api::StorageType::TEXTURE_2D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED);
  vTensor t = graph.get_val(v).toTensor();

  api::utils::uvec3 global_size = t.extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  std::stringstream kernel_name;
  kernel_name << "conv2d_prepack_weights";
  apply_dtype_suffix(kernel_name, t);
  api::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name.str());

  const auto original_sizes =
      api::utils::make_ivec4(tref.sizes, /*reverse=*/true);
  const auto padded_sizes = api::utils::make_ivec4(
      {batch_padded, channels_padded, height, width}, /*reverse=*/true);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      shader,
      global_size,
      local_size,
      vref,
      v,
      {t.gpu_sizes_ubo(),
       graph.create_params_buffer(original_sizes),
       graph.create_params_buffer(padded_sizes)}));

  return v;
}

void check_conv2d_args(const vTensor& in, const vTensor& out) {
  VK_CHECK_COND(
      check_memory_layout_is(in, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED));
  VK_CHECK_COND(check_memory_layout_is(
      out, api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED));
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
  vTensor& t_in = graph.get_val(arg_in).toTensor();
  vTensor& t_out = graph.get_val(out).toTensor();

  check_conv2d_args(t_in, t_out);

  if (t_in.sizes().at(0) > 1) {
    VK_THROW(
        "aten.convolution.default: input batch size > 1 is not supported yet!");
  }

  ValueRef arg_weight = prepack_weights(graph, weight);
  ValueRef arg_bias = prepack_biases(graph, bias);

  api::utils::uvec3 global_size = t_out.virtual_extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  const auto weight_sizes = graph.get_val(weight).toTensorRef().sizes;
  const int64_t k_height = weight_sizes.at(2);
  const int64_t k_width = weight_sizes.at(3);
  const auto kernel_size = api::utils::make_ivec2({k_width, k_height});

  const auto stride_vec = reverse(graph, stride);
  const auto padding_vec = reverse(graph, padding);
  const auto dilation_vec = reverse(graph, dilation);

  KernelParams kernel_params{
      kernel_size,
      stride_vec,
      padding_vec,
      dilation_vec,
  };

  const int64_t d_height = dilation_vec.data[1];
  const int64_t d_width = dilation_vec.data[0];
  const int64_t p_height = padding_vec.data[1];
  const int64_t p_width = padding_vec.data[0];

  if ((p_width > 0 && k_width > 1 && d_width > 1) ||
      (p_height > 0 && k_height > 1 && d_height > 1)) {
    VK_THROW(
        "aten.convolution.default: padding > 0 while dilation, kernel_size > 1 is not supported yet!");
  }

  const auto overlay_region = api::utils::make_ivec2({
      k_width + (k_width - 1) * (d_width - 1),
      k_height + (k_height - 1) * (d_height - 1),
  });
  const int32_t in_group_size = api::utils::safe_downcast<int32_t>(
      api::utils::align_up(weight_sizes.at(1), INT64_C(4)));

  Conv2dParams extra_params{
      overlay_region,
      in_group_size,
  };

  std::stringstream kernel_name;
  kernel_name << "conv2d";
  apply_dtype_suffix(kernel_name, t_out);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name.str()),
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
