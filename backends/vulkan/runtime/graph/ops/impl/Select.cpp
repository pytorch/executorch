/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/Logging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_args(
    const api::vTensor& t_in,
    int64_t dim,
    int64_t index,
    const api::vTensor& t_out) {
  VK_CHECK_COND(check_memory_layout_is(t_in, utils::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(t_out, utils::kChannelsPacked));

  const int64_t in_dim = t_in.dim();
  VK_CHECK_COND(
      in_dim == 3 || in_dim == 4,
      "Vulkan select only support 3d or 4d tensors!");

  const int64_t in_size = t_in.size(dim);

  if (index < -in_size || index >= in_size) {
    VK_CHECK_COND(
        false,
        "select(): index ",
        index,
        " t_outof range for tensor of size ",
        in_size,
        " at dimension ",
        dim);
  }
}

void add_select_int_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef dim_ref,
    const ValueRef index_ref,
    const ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);
  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);
  int64_t index = graph.extract_scalar<int64_t>(index_ref);

  check_args(*t_in, dim, index, *t_out);

  const int64_t in_size = t_in->size(dim);

  if (index < 0) {
    index += in_size;
  }

  std::string kernel_name;

  // for 3d tensors, these values are not used by the shader.
  int32_t num_texel_per_batch = 1;
  int32_t num_batches = 1;

  int64_t in_dim = t_in->dim();
  if (in_dim == 3) {
    if (dim == 0) {
      kernel_name = "select_channel_3d";
    } else if (dim == 1) {
      kernel_name = "select_height_3d";
    } else if (dim == 2) {
      kernel_name = "select_width_3d";
    } else {
      VK_CHECK_COND(
          false, "Unexpected dim value=", dim, "for the input 3d tensor");
    }
  } else { // self.dim() == 4
    num_texel_per_batch =
        static_cast<int32_t>(std::ceil(static_cast<float>(t_in->size(1)) / 4));
    num_batches = t_in->size(0);
    if (dim == 0) {
      kernel_name = "select_batch_4d";
    } else if (dim == 1) {
      kernel_name = "select_channel_4d";
    } else if (dim == 2) {
      kernel_name = "select_height_4d";
    } else if (dim == 3) {
      kernel_name = "select_width_4d";
    } else {
      VK_CHECK_COND(
          false, "Unexpected dim value=", dim, "for the input 4d tensor");
    }
  }

  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  // TODO: add resizing to support dynamic shapes.
  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {in, vkapi::MemoryAccessType::READ}},
      // Parameter buffers
      {t_out->texture_limits_ubo(),
       t_out->sizes_ubo(),
       // TODO: num_batches and num_texel_per_batch are provided by
       // t_out->sizes. Can change the following to reduce params
       // created.
       graph.create_params_buffer(
           utils::make_ivec4({index, num_batches, num_texel_per_batch, 0}))},
      // Specialization Constants
      {}));
}

void select_int(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_select_int_node(graph, args[0], args[1], args[2], args[3]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.select.int, select_int);
  VK_REGISTER_OP(aten.select_copy.int, select_int);
}

} // namespace vkcompute
