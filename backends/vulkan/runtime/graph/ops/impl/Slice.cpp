/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/Logging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

inline int64_t normalize_idx(
    const int64_t index,
    const int64_t max,
    const int64_t default_value) {
  // INT64_MAX is passed when value is unspecified
  if (index == INT64_MAX) {
    return default_value;
  }
  if (index == default_value) {
    return index;
  }
  return normalize(index, max);
}

void add_slice_tensor_out_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef dim_ref,
    ValueRef opt_start_ref,
    ValueRef opt_end_ref,
    ValueRef step_ref,
    ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  VK_CHECK_COND(check_memory_layout_is(*t_in, utils::kChannelsPacked));
  VK_CHECK_COND(check_memory_layout_is(*t_out, utils::kChannelsPacked));

  // Need normalize the dim
  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);

  VK_CHECK_COND(
      -t_in->dim() <= dim && dim < t_in->dim(),
      "dim must be in range of [-self.dim(), self.dim()), but current dim's value is ",
      dim,
      " and self.dim() = ",
      t_in->dim());

  dim = normalize(dim, t_in->dim());

  DimIndex dim_index = normalize_to_dim_index(*t_in, dim);

  std::optional<int64_t> opt_start =
      graph.extract_optional_scalar<int64_t>(opt_start_ref);
  std::optional<int64_t> opt_end =
      graph.extract_optional_scalar<int64_t>(opt_end_ref);
  int64_t step = graph.extract_scalar<int64_t>(step_ref);

  const auto in_sizes = t_in->sizes();
  const auto out_sizes = t_out->sizes();

  int64_t start = opt_start.value_or(0);
  int64_t end = opt_end.value_or(in_sizes[dim]);

  start = normalize_idx(start, in_sizes[dim], 0);
  end = normalize_idx(end, in_sizes[dim], in_sizes[dim]);

  if (dim_index == kChannel4D) {
    // slice by channel
    std::string kernel_name = "slice_channel";
    kernel_name.reserve(kShaderNameReserve);
    add_dtype_suffix(kernel_name, *t_out);

    const struct Block final {
      int offset;
      int step;
    } params{
        static_cast<int32_t>(start),
        static_cast<int32_t>(step),
    };

    graph.execute_nodes().emplace_back(new ExecuteNode(
        graph,
        VK_KERNEL_FROM_STR(kernel_name),
        graph.create_global_wg_size(out),
        graph.create_local_wg_size(out),
        {{out, vkapi::MemoryAccessType::WRITE},
         {in, vkapi::MemoryAccessType::READ}},
        {t_out->sizes_ubo(),
         t_in->sizes_ubo(),
         graph.create_params_buffer(params)}));

  } else {
    // GPU's coordinate is in x, y, z
    int64_t gpu_dim = -1;
    int64_t stride = 1;
    if (dim_index == kWidth4D) {
      gpu_dim = 0; // width: x dimension in gpu
      VK_CHECK_COND(out_sizes[dim] == (1 + (end - start - 1) / step));
    } else if (dim_index == kHeight4D) {
      gpu_dim = 1; // height: y dimension
      VK_CHECK_COND(out_sizes[dim] == (1 + (end - start - 1) / step));
    } else if (dim_index == kBatch4D) {
      gpu_dim = 2; // batch: z dimension

      // Due to channel packing, each batch value is span over stride planes
      int64_t n_channels = dim_at(in_sizes, kChannel4D);
      stride = utils::div_up_4(n_channels);
    } else {
      VK_THROW("Unexpected ncwh_dim!");
    }

    std::string kernel_name = "slice_batch_height_width";
    kernel_name.reserve(kShaderNameReserve);
    add_dtype_suffix(kernel_name, *t_out);

    utils::uvec3 global_size = t_out->image_extents();
    utils::uvec3 local_size = adaptive_work_group_size(global_size);

    const struct Block final {
      int dim;
      int offset;
      int step;
      int stride;
    } params{
        static_cast<int32_t>(gpu_dim),
        static_cast<int32_t>(start),
        static_cast<int32_t>(step),
        static_cast<int32_t>(stride),
    };

    graph.execute_nodes().emplace_back(new ExecuteNode(
        graph,
        VK_KERNEL_FROM_STR(kernel_name),
        global_size,
        local_size,
        {{out, vkapi::MemoryAccessType::WRITE},
         {in, vkapi::MemoryAccessType::READ}},
        {t_out->sizes_ubo(), graph.create_params_buffer(params)}));
  }
}

void slice_tensor_out(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_slice_tensor_out_node(
      graph,
      args[0],
      args[1], // dim
      args[2], // optional start
      args[3], // optional end
      args[4], // step
      args[5]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.slice_copy.Tensor, slice_tensor_out);
  VK_REGISTER_OP(aten.slice.Tensor, slice_tensor_out);
}

} // namespace vkcompute
