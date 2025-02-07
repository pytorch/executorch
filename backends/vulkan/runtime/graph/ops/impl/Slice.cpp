/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/Logging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Slice.h>

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

void add_slice_tensor_copy_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef dim_ref,
    ValueRef opt_start_ref,
    ValueRef opt_end_ref,
    ValueRef step_ref,
    ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  VK_CHECK_COND(check_packed_dim_is(*t_in, WHCN::kChannelsDim));
  VK_CHECK_COND(check_packed_dim_is(*t_out, WHCN::kChannelsDim));

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

    graph.execute_nodes().emplace_back(new DispatchNode(
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

    utils::uvec3 global_size = t_out->logical_limits();
    utils::uvec3 local_size = graph.create_local_wg_size(global_size);

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

    graph.execute_nodes().emplace_back(new DispatchNode(
        graph,
        VK_KERNEL_FROM_STR(kernel_name),
        global_size,
        local_size,
        {{out, vkapi::MemoryAccessType::WRITE},
         {in, vkapi::MemoryAccessType::READ}},
        {t_out->sizes_ubo(), graph.create_params_buffer(params)}));
  }
}

std::vector<int64_t> get_slice_sizes(
    ComputeGraph& graph,
    ValueRef in_ref,
    ValueRef dim_ref,
    ValueRef opt_start_ref,
    ValueRef opt_end_ref) {
  const int64_t dim = graph.extract_scalar<int64_t>(dim_ref);
  std::optional<int64_t> opt_start =
      graph.extract_optional_scalar<int64_t>(opt_start_ref);
  std::optional<int64_t> opt_end =
      graph.extract_optional_scalar<int64_t>(opt_end_ref);

  int64_t dim_size = graph.size_at<int64_t>(dim, in_ref);
  int64_t start = opt_start.value_or(0);
  int64_t end = opt_end.value_or(dim_size);

  start = normalize_idx(start, dim_size, 0);
  end = normalize_idx(end, dim_size, dim_size);

  std::vector<int64_t> new_out_sizes = graph.sizes_of(in_ref);
  new_out_sizes.at(dim) = end - start;

  return new_out_sizes;
}

void resize_slice_view_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)args;
  vTensorPtr out = graph->get_tensor(extra_args[0]);

  std::vector<int64_t> new_out_sizes = get_slice_sizes(
      *graph,
      extra_args[1], // input
      extra_args[2], // dim
      extra_args[3], // optional start
      extra_args[4]); // optional end

  out->virtual_resize(new_out_sizes);
}

void check_slice_view_args(
    ComputeGraph& graph,
    ValueRef in_ref,
    ValueRef dim_ref,
    ValueRef opt_start_ref,
    ValueRef opt_end_ref,
    ValueRef opt_step_ref,
    ValueRef out_ref) {
  VK_CHECK_COND(
      graph.val_is_view_of(out_ref, in_ref),
      "output must be a view of the input");

  const int64_t dim = graph.extract_scalar<int64_t>(dim_ref);
  const int64_t dim_size = graph.size_at<int64_t>(dim, in_ref);

  int64_t start =
      graph.extract_optional_scalar<int64_t>(opt_start_ref).value_or(0);
  int64_t end = graph.extract_optional_scalar<int64_t>(opt_end_ref).value_or(0);
  int64_t step =
      graph.extract_optional_scalar<int64_t>(opt_step_ref).value_or(1);

  start = normalize_idx(start, dim_size, 0);
  end = normalize_idx(end, dim_size, dim_size);

  // The start idx must be 0; this is to ensure that the start of the slice view
  // does not have any offset with respect to the base buffer storage. If the
  // offset is nonzero, then it will potentially change upon a resize; however
  // the buffer offset of the view tensor will have been "locked in" when the
  // descriptor for its buffer storage is bound to a compute shader. Therefore
  // there is no way to update the offset of the view once it has been bound.
  VK_CHECK_COND(start == 0, "start must be 0 for slice view");
  VK_CHECK_COND(step == 1, "step must be 1 for slice view");

  VK_CHECK_COND(
      end < dim_size, "end must be less than dim size for slice view");

  // We must also check that all earlier dims in the dim order have a size of 1.
  // This ensures that the slice view encompasses a contiguous memory region of
  // the source tensor's memory buffer.
  std::vector<int64_t> in_sizes = graph.sizes_of(in_ref);
  std::vector<int64_t> in_dim_order = graph.dim_order_of(in_ref);
  for (int i = 0; i < in_dim_order.size(); ++i) {
    if (in_dim_order[i] == dim) {
      break;
    }
    VK_CHECK_COND(in_sizes[in_dim_order[i]] == 1);
  }
}

void add_slice_view_node(
    ComputeGraph& graph,
    ValueRef in_ref,
    ValueRef dim_ref,
    ValueRef opt_start_ref,
    ValueRef opt_end_ref,
    ValueRef opt_step_ref,
    ValueRef out_ref) {
  check_slice_view_args(
      graph,
      in_ref,
      dim_ref,
      opt_start_ref,
      opt_end_ref,
      opt_step_ref,
      out_ref);

  std::vector<int64_t> new_out_sizes =
      get_slice_sizes(graph, in_ref, dim_ref, opt_start_ref, opt_end_ref);

  graph.get_tensor(out_ref)->virtual_resize(new_out_sizes);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      resize_slice_view_node,
      {out_ref, in_ref, dim_ref, opt_start_ref, opt_end_ref, opt_step_ref}));
}

void slice_tensor_copy(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_slice_tensor_copy_node(
      graph,
      args[0],
      args[1], // dim
      args[2], // optional start
      args[3], // optional end
      args[4], // step
      args[5]);
}

void slice_tensor(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef in = args[0];
  ValueRef out = args[5];

  // Special case if out is a view of in
  if (graph.val_is_view_of(out, in)) {
    add_slice_view_node(
        graph,
        in,
        args[1], // dim
        args[2], // optional start
        args[3], // optional end
        args[4], // step
        out);
    return;
  }

  add_slice_tensor_copy_node(
      graph,
      in,
      args[1], // dim
      args[2], // optional start
      args[3], // optional end
      args[4], // step
      out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.slice_copy.Tensor, slice_tensor_copy);
  VK_REGISTER_OP(aten.slice.Tensor, slice_tensor);
}

} // namespace vkcompute
