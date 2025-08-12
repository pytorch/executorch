/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Slice.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Transfer.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
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

void resize_slice_copy_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  ValueRef out_ref = args.at(0).refs.at(0);
  ValueRef in_ref = args.at(1).refs.at(0);

  int64_t dim = graph->extract_scalar<int64_t>(extra_args.at(0));
  std::optional<int64_t> opt_start =
      graph->extract_optional_scalar<int64_t>(extra_args.at(1));
  std::optional<int64_t> opt_end =
      graph->extract_optional_scalar<int64_t>(extra_args.at(2));
  int64_t step = graph->extract_scalar<int64_t>(extra_args.at(3));

  // Normalize dim
  if (dim < 0) {
    dim += graph->dim_of(in_ref);
  }

  const std::vector<int64_t> in_sizes = graph->sizes_of(in_ref);
  int64_t dim_size = in_sizes.at(dim);

  int64_t start = opt_start.value_or(0);
  int64_t end = opt_end.value_or(dim_size);

  // Normalize start and end indices
  start = normalize_idx(start, dim_size, 0);
  end = normalize_idx(end, dim_size, dim_size);

  // Calculate output size
  std::vector<int64_t> new_out_sizes = in_sizes;
  new_out_sizes.at(dim) = (end - start + step - 1) / step; // Ceiling division

  graph->virtual_resize(out_ref, new_out_sizes);
}

/**
 * Adds a slice_copy operation node to the compute graph.
 *
 * The slice operator extracts a portion of a tensor along a specified
 * dimension. It creates a new tensor that contains a subset of the input
 * tensor's data, defined by start, end, and step parameters along the given
 * dimension.
 *
 * For example, if input is a tensor with shape [4,5,6] and we slice along
 * dimension 1 with start=1, end=4, step=2, the output will have shape [4,2,6],
 * containing elements from the input at positions 1 and 3 along dimension 1.
 */
void add_slice_copy_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef dim_ref,
    ValueRef opt_start_ref,
    ValueRef opt_end_ref,
    ValueRef step_ref,
    ValueRef out) {
  add_transfer_copy_node(
      graph,
      TransferType::SLICE,
      in,
      dim_ref,
      opt_start_ref,
      opt_end_ref,
      step_ref,
      out,
      {dim_ref, opt_start_ref, opt_end_ref, step_ref},
      resize_slice_copy_node);
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
  ValueRef out_ref = extra_args.at(0);

  std::vector<int64_t> new_out_sizes = get_slice_sizes(
      *graph,
      extra_args.at(1), // input
      extra_args.at(2), // dim
      extra_args.at(3), // optional start
      extra_args.at(4)); // optional end

  graph->virtual_resize(out_ref, new_out_sizes);
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

  graph.virtual_resize(out_ref, new_out_sizes);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      resize_slice_view_node,
      {out_ref, in_ref, dim_ref, opt_start_ref, opt_end_ref, opt_step_ref}));
}

void slice_copy(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_slice_copy_node(
      graph,
      args.at(0),
      args.at(1), // dim
      args.at(2), // optional start
      args.at(3), // optional end
      args.at(4), // step
      args.at(5));
}

void slice(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef in = args.at(0);
  ValueRef out = args.at(5);

  // Special case if out is a view of in
  if (graph.val_is_view_of(out, in)) {
    add_slice_view_node(
        graph,
        in,
        args.at(1), // dim
        args.at(2), // optional start
        args.at(3), // optional end
        args.at(4), // step
        out);
    return;
  }

  add_slice_copy_node(
      graph,
      in,
      args.at(1), // dim
      args.at(2), // optional start
      args.at(3), // optional end
      args.at(4), // step
      out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.slice_copy.Tensor, slice_copy);
  VK_REGISTER_OP(aten.slice.Tensor, slice);
}

} // namespace vkcompute
