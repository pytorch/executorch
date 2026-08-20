/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

int64_t normalize_unfold_dim(const int64_t dim, const int64_t ndim) {
  const int64_t normalized_dim = dim < 0 ? dim + ndim : dim;
  VK_CHECK_COND(
      normalized_dim >= 0 && normalized_dim < ndim,
      "unfold_copy: dimension must be in range [-input.dim(), input.dim())");
  return normalized_dim;
}

std::vector<int64_t> get_unfold_sizes(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef dim_ref,
    const ValueRef size_ref,
    const ValueRef step_ref) {
  const std::vector<int64_t> input_sizes = graph.sizes_of(input);
  VK_CHECK_COND(!input_sizes.empty(), "unfold_copy: input must have rank >= 1");

  const int64_t dim = normalize_unfold_dim(
      graph.extract_scalar<int64_t>(dim_ref), input_sizes.size());
  const int64_t size = graph.extract_scalar<int64_t>(size_ref);
  const int64_t step = graph.extract_scalar<int64_t>(step_ref);

  VK_CHECK_COND(size > 0, "unfold_copy: size must be greater than zero");
  VK_CHECK_COND(step > 0, "unfold_copy: step must be greater than zero");
  VK_CHECK_COND(
      size <= input_sizes.at(dim),
      "unfold_copy: size must not exceed selected input dimension");

  std::vector<int64_t> output_sizes = input_sizes;
  output_sizes.at(dim) = (input_sizes.at(dim) - size) / step + 1;
  output_sizes.push_back(size);
  return output_sizes;
}

void resize_unfold_copy_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef input = args.at(1).refs.at(0);
  graph->virtual_resize(
      output,
      get_unfold_sizes(
          *graph,
          input,
          resize_args.at(0),
          resize_args.at(1),
          resize_args.at(2)));
}

void add_unfold_copy_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef dim_ref,
    const ValueRef size_ref,
    const ValueRef step_ref,
    const ValueRef output) {
  const int64_t input_ndim = graph.dim_of(input);
  const int64_t dim =
      normalize_unfold_dim(graph.extract_scalar<int64_t>(dim_ref), input_ndim);
  const int64_t dim_whcn = input_ndim - dim - 1;
  const int64_t step = graph.extract_scalar<int64_t>(step_ref);

  get_unfold_sizes(graph, input, dim_ref, size_ref, step_ref);

  std::string kernel_name = "unfold_copy_buffer";
  add_dtype_suffix(kernel_name, graph.dtype_of(output));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      {{output, vkapi::kWrite}, {input, vkapi::kRead}},
      {graph.meta_ubo(output), graph.meta_ubo(input)},
      {},
      {
          utils::safe_downcast<int32_t>(dim_whcn),
          utils::safe_downcast<int32_t>(step),
      },
      {dim_ref, size_ref, step_ref},
      resize_unfold_copy_node));
}

void unfold_copy(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  add_unfold_copy_node(
      graph, args.at(0), args.at(1), args.at(2), args.at(3), args.at(4));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.unfold_copy.default, unfold_copy);
}

} // namespace vkcompute
