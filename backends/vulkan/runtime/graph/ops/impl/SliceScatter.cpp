/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

namespace {

// INT64_MAX is what the edge dialect passes for an unspecified start/end.
int64_t normalize_slice_idx(
    const int64_t index,
    const int64_t dim_size,
    const int64_t default_value) {
  if (index == INT64_MAX) {
    return default_value;
  }
  if (index < 0) {
    return std::max<int64_t>(index + dim_size, 0);
  }
  return std::min<int64_t>(index, dim_size);
}

} // namespace

void resize_slice_scatter_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);
  // slice_scatter never changes the shape: the output is self with a window
  // overwritten.
  graph->virtual_resize(out, graph->sizes_of(self));
}

void add_slice_scatter_node(
    ComputeGraph& graph,
    const ValueRef self,
    const ValueRef src,
    const ValueRef dim_ref,
    const ValueRef start_ref,
    const ValueRef end_ref,
    const ValueRef step_ref,
    const ValueRef out) {
  const int64_t ndim = graph.dim_of(self);
  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);
  if (dim < 0) {
    dim += ndim;
  }
  VK_CHECK_COND(
      dim >= 0 && dim < ndim, "Vulkan slice_scatter got an out-of-range dim");

  const std::vector<int64_t> self_sizes = graph.sizes_of(self);
  const int64_t dim_size = self_sizes.at(dim);

  const int64_t step = graph.extract_scalar<int64_t>(step_ref);
  VK_CHECK_COND(step > 0, "Vulkan slice_scatter requires step > 0");

  const std::optional<int64_t> opt_start =
      graph.extract_optional_scalar<int64_t>(start_ref);
  const std::optional<int64_t> opt_end =
      graph.extract_optional_scalar<int64_t>(end_ref);
  const int64_t start = normalize_slice_idx(opt_start.value_or(0), dim_size, 0);
  const int64_t end =
      normalize_slice_idx(opt_end.value_or(dim_size), dim_size, dim_size);

  // The shader decides src-vs-self per output element from (start, end, step)
  // alone, so it never reads outside src as long as src is exactly as long as
  // the strided window. aten guarantees this; assert rather than clamp, since
  // a mismatch would otherwise read out of bounds.
  const int64_t window = start < end ? (end - start + step - 1) / step : 0;
  VK_CHECK_COND(
      graph.sizes_of(src).at(dim) == window,
      "Vulkan slice_scatter: src size along dim does not match the slice");

  // The shader hard-codes texture indexing for all three tensors.
  VK_CHECK_COND(
      graph.is_standard_channels_packed_texture_tensor(self) &&
          graph.is_standard_channels_packed_texture_tensor(src) &&
          graph.is_standard_channels_packed_texture_tensor(out),
      "Vulkan slice_scatter requires channels-packed texture tensors");
  VK_CHECK_COND(
      graph.dtype_of(self) == graph.dtype_of(src) &&
          graph.dtype_of(self) == graph.dtype_of(out),
      "Vulkan slice_scatter requires self, src and out to share dtype");

  // `selected_dim` is consumed in WHCN order by the shader's TensorIndex4D.
  const int32_t dim_whcn =
      static_cast<int32_t>(nchw_dim_to_whcn_dim(dim, ndim));

  const struct SliceScatterParams final {
    int32_t selected_dim;
    int32_t start;
    int32_t end;
    int32_t step;
  } params{
      dim_whcn,
      static_cast<int32_t>(start),
      static_cast<int32_t>(end),
      static_cast<int32_t>(step)};

  std::string kernel_name("slice_scatter_texture3d");
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_gwg,
      default_pick_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{self, src}, vkapi::kRead}},
      // Shader params buffers
      {graph.meta_ubo(out), graph.meta_ubo(self), graph.meta_ubo(src)},
      // Push Constants
      {PushConstantDataInfo(&params, sizeof(params))},
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(self),
       graph.hashed_layout_of(src)},
      // Resize Args
      {},
      // Resizing Logic
      resize_slice_scatter_node));
}

void slice_scatter(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // Schema: slice_scatter(self, src, dim, start, end, step) -> Tensor
  return add_slice_scatter_node(
      graph, args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.slice_scatter.default, slice_scatter);
}

} // namespace vkcompute
