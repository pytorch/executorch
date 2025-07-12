/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Transfer.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_select_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  ValueRef out = args.at(0).refs.at(0);
  ValueRef in = args.at(1).refs.at(0);
  int64_t dim = graph->extract_scalar<int64_t>(extra_args.at(0));

  int64_t in_ndim = graph->dim_of(in);

  if (dim < 0) {
    dim += in_ndim;
  }

  std::vector<int64_t> new_out_sizes;
  for (int64_t i = 0; i < in_ndim; ++i) {
    if (i != dim) {
      new_out_sizes.push_back(graph->size_at<int64_t>(i, in));
    }
  }

  graph->virtual_resize(out, new_out_sizes);
}

void check_select_args(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef dim_ref,
    const ValueRef index_ref,
    const ValueRef out) {
  int64_t dim = graph.extract_scalar<int64_t>(dim_ref);
  int64_t index = graph.extract_optional_scalar<int64_t>(index_ref, 0);
  int64_t in_ndim = graph.dim_of(in);

  if (dim < 0) {
    dim += in_ndim;
  }

  VK_CHECK_COND(
      dim >= 0 && dim < in_ndim,
      "Dimension out of range (expected to be in range of [",
      -in_ndim,
      ", ",
      in_ndim - 1,
      "], but got ",
      dim,
      ")");

  const int64_t in_size_at_dim = graph.size_at<int64_t>(dim, in);

  if (index < 0) {
    index += in_size_at_dim;
  }

  VK_CHECK_COND(
      index >= 0 && index < in_size_at_dim,
      "select(): index ",
      index,
      " out of range for tensor of size ",
      in_size_at_dim,
      " at dimension ",
      dim);

  // Check that output tensor has correct dimensions
  int64_t out_dim = graph.dim_of(out);
  VK_CHECK_COND(
      out_dim == in_ndim - 1,
      "Output tensor dimension mismatch (expected ",
      in_size_at_dim - 1,
      ", but got ",
      out_dim,
      ")");

  // Check that output tensor has correct sizes
  int64_t out_idx = 0;
  for (int64_t i = 0; i < in_size_at_dim; ++i) {
    if (i != dim) {
      VK_CHECK_COND(
          graph.size_at<int64_t>(out_idx, out) == graph.size_at<int64_t>(i, in),
          "Output size mismatch at dimension ",
          out_idx,
          " (expected ",
          graph.size_at<int16_t>(i, in),
          ", but got ",
          graph.size_at<int64_t>(out_idx, out),
          ")");
      out_idx++;
    }
  }
}

/**
 * Adds a select operation node to the compute graph.
 *
 * The select operator extracts a slice from a tensor along a specified
 * dimension at a given index. It effectively reduces the dimensionality of the
 * input tensor by one, by selecting a single slice at the specified index along
 * the given dimension. For example, if input is a 3D tensor with shape [2,3,4]
 * and we select dimension 1, index 2, the output will be a 2D tensor with shape
 * [2,4].
 */
void add_select_copy_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef dim_ref,
    const ValueRef index_ref,
    const ValueRef out) {
  check_select_args(graph, in, dim_ref, index_ref, out);

  add_transfer_copy_node(
      graph,
      TransferType::SELECT,
      in,
      dim_ref,
      index_ref,
      kDummyValueRef,
      kDummyValueRef,
      out,
      {dim_ref, index_ref},
      resize_select_node);
}

void select_int(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_select_copy_node(graph, args[0], args[1], args[2], args[3]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.select.int, select_int);
  VK_REGISTER_OP(aten.select_copy.int, select_int);
}

} // namespace vkcompute
