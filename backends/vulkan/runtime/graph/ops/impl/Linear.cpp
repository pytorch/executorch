/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/MatMul.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_addmm_args(
    ComputeGraph& graph,
    const ValueRef self,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef beta,
    const ValueRef alpha,
    const ValueRef out) {
  (void)alpha;
  (void)beta;

  std::vector<int64_t> self_sizes = graph.sizes_of(self);
  std::vector<int64_t> mat1_sizes = graph.sizes_of(mat1);
  std::vector<int64_t> mat2_sizes = graph.sizes_of(mat2_data);

  VK_CHECK_COND(mat1_sizes.size() == 2 || mat1_sizes.size() == 3);
  VK_CHECK_COND(mat1_sizes.size() == mat2_sizes.size());

  VK_CHECK_COND(graph.memory_layout_of(mat1) == graph.memory_layout_of(out));

  VK_CHECK_COND(utils::val_at(-1, mat1_sizes) == utils::val_at(-2, mat2_sizes));

  if (utils::val_at(-1, self_sizes) != 1) {
    VK_CHECK_COND(
        utils::val_at(-1, self_sizes) == utils::val_at(-1, mat2_sizes));
  }
  if (utils::val_at(-2, self_sizes) != 1) {
    VK_CHECK_COND(
        utils::val_at(-2, self_sizes) == utils::val_at(-2, mat1_sizes));
  }
}

void resize_addmm_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr mat1 = graph->get_tensor(args[1].refs[0]);
  vTensorPtr mat2 = graph->get_tensor(args[1].refs[1]);
  vTensorPtr self = graph->get_tensor(args[1].refs[2]);

  bool mat2_is_transposed = graph->get_bool(extra_args[0]);

  const int out_cols = utils::val_at(-2, mat1->sizes());
  const int out_rows = mat2_is_transposed ? utils::val_at(-2, mat2->sizes())
                                          : utils::val_at(-1, mat2->sizes());

  std::vector<int64_t> new_out_sizes(3);
  if (mat1->sizes().size() == 2) {
    new_out_sizes.resize(2);
    new_out_sizes.at(0) = out_cols;
    new_out_sizes.at(1) = out_rows;
  } else {
    new_out_sizes.at(0) = mat1->sizes().at(0);
    new_out_sizes.at(1) = out_cols;
    new_out_sizes.at(2) = out_rows;
  }

  out->virtual_resize(new_out_sizes);
}

struct Params final {
  float alpha;
  float beta;
};

void add_addmm_naive_node(
    ComputeGraph& graph,
    const ValueRef self_data,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef beta,
    const ValueRef alpha,
    const ValueRef out,
    const Params& params,
    const ValueRef mat2_is_transposed) {
  ValueRef self = prepack_if_tensor_ref(graph, self_data, utils::kWidthPacked);
  ValueRef mat2 = prepack_if_tensor_ref(graph, mat2_data, utils::kHeightPacked);

  std::string kernel_name =
      graph.get_bool(mat2_is_transposed) ? "linear_naive" : "addmm_naive";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  utils::uvec3 global_wg_size = graph.logical_extents_of(out);
  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {{mat1, mat2, self}, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {
          graph.sizes_ubo(out),
          graph.logical_limits_ubo(out),
          graph.axis_map_ubo(out),
          graph.sizes_ubo(mat1),
          graph.axis_map_ubo(mat1),
          graph.sizes_ubo(mat2),
          graph.axis_map_ubo(mat2),
          graph.sizes_ubo(self),
          graph.axis_map_ubo(self),
          graph.create_params_buffer(params),
      },
      // Specialization Constants
      {graph.packed_dim_whcn_idx_of(out),
       graph.packed_dim_whcn_idx_of(mat1),
       graph.packed_dim_whcn_idx_of(mat2),
       graph.packed_dim_whcn_idx_of(self)},
      // Resizing Logic
      resize_addmm_node,
      {mat2_is_transposed}));
}

void add_addmm_optimized_node(
    ComputeGraph& graph,
    const ValueRef self_data,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef beta,
    const ValueRef alpha,
    const ValueRef out,
    const Params& params,
    const ValueRef mat2_is_transposed) {
  ValueRef self =
      prepack_if_tensor_ref(graph, self_data, utils::kChannelsPacked);
  ValueRef mat2 = prepack_if_tensor_ref(graph, mat2_data, utils::kHeightPacked);

  // Ensure mat1 is width packed
  ValueRef mat1_W_packed = graph.add_tensor_like(mat1, utils::kWidthPacked);
  auto viewFn = VK_GET_OP_FN("aten.view_copy.default");
  viewFn(graph, {mat1, graph.add_none(), mat1_W_packed});

  const bool mat2_is_transposed_val = graph.get_bool(mat2_is_transposed);

  // Ensure mat2 is height packed
  ValueRef mat2_packed = mat2;
  const utils::GPUMemoryLayout mat2_layout =
      mat2_is_transposed_val ? utils::kWidthPacked : utils::kHeightPacked;
  if (graph.memory_layout_of(mat2) != mat2_layout) {
    mat2_packed = graph.add_tensor_like(mat2, mat2_layout);
    viewFn(graph, {mat2, graph.add_none(), mat2_packed});
  }

  std::string kernel_name = graph.get_bool(mat2_is_transposed)
      ? "linear_optimized"
      : "addmm_optimized";

  std::vector<int64_t> mat1_sizes = graph.sizes_of(mat1_W_packed);
  int mat1_dims = mat1_sizes.size();
  if (mat1_dims == 3) {
    kernel_name = "batch_" + kernel_name;
  }
  if (mat1_sizes.at(mat1_dims - 2) < 8) {
    kernel_name += "_tile_row_2";
  } else {
    kernel_name += "_tile_row_4";
  }

  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  utils::uvec3 global_size;

  // Each thread computes a W=(2/4) x H=4 x C=(1/4) output tile. Therefore, the
  // total number of threads is W/(2 or 4) x H/4 x C/1. Since the out tensor is
  // channels packed, C does not need to be divided by 4. The "identity" of each
  // thread is the (x, y, z) coordinate of the output tile it is computing, and
  // this identity can be used to compute the tensor index of the top left
  // element in the tile, which will be [W=x*(2 or 4), H=y*4, C=z*(1 or 4), N=0]
  if (mat1_sizes.at(mat1_dims - 2) < 8) {
    // Use `logical_extents` instead of `image_extents` because the workgroup
    // axes need to correspond to tensor dimensions.
    global_size = utils::divup_vec(graph.logical_extents_of(out), {4, 2, 1});
  } else {
    global_size = utils::divup_vec(graph.logical_extents_of(out), {4, 4, 1});
  }
  utils::uvec3 local_size = adaptive_work_group_size(global_size);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {{mat1_W_packed, mat2_packed, self}, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {
          graph.sizes_ubo(out),
          graph.axis_map_ubo(out),
          graph.sizes_ubo(mat1_W_packed),
          graph.axis_map_ubo(mat1_W_packed),
          graph.sizes_ubo(mat2_packed),
          graph.axis_map_ubo(mat2_packed),
          graph.sizes_ubo(self),
          graph.axis_map_ubo(self),
          graph.create_params_buffer(params),
      },
      // Specialization Constants
      {graph.packed_dim_whcn_idx_of(out)},
      // Resizing Logic
      resize_addmm_node,
      {mat2_is_transposed}));
}

void add_addmm_node(
    ComputeGraph& graph,
    const ValueRef self,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef beta,
    const ValueRef alpha,
    const ValueRef out,
    const ValueRef mat2_is_transposed) {
  float alpha_val = 1.0f;
  float beta_val = 1.0f;

  if (alpha != kDummyValueRef) {
    alpha_val = graph.extract_scalar<float>(alpha);
  }
  if (beta != kDummyValueRef) {
    beta_val = graph.extract_scalar<float>(beta);
  }

  Params params = {alpha_val, beta_val};
  if (graph.memory_layout_of(mat1) == utils::kChannelsPacked) {
    add_addmm_optimized_node(
        graph, self, mat1, mat2, beta, alpha, out, params, mat2_is_transposed);
  } else if (graph.memory_layout_of(mat1) == utils::kWidthPacked) {
    add_addmm_naive_node(
        graph, self, mat1, mat2, beta, alpha, out, params, mat2_is_transposed);
  } else {
    VK_THROW("Input should be channel packed or width packed.");
  }
}

void addmm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  check_addmm_args(graph, args[0], args[1], args[2], args[3], args[4], args[5]);
  ValueRef mat2_is_transposed = graph.add_scalar(false);
  return add_addmm_node(
      graph,
      args[0],
      args[1],
      args[2],
      args[3],
      args[4],
      args[5],
      mat2_is_transposed);
}

void linear(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef input = args.at(0);
  ValueRef weight_data = args.at(1);
  ValueRef bias = args.at(2);
  ValueRef out = args.at(3);
  ValueRef weight =
      prepack_if_tensor_ref(graph, weight_data, utils::kWidthPacked);
  ValueRef mat2_is_transposed = graph.add_scalar(true);
  if (graph.val_is_none(bias)) {
    return add_matmul_node(graph, input, weight, out, mat2_is_transposed);
  } else {
    return add_addmm_node(
        graph,
        bias,
        input,
        weight,
        kDummyValueRef,
        kDummyValueRef,
        out,
        mat2_is_transposed);
  }
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.addmm.default, addmm);
  VK_REGISTER_OP(aten.linear.default, linear);
}

} // namespace vkcompute
