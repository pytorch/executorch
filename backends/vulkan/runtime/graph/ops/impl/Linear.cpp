/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/MatMul.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

// Custom global workgroup size function for addmm_naive_texture
utils::uvec3 addmm_naive_texture_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  return graph->logical_limits_of(out);
}

// Custom global workgroup size function for addmm_naive_buffer
utils::uvec3 addmm_naive_buffer_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  return {
      graph->size_at<uint32_t>(-1, out),
      graph->size_at<uint32_t>(-2, out),
      graph->size_at<uint32_t>(-3, out) * graph->size_at<uint32_t>(-4, out)};
}

// Custom global workgroup size function for addmm_optimized
utils::uvec3 addmm_optimized_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1 = args.at(1).refs.at(0);

  std::vector<int64_t> mat1_sizes = graph->sizes_of(mat1);
  int mat1_dims = mat1_sizes.size();

  utils::uvec3 global_size = graph->logical_limits_of(out);

  if (mat1_sizes.at(mat1_dims - 2) < 8) {
    global_size = utils::divup_vec(global_size, {4, 2, 1});
  } else {
    global_size = utils::divup_vec(global_size, {4, 4, 1});
  }
  return global_size;
}

// Custom local workgroup size function for addmm_optimized
utils::uvec3 addmm_optimized_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  (void)resize_args;
  return adaptive_work_group_size(global_workgroup_size);
}

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

  VK_CHECK_COND(graph.packed_dim_of(mat1) == graph.packed_dim_of(out));

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
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1 = args.at(1).refs.at(0);
  const ValueRef mat2 = args.at(1).refs.at(1);

  const bool mat2_is_transposed = graph->get_bool(extra_args.at(0));

  const std::vector<int64_t> mat1_sizes = graph->sizes_of(mat1);
  const std::vector<int64_t> mat2_sizes = graph->sizes_of(mat2);

  const int out_cols = utils::val_at(-2, mat1_sizes);
  const int out_rows = mat2_is_transposed ? utils::val_at(-2, mat2_sizes)
                                          : utils::val_at(-1, mat2_sizes);

  std::vector<int64_t> new_out_sizes(3);
  if (mat1_sizes.size() == 2) {
    new_out_sizes.resize(2);
    new_out_sizes.at(0) = out_cols;
    new_out_sizes.at(1) = out_rows;
  } else {
    new_out_sizes.at(0) = mat1_sizes.at(0);
    new_out_sizes.at(1) = out_cols;
    new_out_sizes.at(2) = out_rows;
  }

  graph->virtual_resize(out, new_out_sizes);
}

struct Params final {
  float alpha;
  float beta;
};

void add_addmm_naive_texture_node(
    ComputeGraph& graph,
    const ValueRef self_data,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef beta,
    const ValueRef alpha,
    const ValueRef out,
    const Params& params,
    const ValueRef mat2_is_transposed) {
  utils::StorageType stype = graph.storage_type_of(out);
  ValueRef self = prepack_standard(
      graph, self_data, stype, utils::kWidthPacked, /*passthrough = */ true);
  ValueRef mat2 = prepack_standard(
      graph, mat2_data, stype, utils::kHeightPacked, /*passthrough = */ true);

  std::string kernel_name =
      graph.get_bool(mat2_is_transposed) ? "linear_naive" : "addmm_naive";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  utils::uvec3 global_wg_size = graph.logical_limits_of(out);
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      addmm_naive_texture_global_wg_size,
      pick_hw_square_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{mat1, mat2, self}, vkapi::kRead}},
      // Shader params buffers
      {},
      // Push Constants
      {graph.sizes_pc_of(out),
       graph.sizes_pc_of(mat1),
       graph.sizes_pc_of(mat2),
       graph.logical_limits_pc_of(out),
       graph.sizes_pc_of(self),
       PushConstantDataInfo(&params, sizeof(params))},
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(mat1),
       graph.hashed_layout_of(mat2),
       graph.hashed_layout_of(self)},
      // Resize Args
      {mat2_is_transposed},
      // Resizing Logic
      resize_addmm_node));
}

void add_addmm_naive_buffer_node(
    ComputeGraph& graph,
    const ValueRef self_data,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef beta,
    const ValueRef alpha,
    const ValueRef out,
    const Params& params,
    const ValueRef mat2_is_transposed) {
  (void)beta;
  (void)alpha;
  ValueRef mat2 = prepack_standard(
      graph,
      mat2_data,
      graph.storage_type_of(out),
      utils::kHeightPacked,
      /*passthrough = */ true);
  ValueRef self = prepack_standard(
      graph,
      self_data,
      graph.storage_type_of(out),
      utils::kWidthPacked,
      /*passthrough = */ true);

  std::string kernel_name = "addmm_naive_buffer";
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  utils::uvec3 global_size = {
      graph.size_at<uint32_t>(-1, out),
      graph.size_at<uint32_t>(-2, out),
      graph.size_at<uint32_t>(-3, out) * graph.size_at<uint32_t>(-4, out)};

  int mat2_is_transposed_val = (mat2_is_transposed != kDummyValueRef &&
                                graph.get_bool(mat2_is_transposed))
      ? 1
      : 0;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      addmm_naive_buffer_global_wg_size,
      pick_hw_square_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{mat1, mat2, self}, vkapi::kRead}},
      // Shader params buffers
      {
          graph.sizes_ubo(out),
          graph.strides_ubo(out),
          graph.sizes_ubo(mat1),
          graph.strides_ubo(mat1),
          graph.sizes_ubo(mat2),
          graph.strides_ubo(mat2),
          graph.numel_ubo(out),
          graph.create_params_buffer(params),
      },
      // Push Constants
      {},
      // Specialization Constants
      {mat2_is_transposed_val},
      // Resize Args
      {mat2_is_transposed},
      // Resizing Logic
      resize_addmm_node));
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
  utils::StorageType stype = graph.storage_type_of(out);
  ValueRef self = prepack_standard(
      graph, self_data, stype, utils::kChannelsPacked, /*passthrough=*/true);
  ValueRef mat2 = prepack_standard(
      graph, mat2_data, stype, utils::kHeightPacked, /*passthrough=*/true);

  // Ensure mat1 is width packed
  ValueRef mat1_W_packed = graph.add_tensor_like(mat1, utils::kWidthPacked);
  auto viewFn = VK_GET_OP_FN("aten.view_copy.default");
  viewFn(graph, {mat1, graph.add_none(), mat1_W_packed});

  const bool mat2_is_transposed_val = graph.get_bool(mat2_is_transposed);

  // Ensure mat2 is height packed
  ValueRef mat2_packed = mat2;
  const utils::GPUMemoryLayout mat2_layout =
      mat2_is_transposed_val ? utils::kWidthPacked : utils::kHeightPacked;
  if (graph.estimate_memory_layout_of(mat2) != mat2_layout) {
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

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      addmm_optimized_global_wg_size,
      addmm_optimized_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite},
       {{mat1_W_packed, mat2_packed, self}, vkapi::kRead}},
      // Shader params buffers
      {
          graph.sizes_ubo(out),
          graph.sizes_ubo(mat1_W_packed),
          graph.sizes_ubo(mat2_packed),
          graph.sizes_ubo(self),
          graph.create_params_buffer(params),
      },
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(mat1_W_packed),
       graph.hashed_layout_of(mat2_packed),
       graph.hashed_layout_of(self)},
      // Resize Args
      {mat2_is_transposed},
      // Resizing Logic
      resize_addmm_node));
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
  if (graph.is_buffer_storage(out)) {
    add_addmm_naive_buffer_node(
        graph, self, mat1, mat2, beta, alpha, out, params, mat2_is_transposed);
  } else if (graph.packed_dim_of(mat1) == WHCN::kChannelsDim) {
    add_addmm_optimized_node(
        graph, self, mat1, mat2, beta, alpha, out, params, mat2_is_transposed);
  } else if (graph.packed_dim_of(mat1) == WHCN::kWidthDim) {
    add_addmm_naive_texture_node(
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
  ValueRef weight = prepack_standard(
      graph,
      weight_data,
      graph.storage_type_of(out),
      utils::kWidthPacked,
      /*passthrough = */ true);
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
