/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_matmul_args(
    ComputeGraph& graph,
    const ValueRef arg1,
    const ValueRef arg2,
    const ValueRef out) {
  vTensorPtr t_mat1 = graph.get_tensor(arg1);
  vTensorPtr t_mat2 = graph.get_tensor(arg2);
  vTensorPtr t_out = graph.get_tensor(out);

  VK_CHECK_COND(check_ndim_is(*t_mat1, 2) || check_ndim_is(*t_mat1, 3));
  VK_CHECK_COND(check_same_ndim(*t_mat1, *t_mat2));

  VK_CHECK_COND(check_same_memory_layout(*t_mat1, *t_out));

  VK_CHECK_COND(check_same_sizes_at(*t_mat1, -1, *t_mat2, -2));
}

void resize_matmul_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr mat1 = graph->get_tensor(args[1].refs[0]);
  vTensorPtr mat2 = graph->get_tensor(args[1].refs[1]);

  std::vector<int64_t> new_out_sizes(3);
  if (mat1->sizes().size() == 2) {
    new_out_sizes.resize(2);
    new_out_sizes.at(0) = mat1->sizes().at(0);
    new_out_sizes.at(1) = mat2->sizes().at(1);
  } else {
    new_out_sizes.at(0) = mat1->sizes().at(0);
    new_out_sizes.at(1) = mat1->sizes().at(1);
    new_out_sizes.at(2) = mat2->sizes().at(2);
  }

  out->virtual_resize(new_out_sizes);
}

struct AddmmParams final {
  int broadcast_at_width;
  int broadcast_at_height;
  float alpha;
  float beta;
};

// TODO: `add_matmul_node` and `add_addmm_node` has lots of duplicated code.
// We should do refactoring to simplify.
void add_matmul_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef out) {
  ValueRef arg1 = mat1;
  ValueRef arg2 = prepack_if_tensor_ref(graph, mat2, api::kHeightPacked);

  std::vector<int64_t> t_mat1_sizes = graph.get_tensor(arg1)->sizes();
  std::vector<int64_t> t_mat2_sizes = graph.get_tensor(arg2)->sizes();
  std::vector<int64_t> out_sizes = graph.get_tensor(out)->sizes();
  int64_t t_mat1_dim = t_mat1_sizes.size();
  int64_t out_dim = out_sizes.size();

  check_matmul_args(graph, arg1, arg2, out);
  auto viewFn = VK_GET_OP_FN("aten.view_copy.default");

  // optimized mm
  if (graph.memory_layout_of(arg1) == api::kChannelsPacked) {
    ValueRef t_mat1_width_packed =
        graph.add_tensor_like(arg1, api::kWidthPacked);
    viewFn(graph, {arg1, graph.add_none(), t_mat1_width_packed});
    arg1 = t_mat1_width_packed;

    if (graph.memory_layout_of(arg2) != api::kHeightPacked) {
      ValueRef t_mat2_height_packed =
          graph.add_tensor(t_mat2_sizes, api::kFloat, api::kHeightPacked);
      viewFn(graph, {arg2, graph.add_none(), t_mat2_height_packed});
      arg2 = t_mat2_height_packed;
    }

    vTensorPtr t_mat1 = graph.get_tensor(arg1);
    vTensorPtr t_mat2 = graph.get_tensor(arg2);

    VK_CHECK_COND(check_memory_layout_is(*t_mat1, api::kWidthPacked));
    VK_CHECK_COND(check_memory_layout_is(*t_mat2, api::kHeightPacked));

    // Step size is the 2d input's width dimension / 4.
    int32_t step_size =
        api::utils::div_up(t_mat1_sizes.at(t_mat1_dim - 1), INT64_C(4));

    // reminder is used in shader to detect whether the fetched texel is out of
    // boundary
    int32_t reminder = t_mat1_sizes.at(t_mat1_dim - 1) % INT64_C(4);

    int64_t batch_size = 1;
    if (t_mat1_dim == 3) {
      batch_size = t_mat1_sizes.at(0);
    }

    vTensorPtr t_out = graph.get_tensor(out);

    api::utils::uvec3 global_size = {
        static_cast<unsigned int>(
            api::utils::div_up(out_sizes.at(t_mat1_dim - 1), INT64_C(4))),
        static_cast<unsigned int>(
            api::utils::div_up(out_sizes.at(t_mat1_dim - 2), INT64_C(4))),
        static_cast<unsigned int>(
            out_dim == 3 ? api::utils::div_up(out_sizes.at(0), INT64_C(4))
                         : 1)};
    api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

    std::string kernel_name("matmul_optimized");
    kernel_name.reserve(kShaderNameReserve);

    add_dtype_suffix(kernel_name, *t_out);

    graph.execute_nodes().emplace_back(new ExecuteNode(
        graph,
        VK_KERNEL_FROM_STR(kernel_name),
        global_size,
        local_size,
        // Inputs and Outputs
        {{out, api::MemoryAccessType::WRITE},
         {{arg1, arg2}, api::MemoryAccessType::READ}},
        // Shader params buffers
        {
            t_out->texture_limits_ubo(),
            graph.create_params_buffer(step_size),
            graph.create_params_buffer(reminder),
            graph.create_params_buffer(batch_size),
        },
        // Specialization Constants
        {},
        // Resizing Logic
        resize_matmul_node));
  } else if (graph.memory_layout_of(arg1) == api::kWidthPacked) {
    // native mm
    if (graph.memory_layout_of(arg2) != api::kHeightPacked) {
      ValueRef t_mat2_height_packed =
          graph.add_tensor(t_mat2_sizes, api::kFloat, api::kHeightPacked);
      viewFn(graph, {arg2, graph.add_none(), t_mat2_height_packed});
      arg2 = t_mat2_height_packed;
    }

    vTensorPtr t_mat1 = graph.get_tensor(arg1);
    vTensorPtr t_mat2 = graph.get_tensor(arg2);
    vTensorPtr t_out = graph.get_tensor(out);

    VK_CHECK_COND(check_memory_layout_is(*t_mat2, api::kHeightPacked));

    api::utils::uvec3 global_size = t_out->extents();
    api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

    std::string kernel_name("matmul_naive");
    kernel_name.reserve(kShaderNameReserve);
    add_memory_layout_suffix(kernel_name, *t_mat1);
    add_memory_layout_suffix(kernel_name, *t_mat2);
    add_dtype_suffix(kernel_name, *t_out);

    graph.execute_nodes().emplace_back(new ExecuteNode(
        graph,
        VK_KERNEL_FROM_STR(kernel_name),
        global_size,
        local_size,
        // Inputs and Outputs
        {{out, api::MemoryAccessType::WRITE},
         {{arg1, arg2}, api::MemoryAccessType::READ}},
        // Shader params buffers
        {
            t_out->texture_limits_ubo(),
            t_mat1->sizes_ubo(),
        },
        // Specialization Constants
        {},
        // Resizing Logic
        resize_matmul_node));
  } else {
    VK_THROW("Input should be channel packed or width packed.");
  }
}

void add_addmm_node(
    ComputeGraph& graph,
    const ValueRef self,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef beta,
    const ValueRef alpha,
    const ValueRef out) {
  ValueRef arg1 = prepack_if_tensor_ref(graph, mat1, api::kChannelsPacked);
  ValueRef arg2 = prepack_if_tensor_ref(graph, mat2, api::kHeightPacked);

  std::vector<int64_t> t_mat1_sizes = graph.get_tensor(arg1)->sizes();
  std::vector<int64_t> t_mat2_sizes = graph.get_tensor(arg2)->sizes();
  std::vector<int64_t> out_sizes = graph.get_tensor(out)->sizes();
  int64_t t_mat1_dim = t_mat1_sizes.size();

  ValueRef self_arg;
  int broadcast_at_width = 0;
  int broadcast_at_height = 0;
  float alpha_val = 1.0f;
  float beta_val = 1.0f;
  if (graph.memory_layout_of(arg1) == api::kChannelsPacked) {
    self_arg = prepack_if_tensor_ref(graph, self, api::kChannelsPacked);
  } else if (graph.memory_layout_of(arg1) == api::kWidthPacked) {
    self_arg = prepack_if_tensor_ref(graph, self, api::kWidthPacked);
  } else {
    VK_THROW("Input should be channel packed or width packed.");
  }

  std::vector<int64_t> self_sizes = graph.get_tensor(self_arg)->sizes();
  int64_t self_dim = self_sizes.size();
  if (self_sizes.at(self_dim - 1) < out_sizes.at(t_mat1_dim - 1)) {
    broadcast_at_width = 1;
  }
  if (self_dim < t_mat1_dim || self_sizes.at(0) < out_sizes.at(0)) {
    broadcast_at_height = 1;
  }
  alpha_val = graph.extract_scalar<float>(alpha);
  beta_val = graph.extract_scalar<float>(beta);

  AddmmParams addmm_params = {
      broadcast_at_width, broadcast_at_height, alpha_val, beta_val};

  check_matmul_args(graph, arg1, arg2, out);
  auto viewFn = VK_GET_OP_FN("aten.view_copy.default");

  // optimized mm
  if (graph.memory_layout_of(arg1) == api::kChannelsPacked) {
    ValueRef t_mat1_width_packed =
        graph.add_tensor(t_mat1_sizes, api::kFloat, api::kWidthPacked);
    viewFn(graph, {arg1, graph.add_none(), t_mat1_width_packed});
    arg1 = t_mat1_width_packed;

    if (graph.memory_layout_of(arg2) != api::kHeightPacked) {
      ValueRef t_mat2_height_packed =
          graph.add_tensor(t_mat2_sizes, api::kFloat, api::kHeightPacked);
      viewFn(graph, {arg2, graph.add_none(), t_mat2_height_packed});
      arg2 = t_mat2_height_packed;
    }

    vTensorPtr t_mat1 = graph.get_tensor(arg1);
    vTensorPtr t_mat2 = graph.get_tensor(arg2);

    VK_CHECK_COND(check_memory_layout_is(*t_mat1, api::kWidthPacked));
    VK_CHECK_COND(check_memory_layout_is(*t_mat2, api::kHeightPacked));

    // Step size is the 2d input's width dimension / 4.
    int32_t step_size =
        api::utils::div_up(t_mat1_sizes.at(t_mat1_dim - 1), INT64_C(4));

    // reminder is used in shader to detect whether the fetched texel is out of
    // boundary
    int32_t reminder = t_mat1_sizes.at(t_mat1_dim - 1) % INT64_C(4);

    int64_t batch_size = 1;
    if (t_mat1_dim == 3) {
      batch_size = t_mat1_sizes.at(0);
    }

    vTensorPtr t_out = graph.get_tensor(out);
    int64_t out_dim = out_sizes.size();

    api::utils::uvec3 global_size = {
        static_cast<unsigned int>(
            api::utils::div_up(out_sizes.at(t_mat1_dim - 1), INT64_C(4))),
        static_cast<unsigned int>(
            api::utils::div_up(out_sizes.at(t_mat1_dim - 2), INT64_C(4))),
        static_cast<unsigned int>(
            out_dim == 3 ? api::utils::div_up(out_sizes.at(0), INT64_C(4))
                         : 1)};
    api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

    std::string kernel_name("addmm_optimized");
    kernel_name.reserve(kShaderNameReserve);

    add_dtype_suffix(kernel_name, *t_out);

    graph.execute_nodes().emplace_back(new ExecuteNode(
        graph,
        VK_KERNEL_FROM_STR(kernel_name),
        global_size,
        local_size,
        // Inputs and Outputs
        {{out, api::MemoryAccessType::WRITE},
         {{arg1, arg2, self_arg}, api::MemoryAccessType::READ}},
        // Shader params buffers
        {
            t_out->texture_limits_ubo(),
            graph.create_params_buffer(step_size),
            graph.create_params_buffer(reminder),
            graph.create_params_buffer(batch_size),
            graph.create_params_buffer(addmm_params),
        },
        // Specialization Constants
        {},
        // Resizing Logic
        resize_matmul_node));
  } else if (graph.memory_layout_of(arg1) == api::kWidthPacked) {
    // native mm
    if (graph.memory_layout_of(arg2) != api::kHeightPacked) {
      ValueRef t_mat2_height_packed =
          graph.add_tensor(t_mat2_sizes, api::kFloat, api::kHeightPacked);
      viewFn(graph, {arg2, graph.add_none(), t_mat2_height_packed});
      arg2 = t_mat2_height_packed;
    }

    vTensorPtr t_mat1 = graph.get_tensor(arg1);
    vTensorPtr t_mat2 = graph.get_tensor(arg2);
    vTensorPtr t_out = graph.get_tensor(out);

    VK_CHECK_COND(check_memory_layout_is(*t_mat2, api::kHeightPacked));

    api::utils::uvec3 global_size = t_out->extents();
    api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

    std::string kernel_name("addmm_naive");
    kernel_name.reserve(kShaderNameReserve);
    add_memory_layout_suffix(kernel_name, *t_mat1);
    add_memory_layout_suffix(kernel_name, *t_mat2);
    add_dtype_suffix(kernel_name, *t_out);

    graph.execute_nodes().emplace_back(new ExecuteNode(
        graph,
        VK_KERNEL_FROM_STR(kernel_name),
        global_size,
        local_size,
        // Inputs and Outputs
        {{out, api::MemoryAccessType::WRITE},
         {{arg1, arg2, self_arg}, api::MemoryAccessType::READ}},
        // Shader params buffers
        {
            t_out->texture_limits_ubo(),
            t_mat1->sizes_ubo(),
            graph.create_params_buffer(addmm_params),
        },
        // Specialization Constants
        {},
        // Resizing Logic
        resize_matmul_node));
  } else {
    VK_THROW("Input should be channel packed or width packed.");
  }
}

void addmm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_addmm_node(
      graph, args[0], args[1], args[2], args[3], args[4], args[5]);
}

void matmul(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_matmul_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.mm.default, matmul);
  VK_REGISTER_OP(aten.bmm.default, matmul);
  VK_REGISTER_OP(aten.addmm.default, addmm);
}

} // namespace vkcompute
