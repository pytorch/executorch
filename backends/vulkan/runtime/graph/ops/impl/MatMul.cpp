/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Packing.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_matmul_args(
    const vTensor& mat1,
    const vTensor& mat2,
    const vTensor& out) {
  VK_CHECK_COND(check_same_sizes_at(mat1, -1, mat2, -2));
  VK_CHECK_COND(check_memory_layout_is(out, api::kChannelsPacked));
}

ValueRef pack_inputs_using_width_packing(
    ComputeGraph& graph,
    const ValueRef vref) {
  VK_CHECK_COND(graph.memory_layout_of(vref) == api::kChannelsPacked);
  ValueRef output = convert_image_channels_packed_to_width_packed(graph, vref);
  VK_CHECK_COND(graph.memory_layout_of(output) == api::kWidthPacked);

  return output;
}

ValueRef pack_weights_using_height_packing(
    ComputeGraph& graph,
    const ValueRef vref) {
  VK_CHECK_COND(graph.memory_layout_of(vref) == api::kChannelsPacked);
  ValueRef output = convert_image_channels_packed_to_height_packed(graph, vref);
  VK_CHECK_COND(graph.memory_layout_of(output) == api::kHeightPacked);

  return output;
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

void resize_bmm_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);

  vTensorPtr self_arg = graph->get_tensor(extra_args[0]);
  vTensorPtr mat1_arg = graph->get_tensor(extra_args[1]);

  std::vector<int64_t> new_out_sizes(3);
  new_out_sizes.at(0) = self_arg->sizes().at(0);
  new_out_sizes.at(1) = self_arg->sizes().at(1);
  new_out_sizes.at(2) = self_arg->sizes().at(2);
}

ValueRef reshape_to_2d(ComputeGraph& graph, const ValueRef in) {
  std::vector<int64_t> in_sizes = graph.get_tensor(in)->sizes();
  std::int64_t input_dim = in_sizes.size();

  std::vector<int64_t> new_sizes(2);
  ValueRef reshaped_vref;

  if (input_dim == 1) {
    // unsqueeze at dim 0
    new_sizes.at(0) = 1;
    new_sizes.at(1) = in_sizes.at(input_dim - 1);

    reshaped_vref =
        graph.add_tensor(new_sizes, api::kFloat, graph.memory_layout_of(in));
    auto unsqueezeFn = VK_GET_OP_FN("aten.unsqueeze_copy.default");
    unsqueezeFn(graph, {in, graph.add_scalar<int64_t>(0), reshaped_vref});
  } else {
    // reshape 3d to 2d
    const int64_t d =
        api::utils::multiply_integers(in_sizes.cbegin(), in_sizes.cend() - 1);

    new_sizes.at(0) = d;
    new_sizes.at(1) = in_sizes.at(input_dim - 1);

    reshaped_vref =
        graph.add_tensor(new_sizes, api::kFloat, graph.memory_layout_of(in));

    auto reshapeFn = VK_GET_OP_FN("aten.reshape.default");
    reshapeFn(
        graph,
        {in,
         graph.add_scalar_list<int64_t>(std::move(new_sizes)),
         reshaped_vref});
  }

  return reshaped_vref;
}

void add_matmul_2d_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef out) {
  ValueRef arg1;
  if (graph.val_is_tref(mat1)) {
    arg1 = prepack_if_tensor_ref(graph, mat1, api::kWidthPacked);
  } else {
    arg1 = pack_inputs_using_width_packing(graph, mat1);
  }

  ValueRef arg2;
  if (graph.val_is_tref(mat2)) {
    arg2 = prepack_if_tensor_ref(graph, mat2, api::kHeightPacked);
  } else {
    arg2 = pack_weights_using_height_packing(graph, mat2);
  }

  vTensorPtr t_mat1 = graph.get_tensor(arg1);
  vTensorPtr t_mat2 = graph.get_tensor(arg2);
  vTensorPtr t_out = graph.get_tensor(out);

  check_matmul_args(*t_mat1, *t_mat2, *t_out);

  // Step size is the 2d input's width dimension / 4.
  int32_t step_size = api::utils::div_up(t_mat1->sizes().at(1), INT64_C(4));
  std::vector<int64_t> out_sizes = t_out->sizes();

  api::utils::uvec3 global_size = {
      static_cast<unsigned int>(
          api::utils::div_up(out_sizes.at(1), INT64_C(4))),
      static_cast<unsigned int>(
          api::utils::div_up(out_sizes.at(0), INT64_C(4))),
      1};
  api::utils::uvec3 local_size = {8, 8, 1};

  std::string kernel_name("matmul");
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
      },
      // Specialization Constants
      {},
      // Resizing Logic
      resize_matmul_node));
}

void add_matmul_reshape_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef out) {
  ValueRef arg1 = prepack_if_tensor_ref(graph, mat1, api::kChannelsPacked);
  // Reshape mat1 to 2d
  ValueRef arg1_reshaped = reshape_to_2d(graph, arg1);
  std::vector<int64_t> arg1_reshaped_sizes =
      graph.get_tensor(arg1_reshaped)->sizes();
  std::vector<int64_t> out_sizes = graph.get_tensor(out)->sizes();
  std::vector<int64_t> out_2d_sizes(2);
  out_2d_sizes.at(0) = arg1_reshaped_sizes.at(0);
  out_2d_sizes.at(1) = out_sizes.at(out_sizes.size() - 1);
  ValueRef out_2d_vref =
      graph.add_tensor(out_2d_sizes, api::kFloat, graph.memory_layout_of(arg1));
  add_matmul_2d_node(graph, arg1_reshaped, mat2, out_2d_vref);

  // Reshape back to 3d
  auto reshapeFn = VK_GET_OP_FN("aten.reshape.default");
  reshapeFn(
      graph,
      {out_2d_vref, graph.add_scalar_list<int64_t>(std::move(out_sizes)), out});
}

void add_matmul_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef out) {
  std::vector<int64_t> mat1_sizes = graph.get_tensor(mat1)->sizes();
  int64_t mat1_dim = mat1_sizes.size();
  if (mat1_dim == 2) {
    add_matmul_2d_node(graph, mat1, mat2, out);
  } else if (mat1_dim == 1 || mat1_dim == 3) {
    add_matmul_reshape_node(graph, mat1, mat2, out);
  } else {
    VK_THROW("matmul only support 1d, 2d and 3d inputs.");
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
  // mat1 @ mat2
  ValueRef arg1 = prepack_if_tensor_ref(graph, mat1, api::kChannelsPacked);
  int64_t arg1_dim = graph.get_tensor(arg1)->sizes().size();
  std::vector<int64_t> out_sizes = graph.get_tensor(out)->sizes();
  std::vector<int64_t> mm_out_sizes(arg1_dim);
  for (int i = 0; i < arg1_dim - 1; ++i) {
    mm_out_sizes.at(i) = out_sizes.at(i);
  }
  mm_out_sizes.at(arg1_dim - 1) = out_sizes.at(out_sizes.size() - 1);
  ValueRef mm_out =
      graph.add_tensor(mm_out_sizes, api::kFloat, api::kChannelsPacked);
  add_matmul_node(graph, mat1, mat2, mm_out);

  // We create a dummy zero tensor and use `aten.add.Tensor` to compute `beta *
  // self` through add(dummy, self, beta)
  ValueRef self_vref = prepack_if_tensor_ref(graph, self, api::kChannelsPacked);
  std::vector<int64_t> self_sizes = graph.get_tensor(self_vref)->sizes();
  ValueRef beta_self = graph.add_tensor(
      self_sizes, api::kFloat, graph.memory_layout_of(self_vref));
  ValueRef dummy_vref =
      graph.add_tensor({1}, api::kFloat, graph.memory_layout_of(self_vref));
  auto addFn = VK_GET_OP_FN("aten.add.Tensor");
  addFn(graph, {dummy_vref, self_vref, beta, beta_self});

  // beta * self + alpha * mat1 @ mat2
  addFn(graph, {beta_self, mm_out, alpha, out});
}

void add_bmm_node(
    ComputeGraph& graph,
    const ValueRef self,
    const ValueRef mat2,
    const ValueRef out) {
  ValueRef self_arg;
  if (graph.val_is_tref(self)) {
    self_arg = prepack_if_tensor_ref(graph, self, api::kWidthPacked);
  } else {
    self_arg = pack_inputs_using_width_packing(graph, self);
  }

  ValueRef arg2;
  if (graph.val_is_tref(mat2)) {
    arg2 = prepack_if_tensor_ref(graph, mat2, api::kHeightPacked);
  } else {
    arg2 = pack_weights_using_height_packing(graph, mat2);
  }

  // In the shader, each batch is computed in separate invocation.
  // The result is stored in the .x position of the texel.
  // As the tensor by default is channel packed, the shader is effectively
  // producing 3 all-zeros layer. We workaround this issue by creating
  // a vTensor that is 4 times the batch size.
  // At the end of the computation, we run a "slice" with a step-size of 4
  // to get back the original shape.

  std::vector<int64_t> self_sizes = graph.get_tensor(self_arg)->sizes();
  int64_t input_batch = self_sizes.at(0);
  int64_t input_height = self_sizes.at(1);
  int64_t input_width = self_sizes.at(2);

  std::vector<int64_t> weight_sizes = graph.get_tensor(arg2)->sizes();
  int64_t weight_width = weight_sizes.at(2);

  // Step size is the input's w dimension / 4.
  int64_t mm_step_size = api::utils::div_up(input_width, INT64_C(4));

  // Create variables for the slice operator. Although `slice` is applied at the
  // end, we call `graph.add_*` early to avoid `values_in_use_ != 0` error when
  // calling `ComputeGraph::get_*()`, https://fburl.com/code/m6oi7irq.
  int64_t dim_sliced = 0;
  int64_t start = 0;
  int64_t step = 4;
  int64_t end = input_batch * step;
  ValueRef dim_sliced_ref = graph.add_scalar(dim_sliced);
  ValueRef start_ref = graph.add_scalar(start);
  ValueRef step_ref = graph.add_scalar(step);
  ValueRef end_ref = graph.add_scalar(end);

  std::vector<int64_t> out_packed_sizes = {
      input_batch * 4, input_height, weight_width};
  ValueRef out_packed =
      graph.add_tensor(out_packed_sizes, api::kFloat, api::kChannelsPacked);
  vTensorPtr t_out_packed = graph.get_tensor(out_packed);

  api::utils::uvec3 global_size = {
      static_cast<unsigned int>(
          api::utils::div_up(out_packed_sizes.at(2), INT64_C(4))),
      static_cast<unsigned int>(
          api::utils::div_up(out_packed_sizes.at(1), INT64_C(4))),
      static_cast<unsigned int>(out_packed_sizes.at(0))};
  api::utils::uvec3 local_size = {8, 8, 1};

  std::string kernel_name("matmul");
  kernel_name.reserve(kShaderNameReserve);

  add_dtype_suffix(kernel_name, *t_out_packed);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out_packed, api::MemoryAccessType::WRITE},
       {{self_arg, arg2}, api::MemoryAccessType::READ}},
      // Shader params buffers
      {
          t_out_packed->texture_limits_ubo(),
          graph.create_params_buffer(mm_step_size),
      },
      // Specialization Constants
      {},
      // Resizing Logic
      resize_bmm_node,
      {self_arg, arg2}));

  // After computing the multiplication, we need to slice 4 on the batch
  // dimension to get the channel packed layout.
  auto sliceFn = VK_GET_OP_FN("aten.slice_copy.Tensor");
  sliceFn(
      graph, {out_packed, dim_sliced_ref, start_ref, end_ref, step_ref, out});
}

void addmm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_addmm_node(
      graph, args[0], args[1], args[2], args[3], args[4], args[5]);
}

void bmm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_bmm_node(graph, args[0], args[1], args[2]);
}

void matmul(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_matmul_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.addmm.default, addmm);
  VK_REGISTER_OP(aten.bmm.default, bmm);
  VK_REGISTER_OP(aten.mm.default, matmul);
}

} // namespace vkcompute
