/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

namespace vkcompute {

void resize_quantize_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);

  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  graph->virtual_resize(out, in_sizes);
}

utils::uvec3 quantize_per_channel_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  (void)resize_args;

  const ValueRef input = args.at(1).refs.at(0);

  utils::uvec3 local_wg_size =
      graph->create_local_wg_size(global_workgroup_size);

  // WORKAROUND: The CommandBuffer::dispatch function divides
  // global_workgroup_size by local_workgroup_size to get the number of
  // workgroups to dispatch. For per-channel quantization along the batch axis,
  // we need to ensure that we dispatch the correct number of workgroups in the
  // Z dimension to cover all batch-channel combinations.
  //
  // If local_wg_size[2] > 1, then div_up(global_workgroup_size[2],
  // local_wg_size[2]) might reduce the number of workgroups dispatched. To
  // ensure we dispatch global_workgroup_size[2] workgroups in the Z dimension,
  // we set local_wg_size[2] = 1.
  const auto input_sizes = graph->sizes_of(input);
  if (input_sizes.size() == 4 && !graph->is_buffer_storage(input) &&
      global_workgroup_size[2] > 1) {
    local_wg_size[2] = 1;
  }

  return local_wg_size;
}

utils::uvec3 quantize_block_wise_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef input = args.at(1).refs.at(0);

  utils::uvec3 local_wg_size =
      graph->create_local_wg_size(global_workgroup_size);

  // WORKAROUND: The CommandBuffer::dispatch function divides
  // global_workgroup_size by local_workgroup_size to get the number of
  // workgroups to dispatch. For per-channel quantization along the batch axis,
  // we need to ensure that we dispatch the correct number of workgroups in the
  // Z dimension to cover all batch-channel combinations.
  //
  // If local_wg_size[2] > 1, then div_up(global_workgroup_size[2],
  // local_wg_size[2]) might reduce the number of workgroups dispatched. To
  // ensure we dispatch global_workgroup_size[2] workgroups in the Z dimension,
  // we set local_wg_size[2] = 1.
  const auto input_sizes = graph->sizes_of(input);
  if (input_sizes.size() == 4 && !graph->is_buffer_storage(input) &&
      global_workgroup_size[2] > 1) {
    local_wg_size[2] = 1;
  }

  return local_wg_size;
}

void add_quantize_per_tensor_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& scale,
    const ValueRef& zero_point,
    const ValueRef& quant_min,
    const ValueRef& quant_max,
    const ValueRef& output) {
  std::string kernel_name("quantize_per_tensor");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));
  add_dtype_suffix(kernel_name, graph.dtype_of(scale));
  add_dtype_suffix(kernel_name, graph.dtype_of(zero_point));

  // Handle optional quant_min and quant_max parameters independently
  auto bounds = get_dtype_bounds(graph.dtype_of(output));

  int quant_min_val, quant_max_val;

  // Handle quant_min
  if (graph.val_is_none(quant_min)) {
    quant_min_val = bounds.first;
  } else {
    VK_CHECK_COND(
        graph.val_is_int(quant_min),
        "quant_min must be an integer, got type: ",
        graph.get_val_type(quant_min));
    quant_min_val = static_cast<int>(graph.get_int(quant_min));
  }

  // Handle quant_max
  if (graph.val_is_none(quant_max)) {
    quant_max_val = bounds.second;
  } else {
    VK_CHECK_COND(
        graph.val_is_int(quant_max),
        "quant_max must be an integer, got type: ",
        graph.get_val_type(quant_max));
    quant_max_val = static_cast<int>(graph.get_int(quant_max));
  }

  vkapi::ParamsBindList param_ubos;
  std::vector<PushConstantDataInfo> push_constants;

  if (graph.is_buffer_storage(input)) {
    param_ubos = {
        graph.numel_ubo(input),
        graph.sizes_ubo(input),
        graph.strides_ubo(input),
        graph.sizes_ubo(output),
        graph.strides_ubo(output)};
  } else {
    param_ubos = {
        graph.logical_limits_ubo(input), graph.logical_limits_ubo(output)};
  }

  push_constants = {
      PushConstantDataInfo(&quant_min_val, sizeof(int)),
      PushConstantDataInfo(&quant_max_val, sizeof(int)),
  };

  vkapi::SpecVarList spec_vars = {
      graph.hashed_layout_of(output),
      graph.hashed_layout_of(input),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {input, vkapi::kRead},
       {{scale, zero_point}, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      spec_vars,
      // Resize Args
      {},
      // Resizing Logic
      resize_quantize_node));
}

void add_quantize_per_token_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& scale,
    const ValueRef& zero_point,
    const ValueRef& quant_min,
    const ValueRef& quant_max,
    const ValueRef& output) {
  std::string kernel_name("quantize_per_token");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));
  add_dtype_suffix(kernel_name, graph.dtype_of(scale));
  add_dtype_suffix(kernel_name, graph.dtype_of(zero_point));

  // Handle optional quant_min and quant_max parameters independently
  auto bounds = get_dtype_bounds(graph.dtype_of(output));

  int quant_min_val, quant_max_val;

  // Handle quant_min
  if (graph.val_is_none(quant_min)) {
    quant_min_val = bounds.first;
  } else {
    VK_CHECK_COND(
        graph.val_is_int(quant_min),
        "quant_min must be an integer, got type: ",
        graph.get_val_type(quant_min));
    quant_min_val = static_cast<int>(graph.get_int(quant_min));
  }

  // Handle quant_max
  if (graph.val_is_none(quant_max)) {
    quant_max_val = bounds.second;
  } else {
    VK_CHECK_COND(
        graph.val_is_int(quant_max),
        "quant_max must be an integer, got type: ",
        graph.get_val_type(quant_max));
    quant_max_val = static_cast<int>(graph.get_int(quant_max));
  }

  int num_tokens = static_cast<int>(graph.sizes_of(scale)[0]);

  vkapi::ParamsBindList param_ubos;
  std::vector<PushConstantDataInfo> push_constants;

  if (graph.is_buffer_storage(input)) {
    param_ubos = {
        graph.numel_ubo(input),
        graph.sizes_ubo(input),
        graph.strides_ubo(input),
        graph.sizes_ubo(output),
        graph.strides_ubo(output),
    };
    push_constants = {
        PushConstantDataInfo(&num_tokens, sizeof(int)),
        PushConstantDataInfo(&quant_min_val, sizeof(int)),
        PushConstantDataInfo(&quant_max_val, sizeof(int)),
    };
  } else {
    param_ubos = {
        graph.logical_limits_ubo(input),
        graph.logical_limits_ubo(output),
    };
    push_constants = {
        PushConstantDataInfo(&num_tokens, sizeof(int)),
        PushConstantDataInfo(&quant_min_val, sizeof(int)),
        PushConstantDataInfo(&quant_max_val, sizeof(int)),
    };
  }

  vkapi::SpecVarList spec_vars = {
      graph.hashed_layout_of(output),
      graph.hashed_layout_of(input),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {input, vkapi::kRead},
       {{scale, zero_point}, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      spec_vars,
      // Resize Args
      {},
      // Resizing Logic
      resize_quantize_node));
}

void add_quantize_per_channel_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& scale,
    const ValueRef& zero_point,
    const ValueRef& axis,
    const ValueRef& quant_min,
    const ValueRef& quant_max,
    const ValueRef& output) {
  std::string kernel_name("quantize_per_channel");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));
  add_dtype_suffix(kernel_name, graph.dtype_of(scale));
  add_dtype_suffix(kernel_name, graph.dtype_of(zero_point));

  int axis_val = static_cast<int>(graph.get_int(axis));

  // Handle optional quant_min and quant_max parameters independently
  auto bounds = get_dtype_bounds(graph.dtype_of(output));

  int quant_min_val, quant_max_val;

  // Handle quant_min
  if (graph.val_is_none(quant_min)) {
    quant_min_val = bounds.first;
  } else {
    VK_CHECK_COND(
        graph.val_is_int(quant_min),
        "quant_min must be an integer, got type: ",
        graph.get_val_type(quant_min));
    quant_min_val = static_cast<int>(graph.get_int(quant_min));
  }

  // Handle quant_max
  if (graph.val_is_none(quant_max)) {
    quant_max_val = bounds.second;
  } else {
    VK_CHECK_COND(
        graph.val_is_int(quant_max),
        "quant_max must be an integer, got type: ",
        graph.get_val_type(quant_max));
    quant_max_val = static_cast<int>(graph.get_int(quant_max));
  }

  // Normalize axis and convert from NCHW to WHCN using utility functions
  const auto input_sizes = graph.sizes_of(input);
  const int64_t ndim = graph.dim_of(input);

  // Normalize axis to handle negative indices
  axis_val = normalize(axis_val, ndim);

  // Convert from NCHW axis to WHCN axis for shader (vulkan representation)
  int axis_whcn = nchw_dim_to_whcn_dim(axis_val, ndim);

  int num_channels;
  if (axis_val == 0 && ndim == 4 && !graph.is_buffer_storage(input)) {
    // For batch dimension quantization in 4D tensors, pass the actual number of
    // channels so the shader can correctly unfold the batch-channel folding
    num_channels = static_cast<int>(input_sizes[1]); // Channel dimension
  } else {
    num_channels = static_cast<int>(input_sizes[axis_val]);
  }

  vkapi::ParamsBindList param_ubos;
  std::vector<PushConstantDataInfo> push_constants;

  if (graph.is_buffer_storage(input)) {
    param_ubos = {
        graph.numel_ubo(input),
        graph.sizes_ubo(input),
        graph.strides_ubo(input),
        graph.sizes_ubo(output),
        graph.strides_ubo(output),
    };
    push_constants = {
        PushConstantDataInfo(&axis_whcn, sizeof(int)),
        PushConstantDataInfo(&num_channels, sizeof(int)),
        PushConstantDataInfo(&quant_min_val, sizeof(int)),
        PushConstantDataInfo(&quant_max_val, sizeof(int)),
    };
  } else {
    param_ubos = {
        graph.logical_limits_ubo(input),
        graph.logical_limits_ubo(output),
    };
    push_constants = {
        PushConstantDataInfo(&axis_whcn, sizeof(int)),
        PushConstantDataInfo(&num_channels, sizeof(int)),
        PushConstantDataInfo(&quant_min_val, sizeof(int)),
        PushConstantDataInfo(&quant_max_val, sizeof(int)),
    };
  }

  vkapi::SpecVarList spec_vars = {
      graph.hashed_layout_of(output),
      graph.hashed_layout_of(input),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      quantize_per_channel_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {input, vkapi::kRead},
       {{scale, zero_point}, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      spec_vars,
      // Resize Args
      {},
      // Resizing Logic
      resize_quantize_node));
}

void add_quantize_block_wise_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& block_size,
    const ValueRef& scale,
    const ValueRef& zero_point,
    const ValueRef& quant_min,
    const ValueRef& quant_max,
    const ValueRef& output) {
  std::string kernel_name("quantize_block_wise");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));
  add_dtype_suffix(kernel_name, graph.dtype_of(scale));
  add_dtype_suffix(kernel_name, graph.dtype_of(zero_point));

  // Handle optional quant_min and quant_max parameters independently
  auto bounds = get_dtype_bounds(graph.dtype_of(output));

  int quant_min_val, quant_max_val;

  // Handle quant_min
  if (graph.val_is_none(quant_min)) {
    quant_min_val = bounds.first;
  } else {
    VK_CHECK_COND(
        graph.val_is_int(quant_min),
        "quant_min must be an integer, got type: ",
        graph.get_val_type(quant_min));
    quant_min_val = static_cast<int>(graph.get_int(quant_min));
  }

  // Handle quant_max
  if (graph.val_is_none(quant_max)) {
    quant_max_val = bounds.second;
  } else {
    VK_CHECK_COND(
        graph.val_is_int(quant_max),
        "quant_max must be an integer, got type: ",
        graph.get_val_type(quant_max));
    quant_max_val = static_cast<int>(graph.get_int(quant_max));
  }

  const auto input_sizes = graph.sizes_of(input);
  const auto block_size_list = graph.get_int_list(block_size);

  // Convert PyTorch dimensions to WHCN order for shader
  utils::ivec4 block_size_vec = utils::make_whcn_ivec4(*block_size_list);
  utils::ivec4 tensor_size_whcn = utils::make_whcn_ivec4(input_sizes);

  // Calculate numBlocks: tensorSize / blockSize (both in WHCN order)
  utils::ivec4 num_blocks_vec = {
      tensor_size_whcn[0] / block_size_vec[0],
      tensor_size_whcn[1] / block_size_vec[1],
      tensor_size_whcn[2] / block_size_vec[2],
      tensor_size_whcn[3] / block_size_vec[3]};

  // Calculate blockStride: pre-computed linear strides for the block grid
  utils::ivec4 block_stride_vec = {
      1,
      num_blocks_vec[0],
      num_blocks_vec[0] * num_blocks_vec[1],
      num_blocks_vec[0] * num_blocks_vec[1] * num_blocks_vec[2]};

  vkapi::ParamsBindList param_ubos;
  std::vector<PushConstantDataInfo> push_constants;

  if (graph.is_buffer_storage(input)) {
    param_ubos = {
        graph.numel_ubo(input),
        graph.sizes_ubo(input),
        graph.strides_ubo(input),
        graph.sizes_ubo(output),
        graph.strides_ubo(output)};
  } else {
    param_ubos = {
        graph.logical_limits_ubo(input), graph.logical_limits_ubo(output)};
  }

  push_constants = {
      PushConstantDataInfo(&block_size_vec, sizeof(block_size_vec)),
      PushConstantDataInfo(&num_blocks_vec, sizeof(num_blocks_vec)),
      PushConstantDataInfo(&block_stride_vec, sizeof(block_stride_vec)),
      PushConstantDataInfo(&quant_min_val, sizeof(int)),
      PushConstantDataInfo(&quant_max_val, sizeof(int)),
  };

  vkapi::SpecVarList spec_vars = {
      graph.hashed_layout_of(output),
      graph.hashed_layout_of(input),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      quantize_block_wise_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {input, vkapi::kRead},
       {{scale, zero_point}, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      spec_vars,
      // Resize Args
      {},
      // Resizing Logic
      resize_quantize_node));
}

void quantize_per_tensor_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];
  const ValueRef zero_point = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef dtype = args[arg_idx++];
  const ValueRef output = args[arg_idx++];

  // Suppress unused variable warning - dtype is inferred from output
  (void)dtype;

  // Check tensor types
  VK_CHECK_COND(graph.val_is_tensor(input));
  VK_CHECK_COND(graph.val_is_tensor(scale));
  VK_CHECK_COND(graph.val_is_tensor(zero_point));
  VK_CHECK_COND(graph.val_is_tensor(output));

  // Verify input is a floating point type
  VK_CHECK_COND(
      graph.dtype_of(input) == vkapi::kDouble ||
      graph.dtype_of(input) == vkapi::kFloat ||
      graph.dtype_of(input) == vkapi::kHalf);

  // Get scale and zero point dtypes
  vkapi::ScalarType scale_dtype = graph.dtype_of(scale);
  vkapi::ScalarType zero_point_dtype = graph.dtype_of(zero_point);

  // Verify supported types for scale (fp32 only for now)
  VK_CHECK_COND(scale_dtype == vkapi::kFloat);

  // Verify supported types for zero point (int32, int8, fp32)
  VK_CHECK_COND(
      zero_point_dtype == vkapi::kInt || zero_point_dtype == vkapi::kChar ||
      zero_point_dtype == vkapi::kFloat);

  add_quantize_per_tensor_node(
      graph, input, scale, zero_point, quant_min, quant_max, output);
}

void quantize_per_token_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];
  const ValueRef zero_point = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef dtype = args[arg_idx++];
  const ValueRef output = args[arg_idx++];

  // Suppress unused variable warning - dtype is inferred from output
  (void)dtype;

  // Check tensor types
  VK_CHECK_COND(graph.val_is_tensor(input));
  VK_CHECK_COND(graph.val_is_tensor(scale));
  VK_CHECK_COND(graph.val_is_tensor(zero_point));
  VK_CHECK_COND(graph.val_is_tensor(output));

  // Verify input is a floating point type
  VK_CHECK_COND(
      graph.dtype_of(input) == vkapi::kDouble ||
      graph.dtype_of(input) == vkapi::kFloat ||
      graph.dtype_of(input) == vkapi::kHalf);

  // Get scale and zero point dtypes
  vkapi::ScalarType scale_dtype = graph.dtype_of(scale);
  vkapi::ScalarType zero_point_dtype = graph.dtype_of(zero_point);

  // Verify supported types for scale (fp32 only for now)
  VK_CHECK_COND(scale_dtype == vkapi::kFloat);

  // Verify supported types for zero point (int32, int8, fp32)
  VK_CHECK_COND(
      zero_point_dtype == vkapi::kInt || zero_point_dtype == vkapi::kChar ||
      zero_point_dtype == vkapi::kFloat);

  // Check that scale and zero_point have buffer storage and width packing
  VK_CHECK_COND(graph.is_buffer_storage(scale));
  VK_CHECK_COND(graph.packed_dim_of(scale) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.is_buffer_storage(zero_point));
  VK_CHECK_COND(graph.packed_dim_of(zero_point) == WHCN::kWidthDim);

  // Check that tensors with texture storage have standard axis map
  if (!graph.is_buffer_storage(input)) {
    VK_CHECK_COND(graph.has_standard_axis_map(input));
  }
  if (!graph.is_buffer_storage(output)) {
    VK_CHECK_COND(graph.has_standard_axis_map(output));
  }

  // Calculate number of tokens (product of all dimensions except the last one)
  int64_t num_tokens = 1;
  const auto input_sizes = graph.sizes_of(input);
  for (size_t i = 0; i < input_sizes.size() - 1; i++) {
    num_tokens *= input_sizes[i];
  }

  const auto scale_sizes = graph.sizes_of(scale);
  const auto zero_point_sizes = graph.sizes_of(zero_point);

  // Calculate total number of elements in scale and zero_point tensors
  int64_t scale_numel = 1;
  for (size_t i = 0; i < scale_sizes.size(); i++) {
    scale_numel *= scale_sizes[i];
  }

  int64_t zero_point_numel = 1;
  for (size_t i = 0; i < zero_point_sizes.size(); i++) {
    zero_point_numel *= zero_point_sizes[i];
  }

  // Check that the total number of elements matches num_tokens
  // This allows for both 1D tensors (size [num_tokens]) and reshaped tensors
  // (size [num_tokens, 1])
  VK_CHECK_COND(scale_numel == num_tokens);
  VK_CHECK_COND(zero_point_numel == num_tokens);

  add_quantize_per_token_node(
      graph, input, scale, zero_point, quant_min, quant_max, output);
}

void quantize_per_channel_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];
  const ValueRef zero_point = args[arg_idx++];
  const ValueRef axis = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef dtype = args[arg_idx++];
  const ValueRef output = args[arg_idx++];

  // Suppress unused variable warning - dtype is inferred from output
  (void)dtype;

  // Check tensor types
  VK_CHECK_COND(graph.val_is_tensor(input));
  VK_CHECK_COND(graph.val_is_tensor(scale));
  VK_CHECK_COND(graph.val_is_tensor(zero_point));
  VK_CHECK_COND(graph.val_is_tensor(output));

  // Verify input is a floating point type
  VK_CHECK_COND(
      graph.dtype_of(input) == vkapi::kDouble ||
      graph.dtype_of(input) == vkapi::kFloat ||
      graph.dtype_of(input) == vkapi::kHalf);

  // Get scale and zero point dtypes
  vkapi::ScalarType scale_dtype = graph.dtype_of(scale);
  vkapi::ScalarType zero_point_dtype = graph.dtype_of(zero_point);

  // Verify supported types for scale (fp32 only for now)
  VK_CHECK_COND(scale_dtype == vkapi::kFloat);

  // Verify supported types for zero point (int32, int8, fp32)
  VK_CHECK_COND(
      zero_point_dtype == vkapi::kInt || zero_point_dtype == vkapi::kChar ||
      zero_point_dtype == vkapi::kFloat);

  // Check that scale and zero_point have buffer storage and width packing
  VK_CHECK_COND(graph.is_buffer_storage(scale));
  VK_CHECK_COND(graph.packed_dim_of(scale) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.is_buffer_storage(zero_point));
  VK_CHECK_COND(graph.packed_dim_of(zero_point) == WHCN::kWidthDim);

  // Check that tensors with texture storage have standard axis map
  if (!graph.is_buffer_storage(input)) {
    VK_CHECK_COND(graph.has_standard_axis_map(input));
  }
  if (!graph.is_buffer_storage(output)) {
    VK_CHECK_COND(graph.has_standard_axis_map(output));
  }

  // Normalize axis
  int axis_val = static_cast<int>(graph.get_int(axis));
  const auto input_sizes = graph.sizes_of(input);
  int64_t ndim = graph.dim_of(input);
  if (axis_val < 0) {
    axis_val += ndim;
  }

  // Verify axis is valid
  VK_CHECK_COND(axis_val >= 0 && axis_val < ndim);

  // Get number of channels along the specified axis
  int64_t num_channels = input_sizes[axis_val];

  const auto scale_sizes = graph.sizes_of(scale);
  const auto zero_point_sizes = graph.sizes_of(zero_point);

  // Calculate total number of elements in scale and zero_point tensors
  int64_t scale_numel = 1;
  for (size_t i = 0; i < scale_sizes.size(); i++) {
    scale_numel *= scale_sizes[i];
  }

  int64_t zero_point_numel = 1;
  for (size_t i = 0; i < zero_point_sizes.size(); i++) {
    zero_point_numel *= zero_point_sizes[i];
  }

  // Check that the total number of elements matches num_channels
  VK_CHECK_COND(scale_numel == num_channels);
  VK_CHECK_COND(zero_point_numel == num_channels);

  add_quantize_per_channel_node(
      graph, input, scale, zero_point, axis, quant_min, quant_max, output);
}

void quantize_affine_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef block_size = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];
  const ValueRef zero_point = args[arg_idx++];
  const ValueRef output_dtype = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef output = args[arg_idx++];

  // Suppress unused variable warnings
  (void)output_dtype;

  // Check tensor types
  VK_CHECK_COND(graph.val_is_tensor(input));
  VK_CHECK_COND(graph.val_is_tensor(scale));
  VK_CHECK_COND(graph.val_is_tensor(zero_point));
  VK_CHECK_COND(graph.val_is_tensor(output));

  // Verify input is a floating point type
  VK_CHECK_COND(
      graph.dtype_of(input) == vkapi::kDouble ||
      graph.dtype_of(input) == vkapi::kFloat ||
      graph.dtype_of(input) == vkapi::kHalf);

  // Get scale and zero point dtypes
  vkapi::ScalarType scale_dtype = graph.dtype_of(scale);
  vkapi::ScalarType zero_point_dtype = graph.dtype_of(zero_point);

  // Verify supported types for scale (fp32 only for now)
  VK_CHECK_COND(scale_dtype == vkapi::kFloat);

  // Verify supported types for zero point (int32, int8, fp32)
  VK_CHECK_COND(
      zero_point_dtype == vkapi::kInt || zero_point_dtype == vkapi::kChar ||
      zero_point_dtype == vkapi::kFloat);

  // Check that scale and zero_point have buffer storage and width packing
  VK_CHECK_COND(graph.is_buffer_storage(scale));
  VK_CHECK_COND(graph.packed_dim_of(scale) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.is_buffer_storage(zero_point));
  VK_CHECK_COND(graph.packed_dim_of(zero_point) == WHCN::kWidthDim);

  // Check that tensors with texture storage have standard axis map
  if (!graph.is_buffer_storage(input)) {
    VK_CHECK_COND(graph.has_standard_axis_map(input));
  }
  if (!graph.is_buffer_storage(output)) {
    VK_CHECK_COND(graph.has_standard_axis_map(output));
  }

  // Verify block_size is valid (each dimension must divide evenly into input
  // size)
  const auto input_sizes = graph.sizes_of(input);
  const auto block_size_list = graph.get_int_list(block_size);
  VK_CHECK_COND(block_size_list->size() == input_sizes.size());

  for (size_t i = 0; i < input_sizes.size(); i++) {
    if ((*block_size_list)[i] > 1) {
      VK_CHECK_COND(
          input_sizes[i] % (*block_size_list)[i] == 0,
          "Input size at dimension ",
          i,
          " (",
          input_sizes[i],
          ") must be divisible by block_size at dimension ",
          i,
          " (",
          (*block_size_list)[i],
          ")");
    }
  }

  add_quantize_block_wise_node(
      graph,
      input,
      block_size,
      scale,
      zero_point,
      quant_min,
      quant_max,
      output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(
      quantized_decomposed.quantize_per_tensor.tensor,
      quantize_per_tensor_impl);
  VK_REGISTER_OP(
      quantized_decomposed.quantize_per_token.default, quantize_per_token_impl);
  VK_REGISTER_OP(
      quantized_decomposed.quantize_per_channel.default,
      quantize_per_channel_impl);

  // TorchAO affine quantization operators
  VK_REGISTER_OP(torchao.quantize_affine.default, quantize_affine_impl);
}

} // namespace vkcompute
