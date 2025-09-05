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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

namespace vkcompute {

void resize_choose_qparams_per_row(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;

  ValueRef input_scales = args.at(0).refs.at(0);
  ValueRef input_zeros = args.at(0).refs.at(1);
  ValueRef input = args.at(1).refs.at(0);

  std::vector<int64_t> new_sizes = graph->sizes_of(input_scales);
  const size_t ndim = new_sizes.size();

  const int64_t input_height = graph->size_at<int64_t>(-2, input);
  new_sizes.at(ndim - 1) = input_height;

  graph->virtual_resize(input_scales, new_sizes);
  graph->virtual_resize(input_zeros, new_sizes);
}

utils::uvec3 choose_qparams_pick_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  // For per-tensor quantization, we want a single workgroup that can handle
  // all elements with proper reduction. The shader uses NWORKERS=64 threads.
  const ValueRef input = args.at(1).refs.at(0);

  if (graph->is_buffer_storage(input)) {
    // For buffer storage, use a single workgroup in X dimension
    // The shader will handle strided access across all elements
    return {1u, 1u, 1u};
  } else {
    // For texture storage, use the default logic
    return graph->create_global_wg_size(args.at(0).refs.at(0));
  }
}

utils::uvec3 choose_qparams_pick_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef input = args.at(1).refs.at(0);

  if (graph->is_buffer_storage(input)) {
    // For buffer storage, use 64 threads in X dimension to match NWORKERS
    // This ensures the shared memory arrays are properly sized
    return {64u, 1u, 1u};
  } else {
    // For texture storage, use the default logic
    return graph->create_local_wg_size(global_workgroup_size);
  }
}

utils::uvec3 choose_qparams_per_token_pick_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef input = args.at(1).refs.at(0);

  if (graph->is_buffer_storage(input)) {
    // For per-token quantization, we need one workgroup per token
    // Calculate number of tokens (product of all dimensions except the last
    // one)
    const auto input_sizes = graph->sizes_of(input);
    int64_t num_tokens = 1;
    for (size_t i = 0; i < input_sizes.size() - 1; i++) {
      num_tokens *= input_sizes[i];
    }

    return {static_cast<uint32_t>(num_tokens), 1u, 1u};
  } else {
    // For texture storage, use the default logic
    return graph->create_global_wg_size(args.at(0).refs.at(0));
  }
}

utils::uvec3 choose_qparams_per_token_pick_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef input = args.at(1).refs.at(0);

  if (graph->is_buffer_storage(input)) {
    return {1u, 1u, 1u};
  } else {
    // For texture storage, use the default logic
    return graph->create_local_wg_size(global_workgroup_size);
  }
}

utils::uvec3 choose_qparams_block_wise_pick_global_wg_size(
    ComputeGraph* g,
    const vkapi::ShaderInfo&,
    const std::vector<ArgGroup>& a,
    const std::vector<ValueRef>& r) {
  const ValueRef input = a.at(2).refs.at(0);
  const auto blkRef = r.at(0);
  const auto inSz = g->sizes_of(input);
  const auto blkList = g->get_int_list(blkRef);

  // Use same code as in add_choose_qparams_block_wise_node
  utils::ivec4 block_size_vec = utils::make_whcn_ivec4(*blkList);
  utils::ivec4 tensor_size_whcn = utils::make_whcn_ivec4(inSz);

  // Calculate numBlocks: ceil(tensorSize / blockSize) (both in WHCN order)
  utils::ivec4 nBlk = {
      (tensor_size_whcn[0] + block_size_vec[0] - 1) / block_size_vec[0],
      (tensor_size_whcn[1] + block_size_vec[1] - 1) / block_size_vec[1],
      (tensor_size_whcn[2] + block_size_vec[2] - 1) / block_size_vec[2],
      (tensor_size_whcn[3] + block_size_vec[3] - 1) / block_size_vec[3]};

  uint32_t nBlocks = nBlk[0] * nBlk[1] * nBlk[2] * nBlk[3];

  // For texture storage, use more threads to better utilize GPU parallelism
  // Each thread can process multiple blocks with stride
  if (g->is_buffer_storage(input)) {
    return {nBlocks, 1u, 1u};
  } else {
    // For texture storage, use more workgroups to better utilize GPU
    // Aim for ~64-256 threads per workgroup for good occupancy
    uint32_t preferred_threads_per_wg = 64;
    uint32_t num_workgroups =
        (nBlocks + preferred_threads_per_wg - 1) / preferred_threads_per_wg;
    num_workgroups = std::max(1u, std::min(num_workgroups, nBlocks));
    return {num_workgroups * preferred_threads_per_wg, 1u, 1u};
  }
}

utils::uvec3 choose_qparams_block_wise_pick_local_wg_size(
    ComputeGraph* g,
    const vkapi::ShaderInfo&,
    const utils::uvec3& global_wg_size,
    const std::vector<ArgGroup>& a,
    const std::vector<ValueRef>&) {
  const ValueRef input = a.at(2).refs.at(0);

  if (g->is_buffer_storage(input)) {
    return {1u, 1u, 1u};
  } else {
    // For texture storage, use 64 threads per workgroup for better occupancy
    uint32_t local_size = std::min(64u, global_wg_size[0]);
    return {local_size, 1u, 1u};
  }
}

vkapi::ShaderInfo pick_choose_qparams_per_row_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;

  const ValueRef input = args.at(1).refs.at(0);

  // number of output channels
  const int64_t width = graph->size_at<int64_t>(-1, input);
  const int64_t height = graph->size_at<int64_t>(-2, input);

  std::string kernel_name = "choose_qparams_per_row";
  if (width > 256 || height == 1) {
    kernel_name += "_o1w64";
  } else {
    kernel_name += "_o4w16";
  }
  add_storage_type_suffix(kernel_name, graph->storage_type_of(input));
  add_dtype_suffix(kernel_name, graph->dtype_of(input));

  return VK_KERNEL_FROM_STR(kernel_name);
}

utils::uvec3 pick_choose_qparams_per_row_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef input = args.at(1).refs.at(0);
  const uint32_t height = graph->size_at<uint32_t>(-2, input);
  return {1u, height, 1u};
}

utils::uvec3 pick_choose_qparams_per_row_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;

  uint32_t outputs_per_wg = 1u;
  uint32_t workers_per_output = 64u;

  if (shader.kernel_name.find("o4w16") != std::string::npos) {
    outputs_per_wg = 4u;
    workers_per_output = 16u;
  }

  return {workers_per_output, outputs_per_wg, 1u};
}

void add_choose_qparams_tensor_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& quant_min,
    const ValueRef& quant_max,
    const ValueRef& eps,
    const ValueRef& scale_out,
    const ValueRef& zero_point_out) {
  std::string kernel_name("choose_qparams_tensor");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(scale_out));
  add_dtype_suffix(kernel_name, graph.dtype_of(zero_point_out));

  // Handle optional quant_min and quant_max parameters independently
  auto bounds = get_dtype_bounds(graph.dtype_of(zero_point_out));

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
  float eps_val = static_cast<float>(graph.get_double(eps));

  vkapi::ParamsBindList param_ubos;
  std::vector<PushConstantDataInfo> push_constants;

  if (graph.is_buffer_storage(input)) {
    param_ubos = {
        graph.sizes_ubo(input),
        graph.strides_ubo(input),
        graph.sizes_ubo(scale_out),
        graph.strides_ubo(scale_out),
        graph.sizes_ubo(zero_point_out),
        graph.strides_ubo(zero_point_out)};
  } else {
    param_ubos = {
        graph.logical_limits_ubo(input),
        graph.logical_limits_ubo(scale_out),
        graph.logical_limits_ubo(zero_point_out)};
  }

  push_constants = {
      PushConstantDataInfo(&quant_min_val, sizeof(int)),
      PushConstantDataInfo(&quant_max_val, sizeof(int)),
      PushConstantDataInfo(&eps_val, sizeof(float)),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      choose_qparams_pick_global_wg_size,
      choose_qparams_pick_local_wg_size,
      // Inputs and Outputs
      {{scale_out, vkapi::kWrite},
       {zero_point_out, vkapi::kWrite},
       {input, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

void add_choose_qparams_per_token_asymmetric_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& scale_out,
    const ValueRef& zero_point_out) {
  std::string kernel_name("choose_qparams_per_token_asymmetric");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(scale_out));
  add_dtype_suffix(kernel_name, graph.dtype_of(zero_point_out));

  // Calculate number of tokens (product of all dimensions except the last one)
  int64_t num_tokens = 1;
  const auto input_sizes = graph.sizes_of(input);
  for (size_t i = 0; i < input_sizes.size() - 1; i++) {
    num_tokens *= input_sizes[i];
  }

  int num_tokens_val = static_cast<int>(num_tokens);
  int quant_min_val = -128; // Fixed for asymmetric quantization
  int quant_max_val = 127; // Fixed for asymmetric quantization

  vkapi::ParamsBindList param_ubos;
  std::vector<PushConstantDataInfo> push_constants;

  if (graph.is_buffer_storage(input)) {
    param_ubos = {
        graph.sizes_ubo(input),
        graph.strides_ubo(input),
        graph.sizes_ubo(scale_out),
        graph.strides_ubo(scale_out),
        graph.sizes_ubo(zero_point_out),
        graph.strides_ubo(zero_point_out)};
  } else {
    param_ubos = {
        graph.logical_limits_ubo(input),
        graph.logical_limits_ubo(scale_out),
        graph.logical_limits_ubo(zero_point_out)};
  }

  push_constants = {
      PushConstantDataInfo(&num_tokens_val, sizeof(int)),
      PushConstantDataInfo(&quant_min_val, sizeof(int)),
      PushConstantDataInfo(&quant_max_val, sizeof(int)),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      choose_qparams_per_token_pick_global_wg_size,
      choose_qparams_per_token_pick_local_wg_size,
      // Inputs and Outputs
      {{scale_out, vkapi::kWrite},
       {zero_point_out, vkapi::kWrite},
       {input, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

void add_choose_qparams_per_row_node(
    ComputeGraph& graph,
    const ValueRef& input,
    const ValueRef& quant_min,
    const ValueRef& quant_max,
    const ValueRef& input_scales,
    const ValueRef& input_zps) {
  int32_t quant_min_val = -128;
  int32_t quant_max_val = 127;

  // Int8 range by default
  if (graph.val_is_none(quant_min)) {
    quant_min_val = -128;
  } else {
    quant_min_val = graph.extract_scalar<int32_t>(quant_min);
  }

  // Int8 range by default
  if (graph.val_is_none(quant_min)) {
    quant_max_val = 127;
  } else {
    quant_max_val = graph.extract_scalar<int32_t>(quant_max);
  }

  vkapi::ParamsBindList param_ubos = {
      graph.sizes_ubo(input),
  };
  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&quant_min_val, sizeof(int32_t)),
      PushConstantDataInfo(&quant_max_val, sizeof(int32_t)),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_choose_qparams_per_row_shader,
      pick_choose_qparams_per_row_global_wg_size,
      pick_choose_qparams_per_row_local_wg_size,
      // Inputs and Outputs
      {{{input_scales, input_zps}, vkapi::kWrite}, {input, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_choose_qparams_per_row));
}

void add_choose_qparams_block_wise_node(
    ComputeGraph& graph,
    ValueRef input,
    ValueRef block_size,
    int mapping_type, // 0 / 1 / 2
    ValueRef quant_min,
    ValueRef quant_max,
    ValueRef eps,
    ValueRef scale_out,
    ValueRef zp_out) {
  const auto input_sizes = graph.sizes_of(input);
  const auto block_size_list = graph.get_int_list(block_size);

  // For shader compatibility, we still need to convert to WHCN order
  // but the output shape calculation is now handled correctly in resize
  // function
  utils::ivec4 block_size_vec = utils::make_whcn_ivec4(*block_size_list);
  utils::ivec4 tensor_size_whcn = utils::make_whcn_ivec4(input_sizes);

  // Calculate numBlocks: ceil(tensorSize / blockSize) (both in WHCN order)
  utils::ivec4 num_blocks_vec = {
      (tensor_size_whcn[0] + block_size_vec[0] - 1) / block_size_vec[0],
      (tensor_size_whcn[1] + block_size_vec[1] - 1) / block_size_vec[1],
      (tensor_size_whcn[2] + block_size_vec[2] - 1) / block_size_vec[2],
      (tensor_size_whcn[3] + block_size_vec[3] - 1) / block_size_vec[3]};

  // Calculate blockStride: pre-computed linear strides for the block grid
  utils::ivec4 block_stride_vec = {
      1,
      num_blocks_vec[0],
      num_blocks_vec[0] * num_blocks_vec[1],
      num_blocks_vec[0] * num_blocks_vec[1] * num_blocks_vec[2]};

  // Handle optional quant_min and quant_max parameters
  int qmin, qmax;
  if (graph.val_is_none(quant_min) || graph.val_is_none(quant_max)) {
    // Use default values based on target_dtype (similar to
    // _get_and_check_qmin_qmax) For now, assume int8 range as default - this
    // should match the Python implementation
    qmin = -128;
    qmax = 127;
  } else {
    qmin = static_cast<int>(graph.get_int(quant_min));
    qmax = static_cast<int>(graph.get_int(quant_max));
  }

  float eps_val;
  if (graph.val_is_none(eps)) {
    // Use default eps value (similar to Python implementation)
    eps_val = 1.192092896e-07f; // torch.finfo(torch.float32).eps
  } else {
    eps_val = static_cast<float>(graph.get_double(eps));
  }

  // Create push constants vector
  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&block_size_vec, sizeof(block_size_vec)),
      PushConstantDataInfo(&num_blocks_vec, sizeof(num_blocks_vec)),
      PushConstantDataInfo(&block_stride_vec, sizeof(block_stride_vec)),
      PushConstantDataInfo(&mapping_type, sizeof(int)),
      PushConstantDataInfo(&qmin, sizeof(int)),
      PushConstantDataInfo(&qmax, sizeof(int)),
      PushConstantDataInfo(&eps_val, sizeof(float))};

  std::string kernel_name("choose_qparams_block_wise");
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(scale_out));
  add_dtype_suffix(kernel_name, graph.dtype_of(zp_out));

  vkapi::ParamsBindList param_ubos;

  if (graph.is_buffer_storage(input)) {
    param_ubos = {
        graph.sizes_ubo(input),
        graph.strides_ubo(input),
        graph.sizes_ubo(scale_out),
        graph.strides_ubo(scale_out),
        graph.sizes_ubo(zp_out),
        graph.strides_ubo(zp_out)};
  } else {
    // For texture input, the shader uses buffer storage for outputs
    // so we need buffer UBOs for the output tensors
    param_ubos = {
        graph.logical_limits_ubo(input),
        graph.sizes_ubo(scale_out),
        graph.strides_ubo(scale_out),
        graph.sizes_ubo(zp_out),
        graph.strides_ubo(zp_out)};
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      choose_qparams_block_wise_pick_global_wg_size,
      choose_qparams_block_wise_pick_local_wg_size,
      // Inputs and Outputs
      {{scale_out, vkapi::kWrite},
       {zp_out, vkapi::kWrite},
       {input, vkapi::kRead}},
      // Shader param buffers
      param_ubos,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize Args
      {block_size},
      // Resizing Logic
      nullptr));
}

void choose_qparams_tensor_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef eps = args[arg_idx++];
  const ValueRef dtype = args[arg_idx++];
  const ValueRef out_tuple_ref = args[arg_idx++];

  ValueRef scale_out = kDummyValueRef;
  ValueRef zero_point_out = kDummyValueRef;

  {
    const ValueListPtr out_tuple = graph.get_value_list(out_tuple_ref);
    scale_out = out_tuple->at(0);
    zero_point_out = out_tuple->at(1);
  }

  // Void the unused dtype parameter to match ATen signature
  (void)dtype;

  // Check tensor types
  VK_CHECK_COND(graph.val_is_tensor(input));
  VK_CHECK_COND(graph.val_is_tensor(scale_out));
  VK_CHECK_COND(graph.val_is_tensor(zero_point_out));

  // Verify input is a floating point type
  VK_CHECK_COND(graph.dtype_of(input) == vkapi::kFloat);

  // Get scale and zero point output dtypes
  vkapi::ScalarType scale_out_dtype = graph.dtype_of(scale_out);
  vkapi::ScalarType zero_point_out_dtype = graph.dtype_of(zero_point_out);

  // Verify supported output types for scale (fp32 only for now)
  VK_CHECK_COND(scale_out_dtype == vkapi::kFloat);

  // Verify supported output types for zero point (int32, int8, fp32)
  VK_CHECK_COND(
      zero_point_out_dtype == vkapi::kInt ||
      zero_point_out_dtype == vkapi::kChar ||
      zero_point_out_dtype == vkapi::kFloat);

  // Check that texture storage is width packed
  if (!graph.is_buffer_storage(input)) {
    VK_CHECK_COND(graph.packed_dim_of(input) == WHCN::kWidthDim);
  }

  add_choose_qparams_tensor_node(
      graph, input, quant_min, quant_max, eps, scale_out, zero_point_out);
}

void choose_qparams_per_token_asymmetric_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef dtype = args[arg_idx++];
  const ValueRef out_tuple_ref = args[arg_idx++];

  ValueRef scale_out = kDummyValueRef;
  ValueRef zero_point_out = kDummyValueRef;

  {
    const ValueListPtr out_tuple = graph.get_value_list(out_tuple_ref);
    scale_out = out_tuple->at(0);
    zero_point_out = out_tuple->at(1);
  }

  // Void the unused parameter to match ATen signature
  (void)dtype;

  // Check tensor types
  VK_CHECK_COND(graph.val_is_tensor(input));
  VK_CHECK_COND(graph.val_is_tensor(scale_out));
  VK_CHECK_COND(graph.val_is_tensor(zero_point_out));

  // Verify input is a floating point type
  VK_CHECK_COND(graph.dtype_of(input) == vkapi::kFloat);

  // Get scale and zero point output dtypes
  vkapi::ScalarType scale_out_dtype = graph.dtype_of(scale_out);
  vkapi::ScalarType zero_point_out_dtype = graph.dtype_of(zero_point_out);

  // Verify supported output types for scale (fp32 only for now)
  VK_CHECK_COND(scale_out_dtype == vkapi::kFloat);

  // Verify supported output types for zero point (int32, int8, fp32)
  VK_CHECK_COND(
      zero_point_out_dtype == vkapi::kInt ||
      zero_point_out_dtype == vkapi::kChar ||
      zero_point_out_dtype == vkapi::kFloat);

  // Check that texture storage is width packed
  if (!graph.is_buffer_storage(input)) {
    VK_CHECK_COND(graph.packed_dim_of(input) == WHCN::kWidthDim);
  }

  add_choose_qparams_per_token_asymmetric_node(
      graph, input, scale_out, zero_point_out);
}

bool can_use_choose_qparams_per_row(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef block_size,
    const ValueRef input_zero_point) {
  if (!graph.is_vectorizable_contiguous_2d_matrix(input)) {
    return false;
  }

  std::vector<int64_t> input_sizes = graph.sizes_of(input);
  const IntListPtr block_size_vals = graph.get_int_list(block_size);
  const size_t ndim = block_size_vals->size();

  // Check for per y - dim quantization
  if (utils::val_at(-1, input_sizes) != utils::val_at(-1, *block_size_vals)) {
    return false;
  }

  for (int d = 0; d < ndim - 1; ++d) {
    if (block_size_vals->at(d) != 1) {
      return false;
    }
  }
  return true;
}

void choose_qparams_affine_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef mapping_type = args[arg_idx++];
  const ValueRef block_size = args[arg_idx++];
  const ValueRef target_dtype = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef eps = args[arg_idx++];
  const ValueRef scale_dtype = args[arg_idx++];
  const ValueRef zero_point_dtype = args[arg_idx++];
  const ValueRef out_tuple_ref = args[arg_idx++];

  // Suppress unused variable warnings
  (void)target_dtype;
  (void)scale_dtype;
  (void)zero_point_dtype;

  ValueRef scale_out = kDummyValueRef;
  ValueRef zero_point_out = kDummyValueRef;

  {
    const ValueListPtr out_tuple = graph.get_value_list(out_tuple_ref);
    scale_out = out_tuple->at(0);
    zero_point_out = out_tuple->at(1);
  }

  // Use fast path if certain conditions are met
  if (can_use_choose_qparams_per_row(
          graph, input, block_size, zero_point_out)) {
    return add_choose_qparams_per_row_node(
        graph, input, quant_min, quant_max, scale_out, zero_point_out);
  }

  // Check tensor types
  VK_CHECK_COND(graph.val_is_tensor(input));
  VK_CHECK_COND(graph.val_is_tensor(scale_out));
  VK_CHECK_COND(graph.val_is_tensor(zero_point_out));

  // Verify input is a floating point type
  VK_CHECK_COND(graph.dtype_of(input) == vkapi::kFloat);

  // Get scale and zero point dtypes from arguments
  vkapi::ScalarType scale_out_dtype = graph.dtype_of(scale_out);
  vkapi::ScalarType zero_point_out_dtype = graph.dtype_of(zero_point_out);

  // Verify supported output types for scale (fp32 only for now)
  VK_CHECK_COND(scale_out_dtype == vkapi::kFloat);

  // Verify supported output types for zero point (int32, int8, fp32)
  VK_CHECK_COND(
      zero_point_out_dtype == vkapi::kInt ||
      zero_point_out_dtype == vkapi::kChar ||
      zero_point_out_dtype == vkapi::kFloat);

  // Check that texture storage is width packed
  if (!graph.is_buffer_storage(input)) {
    VK_CHECK_COND(graph.packed_dim_of(input) == WHCN::kWidthDim);
  }

  const auto input_sizes = graph.sizes_of(input);
  const auto block_size_list = graph.get_int_list(block_size);
  VK_CHECK_COND(block_size_list->size() == input_sizes.size());

  std::string mapping_type_str = graph.get_string(mapping_type);
  int mapping_type_val = 0; // Default to ASYMMETRIC

  if (mapping_type_str == "ASYMMETRIC" || mapping_type_str.empty()) {
    mapping_type_val = 0; // ASYMMETRIC
  } else if (mapping_type_str == "SYMMETRIC") {
    mapping_type_val = 1;
  } else if (mapping_type_str == "SYMMETRIC_NO_CLIPPING_ERR") {
    mapping_type_val = 2;
  } else {
    VK_THROW("Unsupported mapping_type: ", mapping_type_str);
  }

  add_choose_qparams_block_wise_node(
      graph,
      input,
      block_size,
      mapping_type_val,
      quant_min,
      quant_max,
      eps,
      scale_out,
      zero_point_out);
}

void choose_qparams_per_row(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int arg_idx = 0;
  const ValueRef input = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  const ValueRef quant_max = args[arg_idx++];
  const ValueRef input_scales = args[arg_idx++];
  const ValueRef input_zps = args[arg_idx++];

  //   ValueRef scale_out = kDummyValueRef;
  //   ValueRef zero_point_out = kDummyValueRef;
  //
  //   {
  //     const ValueListPtr out_tuple = graph.get_value_list(out_tuple_ref);
  //     scale_out = out_tuple->at(0);
  //     zero_point_out = out_tuple->at(1);
  //   }
  //

  add_choose_qparams_per_row_node(
      graph, input, quant_min, quant_max, input_scales, input_zps);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(
      quantized_decomposed.choose_qparams.tensor, choose_qparams_tensor_impl);
  VK_REGISTER_OP(
      quantized_decomposed.choose_qparams_per_token_asymmetric.default,
      choose_qparams_per_token_asymmetric_impl);

  // Register the per-channel quantization operator
  VK_REGISTER_OP(etvk.choose_qparams_per_row.default, choose_qparams_per_row);

  // TorchAO affine choose_qparams operators
  VK_REGISTER_OP(
      torchao.choose_qparams_affine.default, choose_qparams_affine_impl);
}

} // namespace vkcompute
