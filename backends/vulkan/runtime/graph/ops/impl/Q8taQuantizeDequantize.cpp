/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taQuantizeDequantize.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void add_q8ta_quantize_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef packed_int8_output) {
  float inv_scale = 1.0f / graph.extract_scalar<float>(input_scale);
  int32_t zp = graph.extract_scalar<int32_t>(input_zp);

  // Detect input storage type to select appropriate shader variant
  utils::StorageType inp_storage = graph.storage_type_of(fp_input);

  // Build shader name: q8ta_quantize_{buffer|texture3d}_{dtype}
  std::string kernel_name = "q8ta_quantize";
  add_storage_type_suffix(kernel_name, inp_storage);
  add_dtype_suffix(kernel_name, graph.dtype_of(fp_input));

  // Pass metadata for both output and input tensors
  // Output is always buffer, input can be buffer or texture
  vkapi::ParamsBindList param_buffers;
  param_buffers.append(graph.buffer_meta_ubo(packed_int8_output));
  param_buffers.append(graph.meta_ubo(fp_input));

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&inv_scale, sizeof(inv_scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  // Create block config for output tensor: inner_dim = output's packed_dim
  const BlockConfig outp_block_config = create_block_config_from_io_packed_dims(
      graph, packed_int8_output, fp_input);

  // Create block config for input tensor: based on outp_block_config but with
  // inner_dim = input's packed_dim. If input and output have different packed
  // dims, the block axes are transposed.
  const BlockConfig inp_block_config =
      create_block_config_from_other(graph, fp_input, outp_block_config);

  // Cast block config to ValueRef for pick_*_global_wg_with_block_config
  // Use inp_block_config since shader uses inp_block_config for indexing
  const ValueRef block_config_ref =
      static_cast<ValueRef>(inp_block_config.as_packed_int());

  // Choose dispatch function based on FP input storage type:
  // - Buffer: use linear dispatch (better performance)
  // - Texture: use extents-style 3D dispatch (better performance)
  auto pick_global_wg_size = (inp_storage == utils::kBuffer)
      ? pick_linear_global_wg_with_block_config
      : pick_extents_global_wg_with_block_config;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_global_wg_size,
      pick_square_local_wg_with_block_config,
      // Inputs and Outputs
      {{packed_int8_output, vkapi::kWrite}, {fp_input, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {graph.hashed_layout_of(fp_input),
       graph.hashed_layout_of(packed_int8_output),
       inp_block_config.as_packed_int(),
       outp_block_config.as_packed_int()},
      // Resize args
      {block_config_ref}));
}

void add_q8ta_dequantize_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef fp_output) {
  float scale = graph.extract_scalar<float>(output_scale);
  int32_t zp = graph.extract_scalar<int32_t>(output_zp);

  // Detect output storage type to select appropriate shader variant
  utils::StorageType outp_storage = graph.storage_type_of(fp_output);

  // Build shader name: q8ta_dequantize_{buffer|texture3d}_{dtype}
  std::string kernel_name = "q8ta_dequantize";
  add_storage_type_suffix(kernel_name, outp_storage);
  add_dtype_suffix(kernel_name, graph.dtype_of(fp_output));

  // Pass metadata for both output and input tensors
  // Output can be buffer or texture, input is always buffer
  vkapi::ParamsBindList param_buffers;
  param_buffers.append(graph.meta_ubo(fp_output));
  param_buffers.append(graph.buffer_meta_ubo(packed_int8_input));

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&scale, sizeof(scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  // Create block config for output tensor: inner_dim = output's packed_dim
  const BlockConfig outp_block_config = create_block_config_from_io_packed_dims(
      graph, fp_output, packed_int8_input);

  // Create block config for input tensor: based on outp_block_config but with
  // inner_dim = input's packed_dim. If input and output have different packed
  // dims, the block axes are transposed.
  const BlockConfig inp_block_config = create_block_config_from_other(
      graph, packed_int8_input, outp_block_config);

  // Cast block config to ValueRef for pick_*_global_wg_with_block_config
  // Use inp_block_config since shader uses inp_block_config for indexing
  const ValueRef block_config_ref =
      static_cast<ValueRef>(inp_block_config.as_packed_int());

  // Choose dispatch function based on FP output storage type:
  // - Buffer: use linear dispatch (better performance)
  // - Texture: use extents-style 3D dispatch (better performance)
  auto pick_global_wg_size = (outp_storage == utils::kBuffer)
      ? pick_linear_global_wg_with_block_config
      : pick_extents_global_wg_with_block_config;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_global_wg_size,
      pick_square_local_wg_with_block_config,
      // Inputs and Outputs
      {{fp_output, vkapi::kWrite}, {packed_int8_input, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {graph.hashed_layout_of(fp_output),
       graph.hashed_layout_of(packed_int8_input),
       outp_block_config.as_packed_int(),
       inp_block_config.as_packed_int()},
      // Resize args
      {block_config_ref}));
}

} // namespace vkcompute
