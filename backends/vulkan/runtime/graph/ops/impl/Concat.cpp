/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Clone.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

std::vector<int64_t> get_concat_sizes(
    ComputeGraph& graph,
    ValueRef all_input_refs,
    const int64_t concat_dim) {
  ValueListPtr in_value_refs = graph.get_value_list(all_input_refs);
  // Get the sizes of the first input tensor as a starting point
  std::vector<int64_t> new_out_sizes = graph.sizes_of(in_value_refs->at(0));

  // Sum up the sizes along the concatenation dimension
  for (size_t i = 1; i < in_value_refs->size(); ++i) {
    const std::vector<int64_t> in_sizes = graph.sizes_of(in_value_refs->at(i));
    new_out_sizes.at(concat_dim) += in_sizes.at(concat_dim);
  }

  return new_out_sizes;
}

void resize_concat_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef all_inputs = extra_args.at(0);

  int64_t concat_dim = graph->extract_scalar<int64_t>(extra_args.at(1));

  // Normalize concat_dim if negative
  const int64_t ndim = graph->dim_of(out);
  if (concat_dim < 0) {
    concat_dim += ndim;
  }

  // Calculate the new sizes
  std::vector<int64_t> new_out_sizes =
      get_concat_sizes(*graph, all_inputs, concat_dim);

  // Resize the output tensor
  graph->virtual_resize(out, new_out_sizes);
}

utils::uvec3 concat_pick_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)shader;
  (void)extra_args;

  const ValueRef out = args.at(0).refs.at(0);
  const std::vector<ValueRef> inputs_in_batch = args.at(1).refs;

  int64_t concat_dim = graph->extract_scalar<int64_t>(extra_args.at(1));

  // Normalize concat_dim if negative
  const int64_t ndim = graph->dim_of(out);
  if (concat_dim < 0) {
    concat_dim += ndim;
  }

  // The concat shader concatenates N input tensors at a time to the output
  // tensor. Since the shader may need to be invoked multiple times to finish
  // concatenation when the number of input tensors is >N, the global workgroup
  // is based on the volume of input data being concatenated in this batch,
  // as opposed to the overall size of the output tensor. Conceptually, the
  // global work group size represents which elements of the output tensor will
  // be written to during this dispatch.

  uint32_t total_input_numel = 0;
  int64_t concat_dim_numel = 0;
  for (const ValueRef input : inputs_in_batch) {
    total_input_numel += graph->numel_of(input);
    concat_dim_numel += graph->size_at<int64_t>(concat_dim, input);
  }

  if (graph->is_buffer_storage(out)) {
    return {total_input_numel, 1, 1};
  }

  // The texture implementation is similar, except each invocation writes out 4
  // output elements along the packed dim (i.e. one texel). In this case, the
  // global work group size represents the number of output texels that will be
  // written to in this batch, rather than the number of output elements. Note
  // that to update an element of the output, the entire texel that contains it
  // will need to be loaded, updated, then written back.

  std::vector<int64_t> inp_volume_sizes = graph->sizes_of(out);
  inp_volume_sizes.at(concat_dim) = concat_dim_numel;

  // Calculate what the image extents would be of a tensor with the input
  // volume's sizes. This produces the number of texels that would need to be
  // written to.
  const api::PackedDimInfo& packed_dim_info = graph->packed_dim_info_of(out);
  std::vector<int64_t> inp_volume_texel_sizes =
      api::calculate_padded_sizes(inp_volume_sizes, packed_dim_info);
  // If the concat_dim is the same as the packed dim, and the concat_offset for
  // this input batch is not a multiple of 4, then the data from an input texel
  // may be split up between two output texels. For example:
  //                I0 , I1 , I2 , I2
  // O0 , O1 , O2 , X  | X  , X  , X ,  X
  // Therefore, 1 texel is added to the packed dim to account for this.
  inp_volume_texel_sizes.at(3 - packed_dim_info.packed_dim) =
      utils::div_up_4(
          inp_volume_texel_sizes.at(3 - packed_dim_info.packed_dim)) +
      1;

  const uint32_t inp_volume_texel_numel =
      utils::multiply_integers(inp_volume_texel_sizes);

  return {inp_volume_texel_numel, 1, 1};

  // The texture implementation is similar, expect each thread is responsible
  // for writing out an entire output texel. Therefore, the overall global work
  // group size will be the concatenation of the texture extents of the input
  // tensors in this batch.

  // One complication is when the previous concatenation batch does not write
  // up to a texel boundary. An example is if the previous concatenation batch
  // only wrote 7 elements along the concatenation dim. The first input element
  // would then have to be inserted at the last element of the final texel
  // written by the previous batch. To account for this, initialize the
  // workgroup size at the concatenation dim to 1 (need to read N total texels
  // along the concat dim for input tensors + up to 1 texel from the output
  // tensor).

  // The axis along which to concatenate the input texture extents
  int64_t extent_concat_axis = nchw_dim_to_whcn_dim(concat_dim, ndim);
  // For batch concatenation, the concat axis is the batch-concatenation axis
  if (concat_dim == 4) {
    extent_concat_axis = graph->concat_dim_of(out);
  }

  utils::uvec3 global_workgroup_size = graph->create_global_wg_size(out);
  global_workgroup_size[concat_dim] = 0;
  for (const ValueRef input : inputs_in_batch) {
    utils::uvec3 texture_extents = graph->logical_limits_of(input);
    global_workgroup_size[extent_concat_axis] += texture_extents[concat_dim];
  }

  return global_workgroup_size;
}

void add_concat_node(
    ComputeGraph& graph,
    const ValueRef tensors_ref,
    const ValueRef dim_ref,
    const ValueRef out) {
  std::vector<ValueRef> in_value_refs;

  {
    const ValueListPtr tensors = graph.get_value_list(tensors_ref);

    for (const ValueRef in : *tensors) {
      in_value_refs.push_back(in);
    }
  }

  const int64_t dim = graph.extract_scalar<int64_t>(dim_ref);

  const int64_t ndim = graph.dim_of(in_value_refs.at(0));
  int64_t normalized_dim = dim;
  if (normalized_dim < 0) {
    normalized_dim += ndim;
  }

  const int64_t dim_whcn = nchw_dim_to_whcn_dim(normalized_dim, ndim);
  const ValueRef dim_whcn_ref = graph.get_or_add_value_for_int(dim_whcn);

  // Create a temporary tensor to hold the concat offset
  TmpTensor concat_offset(
      &graph, {1}, vkapi::kInt, utils::kBuffer, utils::kWidthPacked);

  // Add node to set concat_offset to 0
  {
    std::string kernel_name = "set_zero";
    add_dtype_suffix(kernel_name, graph.dtype_of(concat_offset));

    vkapi::ParamsBindList param_buffers = {graph.numel_ubo(concat_offset)};

    graph.execute_nodes().emplace_back(new DispatchNode(
        graph,
        VK_KERNEL_FROM_STR(kernel_name),
        {1, 1, 1},
        {1, 1, 1},
        // Inputs and Outputs
        {{concat_offset, vkapi::kWrite}},
        // Parameter buffers
        param_buffers,
        // Push Constants
        {},
        // Specialization Constants
        {},
        // Resize Args
        {},
        // Resizing Logic
        nullptr));
  }

  // Process inputs in batches of up to 3 tensors
  const size_t batch_size = 3;
  for (size_t batch_start = 0; batch_start < in_value_refs.size();
       batch_start += batch_size) {
    const size_t batch_end =
        std::min(batch_start + batch_size, in_value_refs.size());
    const size_t current_batch_size = batch_end - batch_start;

    std::vector<ValueRef> batch_inputs;
    for (size_t i = batch_start; i < batch_end; ++i) {
      batch_inputs.push_back(in_value_refs.at(i));
    }

    // Add concat node for this batch
    {
      vkapi::ParamsBindList param_buffers = {
          graph.get_or_create_int_param_buffer(dim_whcn_ref, 0)};

      std::vector<PushConstantDataInfo> push_constants;
      vkapi::SpecVarList spec_vars;

      if (graph.is_buffer_storage(out)) {
        param_buffers.append(graph.sizes_ubo(out));
        param_buffers.append(graph.strides_ubo(out));

        for (const ValueRef in_ref : batch_inputs) {
          param_buffers.append(graph.sizes_ubo(in_ref));
          param_buffers.append(graph.strides_ubo(in_ref));
        }

        param_buffers.append(graph.numel_ubo(out));

        spec_vars = {graph.hashed_layout_of(out)};
      } else {
        push_constants = {graph.sizes_pc_of(out)};

        spec_vars = {graph.hashed_layout_of(out)};

        for (const ValueRef in_ref : batch_inputs) {
          push_constants.push_back(graph.sizes_pc_of(in_ref));
          spec_vars.append(graph.hashed_layout_of(in_ref));
        }
      }

      std::string kernel_name = "concat";
      if (current_batch_size == 1) {
        kernel_name += "_1";
      } else if (current_batch_size == 2) {
        kernel_name += "_2";
      } else if (current_batch_size == 3) {
        kernel_name += "_3";
      }
      if (graph.is_buffer_storage(out)) {
        kernel_name += "_buffer";
      } else {
        kernel_name += "_texture3d";
      }

      add_dtype_suffix(kernel_name, graph.dtype_of(out));

      DispatchNode::ResizeFunction resize_fn = nullptr;
      if (batch_start == 0) {
        resize_fn = resize_concat_node;
      }
      graph.execute_nodes().emplace_back(new DynamicDispatchNode(
          graph,
          VK_KERNEL_FROM_STR(kernel_name),
          concat_pick_global_wg_size,
          default_pick_local_wg_size,
          // Inputs and Outputs
          {{out, vkapi::kReadWrite},
           {batch_inputs, vkapi::kRead},
           {concat_offset, vkapi::kRead}},
          // Parameter buffers
          param_buffers,
          // Push Constants
          push_constants,
          // Specialization Constants
          spec_vars,
          // Resize Args
          {tensors_ref, dim_ref},
          // Resizing Logic
          resize_fn));
    }

    // Add node to update concat_offset (except for the last batch)
    if (batch_end < in_value_refs.size()) {
      vkapi::ParamsBindList param_buffers = {
          graph.get_or_create_int_param_buffer(dim_whcn_ref, 0)};

      for (const ValueRef in_ref : batch_inputs) {
        param_buffers.append(graph.sizes_ubo(in_ref));
      }

      std::string kernel_name = "update_concat_offset";
      if (current_batch_size == 1) {
        kernel_name += "_1";
      } else if (current_batch_size == 2) {
        kernel_name += "_2";
      } else if (current_batch_size == 3) {
        kernel_name += "_3";
      }
      add_dtype_suffix(kernel_name, graph.dtype_of(concat_offset));

      vkapi::SpecVarList spec_vars = {};

      graph.execute_nodes().emplace_back(new DispatchNode(
          graph,
          VK_KERNEL_FROM_STR(kernel_name),
          {1u, 1u, 1u},
          {1u, 1u, 1u},
          // Inputs and Outputs
          {{concat_offset, vkapi::kWrite}},
          // Parameter buffers
          param_buffers,
          // Push Constants
          {},
          // Specialization Constants
          spec_vars,
          // Resize Args
          {},
          // Resizing Logic
          nullptr));
    }
  }
}

void cat_tensor(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // Extract arguments
  const ValueRef tensors_ref = args.at(0);
  const ValueRef dim_ref = args.at(1);
  const ValueRef out = args.at(2);

  // Add concat node
  add_concat_node(graph, tensors_ref, dim_ref, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.cat.default, cat_tensor);
}

} // namespace vkcompute
