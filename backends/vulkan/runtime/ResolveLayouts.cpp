/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/ResolveLayouts.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <string>

namespace vkcompute {

namespace {

using VkGraphPtr = const vkgraph::VkGraph*;
using OpCallPtr = const vkgraph::OperatorCall*;
using VkValuePtr = const vkgraph::VkValue*;
using VkTensorPtr = const vkgraph::VkTensor*;
using UIntVector = const flatbuffers::Vector<uint32_t>*;

bool is_dynamic_layout(const vkgraph::VkMemoryLayout layout) {
  return layout == vkgraph::VkMemoryLayout::PACKED_INT8_CONV2D;
}

bool is_packed_int8_layout(vkgraph::VkMemoryLayout layout) {
  switch (layout) {
    case vkgraph::VkMemoryLayout::PACKED_INT8_4W4C:
    case vkgraph::VkMemoryLayout::PACKED_INT8_4H4W:
    case vkgraph::VkMemoryLayout::PACKED_INT8_4W:
    case vkgraph::VkMemoryLayout::PACKED_INT8_4C:
    case vkgraph::VkMemoryLayout::PACKED_INT8_4C1W:
      return true;
    default:
      return false;
  }
}

vkgraph::VkMemoryLayout get_resolved_layout(
    uint32_t fb_id,
    VkGraphPtr flatbuffer,
    const std::unordered_map<uint32_t, vkgraph::VkMemoryLayout>&
        memory_layout_overrides) {
  auto it = memory_layout_overrides.find(fb_id);
  if (it != memory_layout_overrides.end()) {
    return it->second;
  }
  VkValuePtr value = flatbuffer->values()->Get(fb_id);
  if (value->value_type() != vkgraph::GraphTypes::VkTensor) {
    return vkgraph::VkMemoryLayout::DEFAULT_LAYOUT;
  }
  return value->value_as_VkTensor()->memory_layout();
}

void resolve_dynamic_args(
    VkGraphPtr flatbuffer,
    OpCallPtr op_call,
    std::unordered_map<uint32_t, vkgraph::VkMemoryLayout>&
        memory_layout_overrides) {
  // Find the first arg tensor with a non-dynamic packed int8 layout
  vkgraph::VkMemoryLayout resolved_layout =
      vkgraph::VkMemoryLayout::DEFAULT_LAYOUT;
  bool found = false;
  for (int i = 0; i < op_call->args()->size(); ++i) {
    const uint32_t fb_id = static_cast<uint32_t>(op_call->args()->Get(i));
    VkValuePtr value = flatbuffer->values()->Get(fb_id);
    if (value->value_type() != vkgraph::GraphTypes::VkTensor) {
      continue;
    }
    vkgraph::VkMemoryLayout layout =
        get_resolved_layout(fb_id, flatbuffer, memory_layout_overrides);
    if (is_packed_int8_layout(layout)) {
      resolved_layout = layout;
      found = true;
      break;
    }
  }

  if (!found) {
    return;
  }

  // Override all args whose resolved layout is still dynamic
  for (int i = 0; i < op_call->args()->size(); ++i) {
    const uint32_t fb_id = static_cast<uint32_t>(op_call->args()->Get(i));
    vkgraph::VkMemoryLayout layout =
        get_resolved_layout(fb_id, flatbuffer, memory_layout_overrides);
    if (is_dynamic_layout(layout)) {
      memory_layout_overrides[fb_id] = resolved_layout;
    }
  }
}

void resolve_q8ta_conv2d(
    VkGraphPtr flatbuffer,
    OpCallPtr op_call,
    ComputeGraph* compute_graph,
    std::unordered_map<uint32_t, vkgraph::VkMemoryLayout>&
        memory_layout_overrides) {
  // q8ta_conv2d args layout:
  //   0: input, 1: input_scale, 2: input_zp, 3: weight, 4: weight_sums,
  //   5: weight_scales, 6: output_scale, 7: output_zp, 8: bias,
  //   9: kernel_size, 10: stride, 11: padding, 12: dilation, 13: groups,
  //   14: activation, 15: output

  const uint32_t input_fb_id = static_cast<uint32_t>(op_call->args()->Get(0));
  const uint32_t groups_fb_id = static_cast<uint32_t>(op_call->args()->Get(13));
  const uint32_t output_fb_id = static_cast<uint32_t>(op_call->args()->Get(15));

  // Only resolve if the input tensor has a dynamic layout
  VkTensorPtr input_tensor =
      flatbuffer->values()->Get(input_fb_id)->value_as_VkTensor();
  if (!is_dynamic_layout(input_tensor->memory_layout())) {
    return;
  }

  // Extract groups value
  VkValuePtr groups_value = flatbuffer->values()->Get(groups_fb_id);
  const int64_t groups = groups_value->value_as_Int()->int_val();

  // Extract input tensor dimensions
  UIntVector input_dims = input_tensor->dims();
  const int64_t input_ndim = input_dims->size();
  const int64_t in_channels = input_dims->Get(input_ndim - 3);
  const int64_t in_channels_per_group = in_channels / groups;

  // Extract output tensor dimensions
  VkTensorPtr output_tensor =
      flatbuffer->values()->Get(output_fb_id)->value_as_VkTensor();
  UIntVector output_dims = output_tensor->dims();
  const int64_t output_ndim = output_dims->size();
  const int64_t H_out = output_dims->Get(output_ndim - 2);
  const int64_t W_out = output_dims->Get(output_ndim - 1);
  const int64_t spatial_out = H_out * W_out;

  // Replicate the im2col decision logic from Q8taConv2d.cpp
  const bool im2col_eligible = in_channels_per_group % 4 == 0;

  bool use_im2col = false;
  if (compute_graph->device_is_mali()) {
    use_im2col = im2col_eligible;
  } else {
    use_im2col = im2col_eligible && groups == 1 &&
        (in_channels_per_group >= 32 || spatial_out <= 4096);
  }

  if (use_im2col) {
    memory_layout_overrides[input_fb_id] =
        vkgraph::VkMemoryLayout::PACKED_INT8_4C;
  } else {
    memory_layout_overrides[input_fb_id] =
        vkgraph::VkMemoryLayout::PACKED_INT8_4C1W;
  }
}

void resolve_q8ta_conv2d_dw(
    VkGraphPtr flatbuffer,
    OpCallPtr op_call,
    std::unordered_map<uint32_t, vkgraph::VkMemoryLayout>&
        memory_layout_overrides) {
  const uint32_t input_fb_id = static_cast<uint32_t>(op_call->args()->Get(0));

  // Only override if not already overridden by a previous op
  if (memory_layout_overrides.count(input_fb_id) > 0) {
    return;
  }

  // Only resolve if the input tensor has a dynamic layout
  VkTensorPtr input_tensor =
      flatbuffer->values()->Get(input_fb_id)->value_as_VkTensor();
  if (!is_dynamic_layout(input_tensor->memory_layout())) {
    return;
  }

  memory_layout_overrides[input_fb_id] =
      vkgraph::VkMemoryLayout::PACKED_INT8_4C1W;
}

} // namespace

void resolve_memory_layouts(
    const vkgraph::VkGraph* flatbuffer,
    ComputeGraph* compute_graph,
    std::unordered_map<uint32_t, vkgraph::VkMemoryLayout>&
        memory_layout_overrides) {
  // First, handle ops where input memory layout is impactful for performance
  for (const auto* op_call : *(flatbuffer->chain())) {
    const std::string op_name = op_call->name()->str();

    if (op_name == "et_vk.q8ta_conv2d.default") {
      resolve_q8ta_conv2d(
          flatbuffer, op_call, compute_graph, memory_layout_overrides);
    } else if (op_name == "et_vk.q8ta_conv2d_dw.default") {
      resolve_q8ta_conv2d_dw(flatbuffer, op_call, memory_layout_overrides);
    }
  }
  // Then, try to ensure ops use the same memory layout whenever possible.
  for (const auto* op_call : *(flatbuffer->chain())) {
    resolve_dynamic_args(flatbuffer, op_call, memory_layout_overrides);
  }
}

} // namespace vkcompute
