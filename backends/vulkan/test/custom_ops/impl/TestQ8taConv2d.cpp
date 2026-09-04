/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taClone.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taConv2d.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taQuantizeDequantize.h>

namespace vkcompute {

namespace {

void assert_im2col_kernel_selection(
    ComputeGraph& graph,
    const bool expect_unsigned,
    const bool expect_buffer_weights) {
  const vkapi::Adapter* const adapter = graph.context()->adapter_ptr();
  std::string expected_execute;
  std::string expected_prepack;
  if (expect_unsigned) {
    expected_execute = expect_buffer_weights
        ? "q8ta_conv2d_pw_unsigned_buffer_float"
        : "q8ta_conv2d_pw_unsigned_float";
    expected_prepack = expect_buffer_weights
        ? "pack_q8_linear_weight_unsigned_buffer"
        : "pack_q8_linear_weight_unsigned_texture2d";
  } else {
    VK_CHECK_COND(!expect_buffer_weights);
    expected_execute = adapter->supports_int8_dot_product()
        ? "q8ta_conv2d_pw_float"
        : "q8ta_conv2d_pw_fallback_float";
    expected_prepack = "pack_q8_linear_weight_texture2d";
  }

  int32_t execute_matches = 0;
  std::string execute_names;
  for (const auto& node : graph.execute_nodes()) {
    const ExecuteNode* const node_ptr = node.get();
    VK_CHECK_COND(node_ptr != nullptr);
    const std::string& node_name = node_ptr->name();
    execute_names += node_name + " ";
    if (node_name.find("q8ta_conv2d_pw") == 0) {
      VK_CHECK_COND(
          node_name == expected_execute,
          "Expected ",
          expected_execute,
          " but selected execute kernel ",
          node_name);
      ++execute_matches;
    }
  }
  VK_CHECK_COND(execute_matches > 0, "Execute kernels: ", execute_names);

  int32_t prepack_matches = 0;
  std::string prepack_names;
  for (const auto& node : graph.prepack_nodes()) {
    const PrepackNode* const node_ptr = node.get();
    VK_CHECK_COND(node_ptr != nullptr);
    const std::string& node_name = node_ptr->name();
    prepack_names += node_name + " ";
    if (node_name.find("pack_q8_linear_weight") == 0) {
      VK_CHECK_COND(
          node_name == expected_prepack,
          "Expected ",
          expected_prepack,
          " but selected prepack kernel ",
          node_name);
      ++prepack_matches;
    }
  }
  VK_CHECK_COND(prepack_matches > 0, "Prepack kernels: ", prepack_names);
}

void assert_pw_kernel_selection(
    ComputeGraph& graph,
    const bool expect_unsigned,
    const bool expect_buffer_weights) {
  const vkapi::Adapter* const adapter = graph.context()->adapter_ptr();
  std::string expected_execute;
  std::string expected_prepack;
  if (expect_unsigned) {
    expected_execute = "q8ta_conv2d_pw_unsigned_float";
    expected_prepack = expect_buffer_weights
        ? "pack_q8_conv2d_weights_unsigned_buffer"
        : "pack_q8_conv2d_weights_unsigned_texture2d";
  } else {
    VK_CHECK_COND(!expect_buffer_weights);
    expected_execute = adapter->supports_int8_dot_product()
        ? "q8ta_conv2d_pw_float"
        : "q8ta_conv2d_pw_fallback_float";
    expected_prepack = "pack_q8_conv2d_weights_texture2d";
  }

  int32_t execute_matches = 0;
  std::string execute_names;
  for (const auto& node : graph.execute_nodes()) {
    const ExecuteNode* const node_ptr = node.get();
    VK_CHECK_COND(node_ptr != nullptr);
    const std::string& node_name = node_ptr->name();
    execute_names += node_name + " ";
    if (node_name.find("q8ta_conv2d_pw") == 0) {
      VK_CHECK_COND(
          node_name == expected_execute,
          "Expected ",
          expected_execute,
          " but selected execute kernel ",
          node_name);
      ++execute_matches;
    }
  }
  VK_CHECK_COND(execute_matches > 0, "Execute kernels: ", execute_names);

  int32_t prepack_matches = 0;
  std::string prepack_names;
  for (const auto& node : graph.prepack_nodes()) {
    const PrepackNode* const node_ptr = node.get();
    VK_CHECK_COND(node_ptr != nullptr);
    const std::string& node_name = node_ptr->name();
    prepack_names += node_name + " ";
    if (node_name.find("pack_q8_conv2d_weights") == 0) {
      VK_CHECK_COND(
          node_name == expected_prepack,
          "Expected ",
          expected_prepack,
          " but selected prepack kernel ",
          node_name);
      ++prepack_matches;
    }
  }
  VK_CHECK_COND(prepack_matches > 0, "Prepack kernels: ", prepack_names);
}

} // namespace

void test_q8ta_conv2d_dw(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef output_scale = args.at(idx++);
  const ValueRef output_zp = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef kernel_size = args.at(idx++);
  const ValueRef stride = args.at(idx++);
  const ValueRef padding = args.at(idx++);
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef activation = args.at(idx++);
  const ValueRef layout_int = args.at(idx++);
  const ValueRef impl_selector_str = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  // Extract the layout parameter and cast to GPUMemoryLayout
  int32_t layout_value = graph.extract_scalar<int32_t>(layout_int);
  utils::GPUMemoryLayout layout =
      static_cast<utils::GPUMemoryLayout>(layout_value);

  // Extract the impl_selector string
  std::string impl_selector = graph.extract_string(impl_selector_str);

  // Create temporary packed int8 tensors for input and output
  TmpTensor packed_int8_input(
      &graph, graph.sizes_of(fp_input), vkapi::kInt8x4, utils::kBuffer, layout);

  TmpTensor packed_int8_output(
      &graph,
      graph.sizes_of(fp_output),
      vkapi::kInt8x4,
      utils::kBuffer,
      layout);

  // Quantize floating point input to packed int8
  add_q8ta_quantize_node(
      graph, fp_input, input_scale, input_zp, packed_int8_input);

  if (impl_selector == "legacy_4w4c") {
    // Legacy path does not support activation
    std::vector<ValueRef> conv_args = {
        packed_int8_input,
        input_scale,
        input_zp,
        weight_data,
        weight_sums_data,
        weight_scales_data,
        output_scale,
        output_zp,
        bias_data,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        packed_int8_output};
    VK_GET_OP_FN("et_vk.conv2d_q8ta_q8csw_q8to.default")(graph, conv_args);
  } else {
    std::vector<ValueRef> conv_args = {
        packed_int8_input,
        input_scale,
        input_zp,
        weight_data,
        weight_sums_data,
        weight_scales_data,
        output_scale,
        output_zp,
        bias_data,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        activation,
        packed_int8_output};
    VK_GET_OP_FN("et_vk.q8ta_conv2d_dw.default")(graph, conv_args);
  }

  // Dequantize packed int8 output to floating point
  add_q8ta_dequantize_node(
      graph, packed_int8_output, output_scale, output_zp, fp_output);
}

void test_q8ta_conv2d(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef output_scale = args.at(idx++);
  const ValueRef output_zp = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef kernel_size = args.at(idx++);
  const ValueRef stride = args.at(idx++);
  const ValueRef padding = args.at(idx++);
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef activation = args.at(idx++);
  const ValueRef layout_int = args.at(idx++);
  const ValueRef impl_selector_str = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  // Extract the layout parameter and cast to GPUMemoryLayout
  int32_t layout_value = graph.extract_scalar<int32_t>(layout_int);
  utils::GPUMemoryLayout layout =
      static_cast<utils::GPUMemoryLayout>(layout_value);

  // Extract the impl_selector string
  std::string impl_selector = graph.extract_string(impl_selector_str);

  // Create temporary packed int8 tensors for input and output
  TmpTensor packed_int8_input(
      &graph, graph.sizes_of(fp_input), vkapi::kInt8x4, utils::kBuffer, layout);

  TmpTensor packed_int8_output(
      &graph,
      graph.sizes_of(fp_output),
      vkapi::kInt8x4,
      utils::kBuffer,
      layout);

  // Quantize floating point input to packed int8
  add_q8ta_quantize_node(
      graph, fp_input, input_scale, input_zp, packed_int8_input);

  if (impl_selector == "legacy_4w4c") {
    // Legacy path does not support activation
    std::vector<ValueRef> conv_args = {
        packed_int8_input,
        input_scale,
        input_zp,
        weight_data,
        weight_sums_data,
        weight_scales_data,
        output_scale,
        output_zp,
        bias_data,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        packed_int8_output};
    VK_GET_OP_FN("et_vk.conv2d_q8ta_q8csw_q8to.default")(graph, conv_args);
  } else {
    std::vector<ValueRef> conv_args = {
        packed_int8_input,
        input_scale,
        input_zp,
        weight_data,
        weight_sums_data,
        weight_scales_data,
        output_scale,
        output_zp,
        bias_data,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        activation,
        packed_int8_output};
    if (impl_selector == "im2col" || impl_selector == "im2col_unsigned" ||
        impl_selector == "im2col_auto") {
      const vkapi::Adapter* const adapter = graph.context()->adapter_ptr();
      bool expect_unsigned = impl_selector == "im2col_unsigned";
      if (impl_selector == "im2col_auto") {
        VK_GET_OP_FN("et_vk.q8ta_conv2d_im2col.default")(graph, conv_args);
        expect_unsigned = can_use_unsigned_pw_dot(
            *adapter, graph.size_at<int64_t>(-1, weight_data));
      } else {
        q8ta_conv2d_im2col_impl(graph, expect_unsigned, conv_args);
      }
      const int64_t packed_height =
          utils::div_up_4(graph.size_at<int64_t>(-1, weight_data));
      const int64_t packed_width =
          utils::div_up_4(graph.size_at<int64_t>(-2, weight_data)) * 4;
      const int64_t max_texture_extent = adapter->max_texture2d_dim();
      const bool expect_buffer_weights =
          packed_width > max_texture_extent * 4 ||
          packed_height > max_texture_extent;
      assert_im2col_kernel_selection(
          graph, expect_unsigned, expect_buffer_weights);
    } else if (impl_selector == "general") {
      VK_GET_OP_FN("et_vk.q8ta_conv2d_general.default")(graph, conv_args);
    } else {
      VK_GET_OP_FN("et_vk.q8ta_conv2d.default")(graph, conv_args);
    }
  }

  // Dequantize packed int8 output to floating point
  add_q8ta_dequantize_node(
      graph, packed_int8_output, output_scale, output_zp, fp_output);
}

void test_q8ta_conv2d_pw(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef output_scale = args.at(idx++);
  const ValueRef output_zp = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef kernel_size = args.at(idx++);
  const ValueRef stride = args.at(idx++);
  const ValueRef padding = args.at(idx++);
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef activation = args.at(idx++);
  const ValueRef layout_int = args.at(idx++);
  const ValueRef impl_selector_str = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  // Extract the layout parameter and cast to GPUMemoryLayout
  int32_t layout_value = graph.extract_scalar<int32_t>(layout_int);
  utils::GPUMemoryLayout layout =
      static_cast<utils::GPUMemoryLayout>(layout_value);

  // Extract the impl_selector string
  std::string impl_selector = graph.extract_string(impl_selector_str);

  // Create temporary packed int8 tensors for input and output
  TmpTensor packed_int8_input(
      &graph,
      graph.sizes_of(fp_input),
      vkapi::kInt8x4,
      utils::kBuffer,
      utils::kPackedInt8_4W4C);

  TmpTensor packed_int8_output(
      &graph,
      graph.sizes_of(fp_output),
      vkapi::kInt8x4,
      utils::kBuffer,
      layout);

  // Quantize floating point input to packed int8
  add_q8ta_quantize_node(
      graph, fp_input, input_scale, input_zp, packed_int8_input);

  if (impl_selector == "legacy_4w4c") {
    // Legacy path does not support activation
    std::vector<ValueRef> conv_args = {
        packed_int8_input,
        input_scale,
        input_zp,
        weight_data,
        weight_sums_data,
        weight_scales_data,
        output_scale,
        output_zp,
        bias_data,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        packed_int8_output};
    VK_GET_OP_FN("et_vk.conv2d_q8ta_q8csw_q8to.default")(graph, conv_args);
  } else {
    std::vector<ValueRef> conv_args = {
        packed_int8_input,
        input_scale,
        input_zp,
        weight_data,
        weight_sums_data,
        weight_scales_data,
        output_scale,
        output_zp,
        bias_data,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        activation,
        packed_int8_output};
    if (impl_selector == "pw_signed" || impl_selector == "pw_unsigned" ||
        impl_selector == "pw_auto") {
      const vkapi::Adapter* const adapter = graph.context()->adapter_ptr();
      bool expect_unsigned = impl_selector == "pw_unsigned";
      if (impl_selector == "pw_auto") {
        VK_GET_OP_FN("et_vk.q8ta_conv2d_pw.default")(graph, conv_args);
        expect_unsigned = can_use_unsigned_pw_dot(
            *adapter, graph.size_at<int64_t>(-1, weight_data));
      } else {
        q8ta_conv2d_pw_impl(graph, expect_unsigned, conv_args);
      }
      const int64_t packed_height =
          utils::div_up_4(graph.size_at<int64_t>(-1, weight_data));
      const int64_t packed_width =
          utils::div_up_4(graph.size_at<int64_t>(-2, weight_data)) * 4;
      const int64_t max_texture_extent = adapter->max_texture2d_dim();
      const bool expect_buffer_weights =
          packed_width > max_texture_extent * 4 ||
          packed_height > max_texture_extent;
      assert_pw_kernel_selection(graph, expect_unsigned, expect_buffer_weights);
    } else {
      VK_GET_OP_FN("et_vk.q8ta_conv2d_pw.default")(graph, conv_args);
    }
  }

  // Dequantize packed int8 output to floating point
  add_q8ta_dequantize_node(
      graph, packed_int8_output, output_scale, output_zp, fp_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_q8ta_conv2d_dw.default, test_q8ta_conv2d_dw);
  VK_REGISTER_OP(test_etvk.test_q8ta_conv2d.default, test_q8ta_conv2d);
  VK_REGISTER_OP(test_etvk.test_q8ta_conv2d_pw.default, test_q8ta_conv2d_pw);
}

} // namespace vkcompute
