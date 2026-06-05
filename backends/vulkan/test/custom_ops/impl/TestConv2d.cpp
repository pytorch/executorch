/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Conv2dGemm.h>

#include <optional>

namespace vkcompute {

void test_conv2d(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // args[0]  = input  [N, C_in, H, W]
  // args[1]  = weight [C_out, C_in, K_h, K_w] (constant)
  // args[2]  = bias   (constant, or none)
  // args[3]  = stride_h    (int)
  // args[4]  = stride_w    (int)
  // args[5]  = padding_h   (int)
  // args[6]  = padding_w   (int)
  // args[7]  = dilation_h  (int)
  // args[8]  = dilation_w  (int)
  // args[9]  = impl_selector (string)
  // args[10] = output [N, C_out, H_out, W_out]
  //
  // impl_selector grammar:
  //   ""             -> aten.convolution.default (direct sliding-window)
  //   "im2col"       -> et_vk.conv2d_gemm.default, auto im2col storage
  //   "im2col_buffer"-> im2col/GEMM, force buffer im2col intermediate
  //   "im2col_tex2d" -> im2col/GEMM, force texture2d im2col intermediate
  //   "im2col_tex3d" -> im2col/GEMM, force texture3d im2col intermediate
  const ValueRef input = args.at(0);
  const ValueRef weight = args.at(1);
  const ValueRef bias = args.at(2);
  const int64_t stride_h = graph.extract_scalar<int64_t>(args.at(3));
  const int64_t stride_w = graph.extract_scalar<int64_t>(args.at(4));
  const int64_t padding_h = graph.extract_scalar<int64_t>(args.at(5));
  const int64_t padding_w = graph.extract_scalar<int64_t>(args.at(6));
  const int64_t dilation_h = graph.extract_scalar<int64_t>(args.at(7));
  const int64_t dilation_w = graph.extract_scalar<int64_t>(args.at(8));
  const std::string impl_selector = graph.extract_string(args.at(9));
  const ValueRef out = args.at(10);

  ValueRef stride =
      graph.add_scalar_list<int64_t>(std::vector<int64_t>{stride_h, stride_w});
  ValueRef padding = graph.add_scalar_list<int64_t>(
      std::vector<int64_t>{padding_h, padding_w});
  ValueRef dilation = graph.add_scalar_list<int64_t>(
      std::vector<int64_t>{dilation_h, dilation_w});

  // The forced-storage variants must reach conv2d_gemm_impl with the override,
  // which the registered op (et_vk.conv2d_gemm.default) cannot express since it
  // always auto-selects. Route those directly to conv2d_gemm_impl; the auto
  // ("im2col") and direct ("") paths stay on the registered-op dispatch.
  std::optional<utils::StorageType> im2col_storage_override;
  if (impl_selector == "im2col_buffer") {
    im2col_storage_override = utils::kBuffer;
  } else if (impl_selector == "im2col_tex2d") {
    im2col_storage_override = utils::kTexture2D;
  } else if (impl_selector == "im2col_tex3d") {
    im2col_storage_override = utils::kTexture3D;
  }

  if (im2col_storage_override.has_value()) {
    conv2d_gemm_impl(
        graph,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        out,
        /*clamp_out=*/false,
        /*out_min_val=*/0.0f,
        /*out_max_val=*/0.0f,
        im2col_storage_override);
    return;
  }

  ValueRef transposed = graph.add_scalar<bool>(false);
  ValueRef output_padding =
      graph.add_scalar_list<int64_t>(std::vector<int64_t>{0, 0});
  ValueRef groups = graph.add_scalar<int64_t>(1);

  const std::string target_op = (impl_selector == "im2col")
      ? "et_vk.conv2d_gemm.default"
      : "aten.convolution.default";

  VK_GET_OP_FN(target_op.c_str())
  (graph,
   {input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    out});
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(test_etvk.test_conv2d.default, test_conv2d);
}

} // namespace vkcompute
