/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

/*
 * Dispatch a single im2col transformation node for an FP32 / FP16 conv2d.
 *
 * Produces a 2D tensor of logical shape
 *   [M, K_total]
 * where
 *   M       = H_out * W_out
 *   K_total = kernel_h * kernel_w * align_up_4(C_in)
 *
 * The K dimension is laid out so that consecutive 4-tiles of K hold 4
 * consecutive ci values for the same (ki, kj) kernel position. This is the
 * layout `conv2d_gemm` consumes for the GEMM step.
 *
 * The im2col output tensor's storage type (texture2d width-packed or
 * buffer) is determined by the caller; this function picks the matching
 * shader variant based on `graph.storage_type_of(im2col_out)`.
 *
 * Dynamic shapes: the spatial output extents (W_out / H_out / M) are derived in
 * the shader from the refreshed in_sizes UBO, and the im2col_out tensor is
 * virtually resized on every trigger_resize() from the current input shape, so
 * this node tracks dynamic input shapes. Cin_padded / K4_total are
 * shape-independent and remain baked into the push constant. `stride`,
 * `padding`, `dilation` are the original graph ValueRefs (used by the resize
 * function to recompute output extents); `weight_data` is the original 4D
 * weight (used only for its kernel dims during resize).
 *
 * Inputs:
 *   in          : input texture3D channels-packed [1, C_in, H_in, W_in]
 *   im2col_out  : output tensor (caller allocates), [M, K_total] for
 *                 buffer/texture2d (kWidthPacked) or [1, K_total, H_out, W_out]
 *                 for texture3d (kChannelsPacked)
 *   weight_data : original [C_out, C_in, kernel_h, kernel_w] weight
 *   stride/padding/dilation : original conv param IntList ValueRefs
 *   kernel_h/w  : conv kernel dimensions
 *   stride_*    : conv strides
 *   padding_*   : conv paddings
 *   dilation_*  : conv dilations
 *   Cin_padded  : align_up_4(C_in)
 */
void add_conv2d_im2col_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef im2col_out,
    const ValueRef weight_data,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t padding_h,
    const int32_t padding_w,
    const int32_t dilation_h,
    const int32_t dilation_w,
    const int32_t Cin_padded);

} // namespace vkcompute
