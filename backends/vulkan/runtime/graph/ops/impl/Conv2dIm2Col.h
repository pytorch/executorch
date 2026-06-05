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
 * Inputs:
 *   in          : input texture3D channels-packed [1, C_in, H_in, W_in]
 *   im2col_out  : output 2D tensor [M, K_total] (caller allocates),
 *                 storage = texture2d (kWidthPacked) or buffer
 *   kernel_h/w  : conv kernel dimensions
 *   stride_*    : conv strides
 *   padding_*   : conv paddings
 *   dilation_*  : conv dilations
 *   Cin_padded  : align_up_4(C_in)
 *   H_out, W_out: output spatial extents
 */
void add_conv2d_im2col_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef im2col_out,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t padding_h,
    const int32_t padding_w,
    const int32_t dilation_h,
    const int32_t dilation_w,
    const int32_t Cin_padded,
    const int32_t H_out,
    const int32_t W_out);

} // namespace vkcompute
