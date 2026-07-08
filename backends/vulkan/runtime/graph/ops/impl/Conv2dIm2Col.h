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
 * Dispatch a single im2col transformation node for one H-row tile of an
 * FP32 / FP16 conv2d.
 *
 * Materializes a 2D tensor of logical shape
 *   [oh_tile * W_out, K_total]
 * holding `oh_tile` output-height rows starting at output-height row
 * `oh_offset`, where
 *   K_total = kernel_h * kernel_w * align_up_4(C_in)
 *
 * Tiling by output-height rows bounds the scratch to a fixed byte budget
 * regardless of resolution (the full conv covers H_out rows across multiple
 * such dispatches); `oh_offset = 0`, `oh_tile = H_out` reproduces the untiled
 * single-dispatch case.
 *
 * The K dimension is laid out so that consecutive 4-tiles of K hold 4
 * consecutive ci values for the same (ki, kj) kernel position. This is the
 * layout `conv2d_gemm` consumes for the GEMM step.
 *
 * The im2col output tensor's storage type (texture2d width-packed, buffer, or
 * texture3d channels-packed) is determined by the caller; this function picks
 * the matching shader variant based on `graph.storage_type_of(im2col_out)`.
 *
 * Dynamic shapes: W_out / H_out are derived in the shader from the refreshed
 * in_sizes UBO, and the im2col_out tensor is virtually resized on every
 * trigger_resize() (its W_out-dependent extent tracks the current input shape;
 * oh_tile is fixed). Cin_padded / K4_total / oh_offset / oh_tile are
 * shape-independent and baked into the push constant. `stride`, `padding`,
 * `dilation` are the original graph ValueRefs (used by the resize function);
 * `weight_data` is the original 4D weight (used only for its kernel dims during
 * resize).
 *
 * Inputs:
 *   in          : input texture3D channels-packed [1, C_in, H_in, W_in]
 *   im2col_out  : scratch tensor (caller allocates), [oh_tile * W_out, K_total]
 *                 for buffer/texture2d (kWidthPacked) or
 *                 [1, K_total, oh_tile, W_out] for texture3d (kChannelsPacked)
 *   weight_data : original [C_out, C_in, kernel_h, kernel_w] weight
 *   stride/padding/dilation : original conv param IntList ValueRefs
 *   kernel_h/w  : conv kernel dimensions
 *   stride_*    : conv strides
 *   padding_*   : conv paddings
 *   dilation_*  : conv dilations
 *   Cin_padded  : align_up_4(C_in)
 *   oh_offset   : first output-height row this tile materializes
 *   oh_tile     : number of output-height rows in this tile (scratch capacity);
 *                 also packed as a raw int into the last resize_args slot for
 *                 the resize fn (not a materialized ValueRef handle)
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
    const int32_t Cin_padded,
    const int32_t oh_offset,
    const int32_t oh_tile);

} // namespace vkcompute
