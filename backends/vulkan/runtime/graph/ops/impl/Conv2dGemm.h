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
 * End-to-end orchestration for an FP32 / FP16 conv2d computed as
 * im2col -> GEMM.  The dataflow is:
 *
 *   t_in [1, C_in, H_in, W_in]
 *     │
 *     ▼   (add_conv2d_im2col_node, conv2d_im2col.glsl)
 *   im2col [1, K_total, H_out, W_out]      K_total = Kh * Kw * align_up_4(C_in)
 *     │
 *     │ + flattened weight [C_out, K_total]   (built CPU-side from
 *     │   [C_out, C_in, Kh, Kw] with the ci ↔ (ki, kj) transpose)
 *     ▼   (add_conv2d_gemm_node, conv2d_gemm.glsl)
 *   t_out [1, C_out, H_out, W_out]
 *
 * This function performs both dispatch and prepack registration. The im2col
 * intermediate is allocated as a graph tensor; the flattened weight is
 * registered as a new TensorRef owned by the graph.
 *
 * Constraints (asserted internally):
 *   - input batch == 1
 *   - weight rank == 4
 *   - groups == 1 (general grouped conv not yet supported)
 *   - transposed == false
 */
void conv2d_gemm_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out,
    const bool clamp_out = false,
    const float out_min_val = 0.0f,
    const float out_max_val = 0.0f);

} // namespace vkcompute
