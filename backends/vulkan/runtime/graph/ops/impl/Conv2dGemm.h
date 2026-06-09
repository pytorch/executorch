/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <optional>

namespace vkcompute {

/*
 * End-to-end orchestration for an FP32 / FP16 conv2d computed as im2col ->
 * GEMM.
 *
 * Dataflow (input t_in [1, C_in, H_in, W_in] -> output t_out [1, C_out, H_out,
 * W_out]):
 *
 *   1. im2col: add_conv2d_im2col_node (conv2d_im2col.glsl) expands t_in into an
 *      im2col matrix. The K (reduction) axis is laid out as
 *      K = (ki * Kw + kj) * Cin_padded + ci, with K_total = Kh * Kw *
 *      align_up_4(C_in). The im2col storage type selects its shape (see
 *      conv2d_gemm_impl):
 *        - buffer / texture2d: flat [M, K_total], width-packed, with
 *          M = H_out * W_out.
 *        - texture3d: [1, K_total, H_out, W_out], channels-packed, so the K4
 *          tiles lay along Z.
 *   2. GEMM: add_conv2d_gemm_node (conv2d_gemm.glsl) multiplies the im2col
 *      matrix by the packed weight [C_out, K_total] to produce t_out. The
 *      packed weight is prepacked on the GPU directly from the serialized
 *      [C_out, C_in, Kh, Kw] weight (no CPU repack), applying the im2col K-axis
 *      decode plus a 4OC x 4IC blocked transpose.
 *
 * This function performs both dispatch and prepack registration. The im2col
 * intermediate is allocated as a scoped TmpTensor scratch tensor, so the memory
 * planner can alias one backing buffer across the non-overlapping im2col
 * lifetimes of every conv2d layer (peak memory tracks the largest single im2col
 * rather than the sum). The packed weight is produced by a GPU prepack node
 * (PrepackNode running pack_conv2d_gemm_weight.glsl) from the serialized
 * weight.
 *
 * Constraints (asserted internally):
 *   - input batch == 1
 *   - weight rank == 4
 *   - groups == 1 (general grouped conv not yet supported)
 *   - transposed == false
 *
 * `im2col_storage_override` controls the storage type of the im2col
 * intermediate tensor (and, by extension, the conv2d_gemm input-load variant):
 *   - std::nullopt (default): the production path. Storage is auto-selected
 *     from device characteristics and texture-extent limits — byte-for-byte
 *     the same selection used by the registered op.
 *   - a concrete StorageType: force that storage, skipping auto-selection.
 *     Used by tests to exercise each storage variant deterministically and
 *     independently of the device. Must be one of kBuffer / kTexture2D /
 *     kTexture3D.
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
    const float out_max_val = 0.0f,
    const std::optional<utils::StorageType> im2col_storage_override =
        std::nullopt);

} // namespace vkcompute
