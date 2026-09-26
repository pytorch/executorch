/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace impl {
namespace generic {
namespace native {

/**
 * @brief Compute quantized softmax with optional masking support.
 *
 * Computes softmax on the input tensor along the specified dimension,
 * with support for different masking strategies controlled by mask_type.
 *
 * @param ctx Kernel runtime context (unused)
 * @param input Input quantized tensor
 * @param mask Mask tensor (currently unused, reserved for future mask types)
 * @param dim Dimension along which to compute softmax. Only the last dimension
 *            is currently supported (dim == -1 or dim == input.dim() - 1)
 * @param mask_type Masking strategy to use:
 *                  - 0: No masking. Standard softmax is computed over all
 *                       elements in the dimension.
 *                  - 1: Position-based causal masking. Uses the pos tensor to
 *                       determine which positions to attend to. For each row i,
 *                       positions 0 to (pos[0] + i) are attended, and positions
 *                       beyond that are masked out (set to 0 probability).
 *                       This implements incremental causal attention where
 *                       each subsequent row can attend to one additional
 *                       position.
 * @param pos Position tensor for causal masking (used when mask_type == 1).
 *            Contains the base position value. Supports int16 (Short) or
 *            int64 (Long) scalar types.
 * @param in_scale Input quantization scale tensor
 * @param in_zero_point Input quantization zero point tensor
 * @param out_scale Output quantization scale tensor
 * @param out_zero_point Output quantization zero point tensor
 * @param out Output tensor to store the result
 * @return Reference to the output tensor
 */
::executorch::aten::Tensor& quantized_softmax_out(
    __ET_UNUSED ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& mask,
    int64_t dim,
    int64_t mask_type,
    const ::executorch::aten::Tensor& pos,
    const ::executorch::aten::Tensor& in_scale,
    const ::executorch::aten::Tensor& in_zero_point,
    const ::executorch::aten::Tensor& out_scale,
    const ::executorch::aten::Tensor& out_zero_point,
    ::executorch::aten::Tensor& out);

::executorch::aten::Tensor& quantized_softmax_per_tensor_out(
    __ET_UNUSED ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& mask,
    int64_t dim,
    int64_t mask_type,
    const ::executorch::aten::Tensor& pos,
    double in_scale,
    int64_t in_zero_point,
    double out_scale,
    int64_t out_zero_point,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
