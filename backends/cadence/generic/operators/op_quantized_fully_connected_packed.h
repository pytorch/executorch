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

// Fully connected with a row-packed sub-byte weight. `weight` is an int8 blob
// of shape [out_dim, packed_row_bytes], so `in_dim` and `weight_bits` are
// explicit rather than read off the shape.
::executorch::aten::Tensor& quantized_fully_connected_packed_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& in,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    int64_t in_dim,
    int64_t weight_bits,
    int64_t in_zero_point,
    const std::optional<::executorch::aten::Tensor>& weight_zero_point,
    const ::executorch::aten::Tensor& out_multiplier,
    const ::executorch::aten::Tensor& out_shift,
    int64_t out_zero_point,
    const std::optional<::executorch::aten::Tensor>& offset,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
