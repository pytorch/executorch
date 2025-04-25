/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#define ET_FORALL_CADENCE_QUANTIZED_TYPES(_) \
  _(uint8_t, Byte)                           \
  _(int8_t, Char)

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

// Quantize the input tensor (PT2 version). Note that quant_<min,max> are not
// used in any computation.
void quantize_per_tensor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ::executorch::aten::ScalarType dtype,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
