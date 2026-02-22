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

// pack fp/bool mask to bitwise uint8 mask for SDPA (Scaled Dot-Product
// Attention) op
::executorch::aten::Tensor& sdpa_bitwise_mask_gen_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& mask,
    double threshold,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
