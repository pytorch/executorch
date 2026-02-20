/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <tuple>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace generic {
namespace native {

std::tuple<
    ::executorch::aten::Tensor&,
    ::executorch::aten::Tensor&,
    ::executorch::aten::Tensor&>
linalg_svd_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& A,
    bool full_matrices,
    bool compute_uv,
    ::executorch::aten::optional<::executorch::aten::string_view> driver,
    ::executorch::aten::Tensor& U,
    ::executorch::aten::Tensor& S,
    ::executorch::aten::Tensor& Vh);

} // namespace native
} // namespace generic
} // namespace impl
