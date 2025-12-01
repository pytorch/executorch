/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::optional;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

Tensor& quantized_matmul_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    int64_t X_zero_point,
    const Tensor& Y,
    int64_t Y_zero_point,
    const optional<Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    Tensor& out);

Tensor& quantized_matmul_asym8sxasym8s_asym8s_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    int64_t X_zero_point,
    const Tensor& Y,
    int64_t Y_zero_point,
    const optional<Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    Tensor& out);

Tensor& quantized_matmul_asym8uxasym8u_asym8u_out(
    KernelRuntimeContext& ctx,
    const Tensor& X,
    int64_t X_zero_point,
    const Tensor& Y,
    int64_t Y_zero_point,
    const optional<Tensor>& bias,
    int64_t out_multiplier,
    int64_t out_shift,
    int64_t out_zero_point,
    bool transposed,
    Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
