// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {

Tensor allocate_tensor(
    KernelRuntimeContext& ctx,
    const ArrayRef<Tensor::SizesType>& sizes,
    const ArrayRef<Tensor::DimOrderType>& dim_order,
    const ArrayRef<Tensor::StridesType>& strides,
    const ScalarType& dtype);

} // namespace executor
} // namespace torch
