// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace impl {
namespace generic {
namespace native {

::executorch::aten::Tensor& rope_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& sin_tensor,
    const ::executorch::aten::Tensor& cos_tensor,
    const ::executorch::aten::optional<::executorch::aten::Tensor>& pos,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
