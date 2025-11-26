// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace impl {
namespace generic {
namespace native {

::executorch::aten::Tensor& roi_align_box_processor_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& rois,
    int64_t output_size_h,
    int64_t output_size_w,
    int64_t sampling_ratio,
    bool aligned,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
