/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/cuda/runtime/shims/int4_plain_mm.h>
#include <executorch/backends/cuda/runtime/shims/int4_plain_mm.cuh>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::cuda {
#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cuda_int4_plain_mm(
    Tensor* self,
    Tensor* qdata,
    Tensor* scale,
    Tensor* scale_step,
    Tensor* zero,
    Tensor* zero_step,
    int64_t group_size,
    Tensor** ret0) {
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: self is null");

  ET_CHECK_OR_RETURN_ERROR(
      qdata != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: qdata is null");

  ET_CHECK_OR_RETURN_ERROR(
      scale != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: scale is null");

  ET_CHECK_OR_RETURN_ERROR(
      scale_step != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: scale_step is null");

  ET_CHECK_OR_RETURN_ERROR(
      zero != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: zero is null");

  ET_CHECK_OR_RETURN_ERROR(
      zero_step != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: zero_step is null");

  ET_CHECK_OR_RETURN_ERROR(
      ret0 != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: ret0 is null");

  // Validate the coalesced metadata layout.

  const int64_t N = qdata->size(0);
  const int64_t K = qdata->size(1) * 2;

  ET_CHECK_OR_RETURN_ERROR(
      group_size > 0 && (group_size & (group_size - 1)) == 0,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: group_size=%lld must be a positive power of 2",
      static_cast<long long>(group_size));

  const int64_t n_groups = K / group_size;

  ET_CHECK_OR_RETURN_ERROR(
      scale->dim() == 2 && zero->dim() == 2,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: scale/zero must be 2D (got scale.dim()=%lld, zero.dim()=%lld)",
      static_cast<long long>(scale->dim()),
      static_cast<long long>(zero->dim()));

  ET_CHECK_OR_RETURN_ERROR(
      scale->size(0) == N && zero->size(0) == N,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: scale/zero must be coalesced [N, K/group_size] (AOT layout); native [n_groups, N] is not supported - repack via pack_linear_for_cuda. Expected size(0)=N=%lld, got scale.size(0)=%lld, zero.size(0)=%lld",
      static_cast<long long>(N),
      static_cast<long long>(scale->size(0)),
      static_cast<long long>(zero->size(0)));

  ET_CHECK_OR_RETURN_ERROR(
      scale->size(1) == n_groups && zero->size(1) == n_groups,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: scale/zero must be coalesced [N, K/group_size] (AOT layout); native [n_groups, N] is not supported - repack via pack_linear_for_cuda. Expected size(1)=K/group_size=%lld, got scale.size(1)=%lld, zero.size(1)=%lld",
      static_cast<long long>(n_groups),
      static_cast<long long>(scale->size(1)),
      static_cast<long long>(zero->size(1)));

  // Scale step: per-256-super-block fp16, [N, K/256].
  const int64_t n_super = K / 256;
  ET_CHECK_OR_RETURN_ERROR(
      scale_step->dim() == 2 && scale_step->size(0) == N &&
          scale_step->size(1) == n_super,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: scale_step must be [N, K/256] fp16. Expected size(0)=N=%lld size(1)=%lld, got dim()=%lld size(0)=%lld size(1)=%lld",
      static_cast<long long>(N),
      static_cast<long long>(n_super),
      static_cast<long long>(scale_step->dim()),
      static_cast<long long>(scale_step->dim() == 2 ? scale_step->size(0) : -1),
      static_cast<long long>(scale_step->dim() == 2 ? scale_step->size(1) : -1));

  // Zero step: per-row bf16, [N, 1].
  ET_CHECK_OR_RETURN_ERROR(
      zero_step->dim() == 2 && zero_step->size(0) == N &&
          zero_step->size(1) == 1,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: zero_step must be [N, 1] bf16. Expected size(0)=N=%lld size(1)=1, got dim()=%lld size(0)=%lld size(1)=%lld",
      static_cast<long long>(N),
      static_cast<long long>(zero_step->dim()),
      static_cast<long long>(zero_step->dim() == 2 ? zero_step->size(0) : -1),
      static_cast<long long>(zero_step->dim() == 2 ? zero_step->size(1) : -1));

  int32_t M = self->size(0);
  Tensor* C = nullptr;
  std::array<int64_t, 2> c_shape = {M, N};
  std::array<int64_t, 2> c_stride = {N, 1};
  aoti_torch_empty_strided(
      2,
      c_shape.data(),
      c_stride.data(),
      static_cast<int32_t>(
          executorch::backends::aoti::slim::c10::ScalarType::BFloat16),
      static_cast<int32_t>(
          executorch::backends::aoti::slim::c10::DeviceType::CUDA),
      0,
      &C);

  _int4_plain_mm_cuda(
      *self, *qdata, *scale, *scale_step, *zero, *zero_step, group_size, C);
  ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();

  *ret0 = C;
  return Error::Ok;
}

#ifdef __cplusplus
}
#endif
} // namespace executorch::backends::cuda
