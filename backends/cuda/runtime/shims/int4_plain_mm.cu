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
    Tensor* zero,
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
      zero != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: zero is null");

  ET_CHECK_OR_RETURN_ERROR(
      ret0 != nullptr,
      InvalidArgument,
      "aoti_torch_cuda_int4_plain_mm: ret0 is null");

  int32_t M = self->size(0);
  int32_t N = qdata->size(0);
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

  _int4_plain_mm_cuda(*self, *qdata, *scale, *zero, group_size, C);
  ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();

  *ret0 = C;
  return Error::Ok;
}

#ifdef __cplusplus
}
#endif
} // namespace executorch::backends::cuda
