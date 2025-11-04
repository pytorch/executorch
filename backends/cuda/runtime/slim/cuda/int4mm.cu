/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <executorch/backends/cuda/runtime/shims/aoti_torch/c/macros.h>
#include <executorch/backends/cuda/runtime/slim/cuda/int4mm.h>
#include <executorch/backends/cuda/runtime/slim/cuda/int4mm.cuh>

namespace executorch::backends::cuda::slim::cuda {
#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cuda__weight_int4pack_mm(
    AOTITensorHandle self,
    AOTITensorHandle mat2,
    int64_t qGroupSize,
    AOTITensorHandle qScaleAndZeros,
    AOTITensorHandle* ret0) {
  // Validate input parameters first
  // Only check for null pointers here, as the actual validation of tensor
  // properties is done in _weight_int4pack_mm_cuda
  //   ET_CHECK_OR_RETURN_ERROR(
  //       self != nullptr,
  //       InvalidArgument,
  //       "aoti_torch_cuda__weight_int4pack_mm failed: self tensor is null");

  //   ET_CHECK_OR_RETURN_ERROR(
  //       mat2 != nullptr,
  //       InvalidArgument,
  //       "aoti_torch_cuda__weight_int4pack_mm failed: mat2 tensor is null");

  //   ET_CHECK_OR_RETURN_ERROR(
  //       qScaleAndZeros != nullptr,
  //       InvalidArgument,
  //       "aoti_torch_cuda__weight_int4pack_mm failed: qScaleAndZeros tensor is
  //       null");

  //   ET_CHECK_OR_RETURN_ERROR(
  //       ret0 != nullptr,
  //       InvalidArgument,
  //       "aoti_torch_cuda__weight_int4pack_mm failed: ret0 is null");

  *ret0 = _weight_int4pack_mm_cuda(*self, *mat2, qGroupSize, *qScaleAndZeros);
  //   ET_CUDA_KERNEL_LAUNCH_CHECK_OR_RETURN_ERROR();
  //   return Error::Ok;
  return AOTI_TORCH_SUCCESS;
}

#ifdef __cplusplus
}
#endif
} // namespace executorch::backends::cuda::slim::cuda
