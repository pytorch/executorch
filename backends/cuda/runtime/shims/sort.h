/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/export.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Sorts a tensor along a given dimension.
 *
 * @param self Input tensor to sort (any numeric dtype, CUDA device)
 * @param stable Pointer to bool — if non-null and *stable != 0, uses stable
 * sort
 * @param dim Dimension along which to sort
 * @param descending If non-zero, sort in descending order
 * @param ret0 Output: sorted values tensor (same shape/dtype as self)
 * @param ret1 Output: indices tensor (int64, same shape as self)
 * @return AOTITorchError
 */
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_cuda_sort_stable(
    Tensor* self,
    int32_t* stable,
    int64_t dim,
    int32_t descending,
    Tensor** ret0,
    Tensor** ret1);

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
