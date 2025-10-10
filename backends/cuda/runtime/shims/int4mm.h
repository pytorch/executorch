/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <executorch/backends/aoti/common_shims.h>

namespace executorch::backends::cuda {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

#ifdef __cplusplus
extern "C" {
#endif

AOTITorchError aoti_torch_cuda__weight_int4pack_mm(
    Tensor* self,
    Tensor* mat2,
    int64_t qGroupSize,
    Tensor* qScaleAndZeros,
    Tensor** ret0);

#ifdef __cplusplus
}
#endif

} // namespace executorch::backends::cuda
