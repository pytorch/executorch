/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// AOTI op-registry-backed fallbacks for the v2 Metal backend.
//
// AOTI declares a small set of aten ops (mm, bmm, ...) as
// supported_fallback_kernels, so the AOTI .so emits direct calls to these
// symbols rather than generated shaders. v2 routes every fallback through
// the metal_v2 MetalOpRegistry — if an op isn't registered there, we
// return Error::NotImplemented (clean error rather than a missing-symbol
// dlopen crash).

#pragma once

#include <executorch/backends/apple/metal/runtime/shims/v2/aoti_types.h>

namespace executorch {
namespace backends {
namespace metal {

#ifdef __cplusplus
extern "C" {
#endif

// 2D matmul: out = self @ mat2.
AOTITorchError aoti_torch_mps_mm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2);

// 3D batched matmul: out = self @ mat2 (per batch).
AOTITorchError aoti_torch_mps_bmm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2);

#ifdef __cplusplus
} // extern "C"
#endif

} // namespace metal
} // namespace backends
} // namespace executorch
