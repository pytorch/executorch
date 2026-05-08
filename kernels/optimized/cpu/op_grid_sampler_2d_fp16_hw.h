/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __aarch64__

namespace torch {
namespace executor {
namespace native {
namespace opt_grid_sampler_2d_internal {

// Hardware-fp16 NEON bilinear + zeros-padding fast path. Defined in
// op_grid_sampler_2d_fp16_hw.cpp, which is the only translation unit
// compiled with `-march=armv8.2-a+fp16`. Only safe to call when
// cpuinfo_has_arm_neon_fp16() reports true — see the runtime dispatcher
// in op_grid_sampler_2d.cpp.
//
// Input/output buffers are passed as void* (raw uint16_t storage
// interpreted as __fp16) so this header doesn't need <arm_neon.h> and
// callers don't need the +fp16 march flag just to declare it.
void grid_sampler_2d_bilinear_fp16_hw(
    const void* input,
    const void* grid,
    void* output,
    int N,
    int C,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    bool align_corners);

} // namespace opt_grid_sampler_2d_internal
} // namespace native
} // namespace executor
} // namespace torch

#endif // __aarch64__
