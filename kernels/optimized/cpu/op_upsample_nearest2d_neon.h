/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#if defined(__aarch64__)

namespace torch {
namespace executor {
namespace native {
namespace opt_upsample_nearest2d_internal {

void upsample_nearest2d_2x_nchw_u16(
    const uint16_t* input,
    uint16_t* output,
    int64_t begin,
    int64_t end,
    int64_t input_height,
    int64_t input_width);

} // namespace opt_upsample_nearest2d_internal
} // namespace native
} // namespace executor
} // namespace torch

#endif
