/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/q8ta/Q8taConv2dCommon.h>

namespace vkcompute {

//
// Fused PixelShuffle operating on int8x4 packed tensors.
//
// Replaces the decomposed chain:
//   q8ta_dequantize -> view -> permute -> view -> [relu] -> q8ta_quantize
// with a single byte-shuffle (and optional requantize when scales differ or an
// activation is applied).
//

void add_q8ta_pixel_shuffle_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef output_inv_scale,
    const ValueRef output_zp,
    const ValueRef upscale_factor,
    const ActivationType activation,
    const ValueRef packed_int8_output);

} // namespace vkcompute
