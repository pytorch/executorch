/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>

#include <optional>
#include <vector>

namespace executorch::backends::webgpu {

struct ConvClamp {
  float min;
  float max;
};

// aten.convolution.default handler body (args laid out as convolution). A
// `clamp` is fused into the output write (groups == 1 only); this is how
// et_vk.conv_with_clamp shares the convolution kernels. A caller with a kernel
// of its own passes `can_decline`: then nothing is recorded and false is
// returned when the GEMM is a poor fit (its tile grid exceeds the dispatch
// limit, or its tiles would be mostly padding).
bool add_conv2d(
    WebGPUGraph& graph,
    const std::vector<int>& args,
    std::optional<ConvClamp> clamp,
    bool can_decline = false);

} // namespace executorch::backends::webgpu
