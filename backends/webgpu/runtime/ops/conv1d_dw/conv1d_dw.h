/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>

#include <vector>

namespace executorch::backends::webgpu {

// conv1d (3D) entry: pointwise (K=1, groups=1) or depthwise (groups==C). Called
// by the single aten.convolution.default handler in Conv2d.cpp, which
// dispatches on input rank (3D -> conv1d, 4D -> conv2d) so one registration
// serves both.
void conv1d_dispatch(WebGPUGraph& graph, const std::vector<int>& args);

} // namespace executorch::backends::webgpu
