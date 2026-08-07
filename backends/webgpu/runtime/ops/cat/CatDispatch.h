/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/webgpu/runtime/WebGPUDispatchMath.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>

#include <cstddef>

namespace executorch::backends::webgpu {

inline void set_cat_dispatch_grid(
    WebGPUGraph& graph,
    size_t dispatch_index,
    const utils::WgCount& grid) {
  WebGPUDispatch& dispatch = graph.dispatch_at(dispatch_index);
  dispatch.workgroup_count_x = grid.x;
  dispatch.workgroup_count_y = grid.y;
}

} // namespace executorch::backends::webgpu
