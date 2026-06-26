/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>

namespace executorch::backends::webgpu {

// Flat copy output[i]=input[i]; mirrors Vulkan add_view_copy_node (View.h).
void add_flat_copy(WebGPUGraph& graph, int in_id, int out_id);

} // namespace executorch::backends::webgpu
