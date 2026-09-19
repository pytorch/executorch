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

// Copy: flat byte for same dtype, numeric convert for int<->float (Vulkan).
void add_to_copy_node(WebGPUGraph& graph, int in_id, int out_id);

} // namespace executorch::backends::webgpu
