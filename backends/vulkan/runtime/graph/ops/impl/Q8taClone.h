/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

//
// Clone for int8x4 tensors (memory layout agnostic)
//

void add_q8ta_clone_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef packed_int8_output);

} // namespace vkcompute
