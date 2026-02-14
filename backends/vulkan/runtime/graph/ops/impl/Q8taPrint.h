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
// Debug print for int8x4 tensors (prints the first texel)
//

void add_q8ta_print_node(ComputeGraph& graph, const ValueRef packed_int8_input);

} // namespace vkcompute
