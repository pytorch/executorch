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

// Cooperative-matrix family path of et_vk.linear_dq8ca_q4gsw
// (CoopmatFamily.h). If the device belongs to a family and one of its rows
// applies to the op's build-time shape, adds the whole op (activation
// quantize/pack and the int8 cooperative-matrix linear) and returns true;
// otherwise adds nothing and returns false, and the caller builds its usual
// nodes. `args`: {input, input_scale, input_zp, weight, weight_sums,
// weight_scales, group_size, bias, output}.
bool try_add_dq8ca_q4gsw_linear_wmma_node(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args);

} // namespace vkcompute
