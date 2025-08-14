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

vkapi::ShaderInfo get_nchw_to_tensor_shader(
    ComputeGraph& graph,
    const ValueRef dst,
    bool int8_buffer_enabled = true,
    bool push_constant_variant = true);
vkapi::ShaderInfo get_tensor_to_nchw_shader(
    ComputeGraph& graph,
    const ValueRef src,
    bool int8_buffer_enabled = true,
    bool push_constant_variant = true);

} // namespace vkcompute
