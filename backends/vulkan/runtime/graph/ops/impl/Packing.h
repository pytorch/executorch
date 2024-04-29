/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

ValueRef channel_image_repacking(
    ComputeGraph& graph,
    ValueRef in,
    api::GPUMemoryLayout target_layout,
    const api::ShaderInfo& shader);

ValueRef convert_image_channels_packed_to_width_packed(
    ComputeGraph& graph,
    ValueRef in);

ValueRef convert_image_channels_packed_to_height_packed(
    ComputeGraph& graph,
    ValueRef in);

} // namespace vkcompute
