/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/api/api.h>

namespace vkcompute {

// add_copy_offset_node resumes the vkCmdCopyImage command. It copies the
// texture extents specified by the range, src_offset, and dst_offset (all are
// in texture coordinate (x, y, z) from the input image to the output image.
//
// It is possible to have input and output to point to the same image
// object. But when the source range and destination range overlap, the behavior
// is undefined.
void add_copy_offset_node(
    ComputeGraph& graph,
    const ValueRef in,
    const utils::ivec3& range,
    const utils::ivec3& src_offset,
    const utils::ivec3& dst_offset,
    const ValueRef out);

// add_copy_channel_offset_node behaves similar to add_copy_node, except that it
// works on the channel dimensions of the tensor (up to 4 dimensions in NCHW).
// The range and offset arguments are in the tensor coordinate. It assumes the
// underlying texture is channel-packed.
//
// This function is specialized implementation for copying
// channel packed values. The complication comes from when reading / writing the
// channel dimension on indices that are not aligned to packing, we will need
// be careful about the boundaries.
//
// It achieves the following:
//   out[:, dst_channel_offset:dst_channel_offset + channel_range, :, :] =
//       in [:, src_channel_offset:src_channel_offset + channel_range, :, :]
void add_copy_channel_offset_node(
    ComputeGraph& graph,
    const ValueRef in,
    int32_t channel_range,
    int32_t src_channel_offset,
    int32_t dst_channel_offset,
    const ValueRef out);

} // namespace vkcompute
