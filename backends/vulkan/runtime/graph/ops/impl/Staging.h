/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <cstring>

namespace vkcompute {

//
// Staging Buffer <-> Tensor
//

void add_staging_to_tensor_node(
    ComputeGraph& graph,
    const ValueRef in_staging,
    const ValueRef out_tensor);

void add_tensor_to_staging_node(
    ComputeGraph& graph,
    const ValueRef in_tensor,
    const ValueRef out_staging);

//
// Standard Prepack
//

/*
 * Given that `v` is a `TensorRef`, create a new `Tensor` value with the
 * specified `storage_type` and `memory_layout`, and add a a prepacking node to
 * transfer the `TensorRef` data to the new `Tensor` object via a staging to
 * tensor shader. The created `Tensor` value is then returned.
 *
 * If `passthrough` is `true`, then `v` may be a `Tensor` as well. If `v` is a
 * `Tensor`, then it is returned as-is. If `passthrough` is `false` (default),
 * then an exception will be thrown.
 */

ValueRef prepack_standard(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout layout,
    const bool passthrough = false,
    const utils::AxisMapLayout axis_map_layout = utils::kDefaultAxisMap);

/*
 * Same as prepack_standard, but transpose the height and width dimensions of
 * the tensor while packing.
 */
ValueRef prepack_standard_hw_transposed(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout layout,
    const bool passthrough = false,
    const utils::AxisMapLayout axis_map_layout = utils::kDefaultAxisMap);

/*
 * Equivalent to `prepack_standard()` function, except the `storage_type` and
 * `memory_layout` are set to match `to_copy`, which must be a `Tensor`.
 */
ValueRef prepack_standard_like(
    ComputeGraph& graph,
    const ValueRef tensor_data,
    const ValueRef to_copy,
    const bool passthrough = false);

//
// Direct buffer copy prepack
//

/*
 * Given that `v` is a `TensorRef`, create a new `Tensor` value with buffer
 * storage and `kWidthPacked` memory layout, and add a prepacking node to
 * transfer the `TensorRef` data to the new `Tensor` object via a direct buffer
 * to buffer copy shader.
 */
ValueRef prepack_direct_copy_buffer(
    ComputeGraph& graph,
    const ValueRef tensor_data);

//
// Op specific prepack functions
//

ValueRef prepack_int4_linear_weight_transposed_interleaved(
    ComputeGraph& graph,
    const ValueRef qmat2_data);

} // namespace vkcompute
