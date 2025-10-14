/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

namespace vkcompute {

utils::uvec3 default_pick_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  return graph->create_global_wg_size(out);
}

utils::uvec3 default_pick_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)args;
  (void)resize_args;
  return graph->create_local_wg_size(global_workgroup_size);
}

utils::uvec3 pick_hw_square_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)args;
  (void)resize_args;
  // Some inactive invocations are okay; set 6 as the threshold to use the
  // a square wg size.
  if (global_workgroup_size[0u] >= 6 && global_workgroup_size[1u] >= 6) {
    return {8u, 8u, 1u};
  }
  // If width dim is sufficiently small, then bias towards height dim to reduce
  // the number of inactive invocations.
  if (global_workgroup_size[0u] < 6u) {
    return {4u, 16u, 1u};
  }
  return {16u, 4u, 1u};
}

utils::uvec3 pick_wc_square_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)args;
  (void)resize_args;
  // Some inactive invocations are okay; set 6 as the threshold to use the
  // a square wg size.
  if (global_workgroup_size[0u] >= 6 && global_workgroup_size[2u] >= 6) {
    return {8u, 1u, 8u};
  }
  // If channels dim is sufficiently small, then bias towards width dim to
  // reduce the number of inactive invocations.
  if (global_workgroup_size[2u] < 2u) {
    return {64u, 1u, 1u};
  }
  return {16u, 1u, 4u};
}

} // namespace vkcompute
