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

//
// Tensor output size calculation functions
//

std::vector<int64_t> calculate_broadcasted_output_size(
    const std::vector<int64_t>& sizes1,
    const std::vector<int64_t>& sizes2);

//
// Tensor property checking functions
//

bool check_same_packed_dim(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out);

//
// Broadcast flag functions
//

bool is_packed_dim_broadcasted(
    ComputeGraph& graph,
    const ValueRef sndr,
    const ValueRef rcvr);

utils::ivec2 create_broadcast_params(
    ComputeGraph& graph,
    const ValueRef t1,
    const ValueRef t2);

//
// Work group size calculation functions
//

utils::uvec3 adaptive_work_group_size(const utils::uvec3& global_work_group);

//
// Tensor dim utilities
//

template <
    typename T,
    typename std::enable_if<
        std::is_integral<T>::value && std::is_signed<T>::value,
        int>::type = 0>
T normalize(const T& nchw_dim, const int64_t ndim) {
  return (nchw_dim % ndim + ndim) % ndim;
}

template <
    typename T,
    typename std::enable_if<
        std::is_integral<T>::value && std::is_signed<T>::value,
        int>::type = 0>
T nchw_dim_to_whcn_dim(const T& nchw_dim, const int64_t ndim) {
  return ndim - 1 - normalize(nchw_dim, ndim);
}

} // namespace vkcompute
