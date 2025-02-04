/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

namespace vkcompute {

//
// Tensor output size calculation functions
//

std::vector<int64_t> calculate_broadcasted_output_size(
    const api::vTensor& t1,
    const api::vTensor& t2);

//
// Tensor property checking functions
//

bool check_ndim_is(const api::vTensor& t, size_t ndim);

bool check_same_ndim(const api::vTensor& t1, const api::vTensor& t2);

bool check_same_sizes_at(
    const api::vTensor& t1,
    int64_t d1,
    const api::vTensor& t2,
    int64_t d2);

bool check_packed_dim_is(const api::vTensor& t, const int32_t packed_dim);

bool check_same_packed_dim(const api::vTensor& t1, const api::vTensor& t2);

bool check_same_packed_dim(
    const api::vTensor& t1,
    const api::vTensor& t2,
    const api::vTensor& t3);

//
// Broadcast flag functions
//

utils::ivec2 create_broadcast_params(
    const api::vTensor& t1,
    const api::vTensor& t2);

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
  return ndim - 1 - nchw_dim;
}

} // namespace vkcompute
