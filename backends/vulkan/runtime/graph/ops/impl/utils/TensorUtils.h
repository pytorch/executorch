/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/api.h>

namespace at {
namespace native {
namespace vulkan {

//
// Tensor output size calculation functions
//

std::vector<int64_t> calculate_broadcasted_output_size(
    const vTensor& t1,
    const vTensor& t2);

//
// Tensor property checking functions
//

bool check_ndim_is(const vTensor& t, size_t ndim);

bool check_same_ndim(const vTensor& t1, const vTensor& t2);

bool check_same_sizes_at(
    const vTensor& t1,
    int64_t d1,
    const vTensor& t2,
    int64_t d2);

bool check_memory_layout_is(const vTensor& t, api::GPUMemoryLayout layout);

bool check_same_memory_layout(const vTensor& t1, const vTensor& t2);

bool check_same_memory_layout(
    const vTensor& t1,
    const vTensor& t2,
    const vTensor& t3);

bool check_broadcastable(const vTensor& t1, const vTensor& t2);

//
// Work Group Size Calculation Utilities
//

api::utils::uvec3 adaptive_work_group_size(
    const api::utils::uvec3& global_work_group);

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
