/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <set>

namespace vkcompute {

//
// Sum, Mean, Var output size calculation functions
//

std::vector<int64_t>
calc_out_sizes(std::vector<int64_t> in_sizes, int64_t dim, bool keepdim);

//
// Sum, Mean, Var dims to aggregate calculation functions
//

std::set<int64_t> calc_dims_to_aggregate(
    const std::vector<int64_t>& dims_to_sum,
    int64_t in_dim);

} // namespace vkcompute
