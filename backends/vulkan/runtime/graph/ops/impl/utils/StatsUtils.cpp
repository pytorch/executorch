/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/StatsUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

namespace vkcompute {

//
// Sum, Mean, Var output size calculation functions
//

std::vector<int64_t>
calc_out_sizes(std::vector<int64_t> in_sizes, int64_t dim, bool keepdim) {
  std::vector<int64_t> output_sizes = in_sizes;
  if (keepdim) {
    output_sizes.at(dim) = 1;
  } else {
    output_sizes.erase(output_sizes.begin() + dim);
  }
  return output_sizes;
}

//
// Sum, Mean, Var dims to aggregate calculation functions
//

std::set<int64_t> calc_dims_to_aggregate(
    const std::vector<int64_t>& dims_to_sum,
    int64_t in_dim) {
  std::set<int64_t> dims_set;

  if (dims_to_sum.empty()) {
    // If dim is not specified, reduce over all dims
    for (int64_t i = 0; i < in_dim; ++i) {
      dims_set.insert(i);
    }
  } else {
    for (const auto& dim : dims_to_sum) {
      // Normalize (negative) dim into range [0, in_dim - 1]
      int64_t dim_normalized = normalize(dim, in_dim);
      dims_set.insert(dim_normalized);
    }
  }

  return dims_set;
}

} // namespace vkcompute
