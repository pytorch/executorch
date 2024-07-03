/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/containers/Constant.h>

namespace vkcompute {

TensorRef::TensorRef(
    const std::vector<int64_t>& t_sizes,
    vkapi::ScalarType t_dtype,
    const void* const t_data)
    : sizes{}, dtype{t_dtype}, data{t_data} {
  size_t ndim = t_sizes.size();
  sizes.resize(ndim);
  for (int i = 0; i < ndim; ++i) {
    sizes[i] = t_sizes.at(i);
  }
}

} // namespace vkcompute
