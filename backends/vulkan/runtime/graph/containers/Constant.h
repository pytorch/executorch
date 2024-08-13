/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/Context.h>

namespace vkcompute {

/*
 * Represents a reference to a tensor that has been
 * serialized with the model, such as a serialized weight
 * tensor. It contains some metadata as well as a raw
 * pointer to the data of the tensor, which is assumed to
 * be contiguous.
 */
struct TensorRef final {
  std::vector<int64_t> sizes;
  vkapi::ScalarType dtype;
  const void* data;

  explicit TensorRef(
      const std::vector<int64_t>& t_sizes,
      vkapi::ScalarType t_dtype,
      const void* const t_data);
};

} // namespace vkcompute
