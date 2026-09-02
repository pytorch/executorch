// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/TensorMeta.h>

#include <algorithm>
#include <limits>
#include <ranges>
#include <stdexcept>

namespace ptn {

bool TensorMeta::is_contiguous() const {
  if (dim_order_hint.empty()) {
    return true;
  }
  // A length mismatch already makes this unequal.
  return std::ranges::equal(
      dim_order_hint,
      std::views::iota(int32_t{0}, static_cast<int32_t>(sizes.size())));
}

int64_t TensorMeta::numel() const {
  int64_t n = 1;
  for (const int64_t dim_size : sizes) {
    if (dim_size < 0) {
      throw std::runtime_error("TensorMeta::numel: negative extent");
    }
    // Signed overflow is UB, so the product must be checked before it happens.
    if (dim_size != 0 && n > std::numeric_limits<int64_t>::max() / dim_size) {
      throw std::runtime_error("TensorMeta::numel: element count overflows");
    }
    n *= dim_size;
  }
  return n;
}

} // namespace ptn
