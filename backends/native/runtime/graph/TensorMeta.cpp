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

Dim::Dim(int64_t min_v, int64_t max_v) : min(min_v), max(max_v) {
  if (min_v < 0 || (max_v >= 0 && max_v < min_v)) {
    throw std::runtime_error(
        "Dim: no shape has the range " + std::to_string(min_v) + ".." +
        std::to_string(max_v));
  }
}

bool TensorMeta::is_static() const {
  return std::ranges::all_of(sizes, &Dim::is_static);
}

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
  for (const Dim& d : sizes) {
    const int64_t extent = d.is_static() ? d.min : d.max;
    if (extent < 0) {
      throw std::runtime_error("TensorMeta::numel: unbounded dynamic dim");
    }
    // Signed overflow is UB, so the product must be checked before it happens.
    if (extent != 0 && n > std::numeric_limits<int64_t>::max() / extent) {
      throw std::runtime_error("TensorMeta::numel: element count overflows");
    }
    n *= extent;
  }
  return n;
}

std::string TensorMeta::to_string() const {
  std::string s = scalar_type_name(dtype);
  s += "[";
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (i != 0) {
      s += ",";
    }
    const Dim& d = sizes[i];
    if (d.is_static()) {
      s += std::to_string(d.min);
    } else if (d.max < 0) {
      s += std::to_string(d.min) + "..?";
    } else {
      s += std::to_string(d.min) + ".." + std::to_string(d.max);
    }
  }
  s += "]";
  return s;
}

} // namespace ptn
