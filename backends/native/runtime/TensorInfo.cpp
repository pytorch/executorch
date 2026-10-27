// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/TensorInfo.h>

#include <limits>
#include <stdexcept>
#include <utility>

#include <executorch/backends/native/runtime/deserialize/CheckedMath.h>

namespace ptn {

TensorInfo::TensorInfo(
    ScalarType dtype,
    std::vector<int64_t> sizes,
    std::vector<uint8_t> dim_order)
    : dtype_(dtype), sizes_(std::move(sizes)) {
  if (sizes_.size() > std::numeric_limits<uint8_t>::max()) {
    throw std::runtime_error("tensor rank is not representable");
  }
  if (dim_order.empty()) {
    dim_order.reserve(sizes_.size());
    for (size_t i = 0; i < sizes_.size(); ++i) {
      dim_order.push_back(static_cast<uint8_t>(i));
    }
  }
  if (dim_order.size() != sizes_.size()) {
    throw std::runtime_error("tensor dimension order has the wrong rank");
  }

  std::vector<bool> seen_dims(sizes_.size());
  for (const uint8_t dim : dim_order) {
    if (dim >= seen_dims.size() || seen_dims.at(dim)) {
      throw std::runtime_error("tensor dimension order is not a permutation");
    }
    seen_dims.at(dim) = true;
  }
  dim_order_ = std::move(dim_order);

  numel_ = 1;
  for (const int64_t size : sizes_) {
    if (size < 0 ||
        static_cast<uint64_t>(size) > std::numeric_limits<size_t>::max() ||
        !detail::checked_mul(numel_, static_cast<size_t>(size), numel_)) {
      throw std::runtime_error("tensor element count is not representable");
    }
  }
  if (!detail::checked_mul(numel_, element_size(dtype_), nbytes_)) {
    throw std::runtime_error("tensor byte size is not representable");
  }

  strides_.resize(sizes_.size());
  int64_t stride = 1;
  for (auto it = dim_order_.rbegin(); it != dim_order_.rend(); ++it) {
    const size_t dim = *it;
    strides_[dim] = stride;
    const int64_t extent = sizes_[dim] == 0 ? 1 : sizes_[dim];
    if (stride > std::numeric_limits<int64_t>::max() / extent) {
      throw std::runtime_error("tensor stride is not representable");
    }
    stride *= extent;
  }
}

} // namespace ptn
