/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_dimension_limit.h>

namespace torch::executor {

namespace internal {
class DelinearizedIndexesIterator {
 public:
  using difference_type = ssize_t;
  using value_type = std::array<std::size_t, ::executorch::runtime::kTensorDimensionLimit>;
  using reference = const value_type&;
  using pointer = const value_type*;
  using iterator_category = std::forward_iterator_tag;

  DelinearizedIndexesIterator() = default;

  explicit DelinearizedIndexesIterator(const Tensor& t)
      : idx_(0), dim_(t.dim()), shape_(t.sizes()) {
  }

  struct make_end_t {
    explicit constexpr make_end_t() = default;
  };

  DelinearizedIndexesIterator(make_end_t, const Tensor& t)
      : idx_(t.numel()) {}

  bool operator==(const DelinearizedIndexesIterator& rhs) const {
    return idx_ == rhs.idx_;
  }

  bool operator!=(const DelinearizedIndexesIterator& rhs) const {
    return !operator==(rhs);
  }

  reference operator*() const {
    return repr_;
  }

  pointer operator->() const {
    return &repr_;
  }

  DelinearizedIndexesIterator& operator++() {
    idx_++;
    for (auto ii = dim_ - 1; ii >= 0; --ii) {
      repr_[ii]++;
      ET_DCHECK(repr_[ii] <= shape_[ii]);
      if ET_LIKELY (repr_[ii] < shape_[ii]) {
        break;
      } else {
        repr_[ii] = 0;
      }
    }
    return *this;
  }

  DelinearizedIndexesIterator operator++(int) {
    auto it = *this;
    operator++();
    return it;
  }

  difference_type operator-(const DelinearizedIndexesIterator& rhs) const {
    return difference_type(idx_ - rhs.idx_);
  }

 private:
  std::size_t idx_ = 0;
  value_type repr_ = {0,};
  ssize_t dim_;
  ArrayRef<exec_aten::SizesType> shape_;
};
} // namespace internal

class DelinearizedIndexesRange {
 public:
  using iterator = internal::DelinearizedIndexesIterator;

  DelinearizedIndexesRange(const Tensor& t) :
      tensor_(t) {}

  iterator begin() const {
    return iterator(tensor_);
  }

  iterator end() {
    return iterator(iterator::make_end_t(), tensor_);
  }
 private:
  const Tensor& tensor_;
};
} // namespace torch::executor
