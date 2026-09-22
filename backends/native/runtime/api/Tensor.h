// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include <executorch/backends/native/runtime/graph/ScalarType.h>

namespace ptn {

// Borrowed caller storage. Strides are in elements and must describe a
// canonical contiguous layout; `nbytes` is the available allocation extent.
template <typename Data>
class TensorView final {
  static_assert(std::is_same_v<Data, void> || std::is_same_v<Data, const void>);

 public:
  TensorView(
      ScalarType dtype,
      std::span<const int64_t> sizes,
      std::span<const int64_t> strides,
      Data* data,
      size_t nbytes)
      : dtype_(dtype),
        sizes_(sizes),
        strides_(strides),
        data_(data),
        nbytes_(nbytes) {}

  ScalarType dtype() const {
    return dtype_;
  }
  std::span<const int64_t> sizes() const {
    return sizes_;
  }
  std::span<const int64_t> strides() const {
    return strides_;
  }
  Data* data() const {
    return data_;
  }
  size_t nbytes() const {
    return nbytes_;
  }

 private:
  ScalarType dtype_;
  std::span<const int64_t> sizes_;
  std::span<const int64_t> strides_;
  Data* data_;
  size_t nbytes_;
};

using ConstTensorView = TensorView<const void>;
using MutableTensorView = TensorView<void>;

} // namespace ptn
