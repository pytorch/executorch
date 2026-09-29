// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <executorch/backends/native/runtime/graph/ScalarType.h>

namespace ptn {

// Owned tensor requirements shared by metadata inspection and execution.
class TensorInfo final {
 public:
  TensorInfo(
      ScalarType dtype,
      std::vector<int64_t> sizes,
      std::vector<uint8_t> dim_order = {});

  ScalarType dtype() const {
    return dtype_;
  }
  std::span<const int64_t> sizes() const {
    return sizes_;
  }
  std::span<const uint8_t> dim_order() const {
    return dim_order_;
  }
  std::span<const int64_t> strides() const {
    return strides_;
  }
  size_t numel() const {
    return numel_;
  }
  size_t nbytes() const {
    return nbytes_;
  }

 private:
  ScalarType dtype_;
  std::vector<int64_t> sizes_;
  std::vector<uint8_t> dim_order_;
  std::vector<int64_t> strides_;
  size_t numel_{0};
  size_t nbytes_{0};
};

} // namespace ptn
