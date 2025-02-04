/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/operator_registry.h>

namespace executorch {
namespace runtime {

namespace testing {

inline void make_kernel_key(
    std::vector<std::pair<
        executorch::aten::ScalarType,
        std::vector<executorch::aten::DimOrderType>>> tensors,
    char* buf) {
  std::vector<TensorMeta> meta;
  for (auto& t : tensors) {
    Span<executorch::aten::DimOrderType> dim_order(
        t.second.data(), t.second.size());
    meta.emplace_back(t.first, dim_order);
  }
  Span<const TensorMeta> metadata(meta.data(), meta.size());
  internal::make_kernel_key_string(metadata, buf);
}

} // namespace testing

} // namespace runtime
} // namespace executorch
