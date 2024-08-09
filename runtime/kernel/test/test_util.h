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

namespace executorch {
namespace runtime {
// Defined in //executorch/runtime/kernel/operator_registry.cpp.
void make_kernel_key_string(ArrayRef<TensorMeta> key, char* buf);
} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
inline void make_kernel_key(
    std::vector<std::pair<ScalarType, std::vector<exec_aten::DimOrderType>>>
        tensors,
    char* buf) {
  std::vector<TensorMeta> meta;
  for (auto& t : tensors) {
    ArrayRef<exec_aten::DimOrderType> dim_order(
        t.second.data(), t.second.size());
    meta.emplace_back(t.first, dim_order);
  }
  auto meatadata = ArrayRef<TensorMeta>(meta.data(), meta.size());
  make_kernel_key_string(meatadata, buf);
}

} // namespace executor
} // namespace torch
