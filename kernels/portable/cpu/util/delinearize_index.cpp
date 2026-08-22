/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/kernels/portable/cpu/util/delinearize_index.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace torch::executor {
void delinearize_index(
    size_t linear_index,
    executorch::aten::ArrayRef<Tensor::SizesType> shape,
    size_t* out_indexes,
    const size_t out_indexes_len) {
  ET_CHECK(shape.size() <= out_indexes_len);
  for (size_t i = 0; i < shape.size(); ++i) {
    auto dim = shape.size() - 1 - i;
    auto dim_size = shape[dim];
    out_indexes[dim] = linear_index % dim_size;
    linear_index /= dim_size;
  }
}

void delinearize_index(
    size_t linear_index,
    const Tensor& t,
    size_t* out_indexes,
    const size_t out_indexes_len) {
  delinearize_index(linear_index, t.sizes(), out_indexes, out_indexes_len);
}
} // namespace torch::executor
