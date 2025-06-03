/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace torch::executor {
/**
 * Delinearize a flattened index to per-dimension indexes.
 *
 * @param[in] linear_index The flattened index
 * @param[in] shape The tensor shape
 * @param[out] out_indexes The per-dimension indexes
 * @param[in] out_indexes_len The maximum size of the out_indexes array
 * @returns void
 */
void delinearize_index(
    size_t linear_index,
    executorch::aten::ArrayRef<Tensor::SizesType> shape,
    size_t* out_indexes,
    const size_t out_indexes_len);

/**
 * Delinearize a flattened index to per-dimension indexes.
 *
 * @param[in] linear_index The flattened index
 * @param[in] t The tensor object
 * @param[out] out_indexes The per-dimension indexes
 * @param[in] out_indexes_len The maximum size of the out_indexes array
 * @returns void
 */
void delinearize_index(
    size_t linear_index,
    const Tensor& t,
    size_t* out_indexes,
    const size_t out_indexes_len);
} // namespace torch::executor
