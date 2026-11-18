/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/kernel/kernel_includes.h>

#include <optional>

namespace torch {
namespace executor {

// aten::_embedding_bag's `mode` argument.
constexpr int64_t kEmbeddingBagModeSum = 0;
constexpr int64_t kEmbeddingBagModeMean = 1;
constexpr int64_t kEmbeddingBagModeMax = 2;

// The parts of ATen's CPU path choice that change the outputs.
struct EmbeddingBagLayout {
  // offset2bag holds one bag id per index; otherwise it is empty.
  bool fills_offset2bag;
  // With include_last_offset, the last sum/mean bag stops at the final
  // offset, while offset2bag and the mean divisor still count the indices
  // after it. Otherwise the last bag takes every remaining index.
  bool truncate_last_bag;
};

// Validates the arguments, including every offsets and indices value, then
// resizes the four outputs to the shapes ATen gives them and reports the
// layout. On success, kernels can index weight and every output without bounds
// checks.
Error check_embedding_bag_args_and_resize_outputs(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    int64_t mode,
    const std::optional<Tensor>& per_sample_weights,
    bool include_last_offset,
    int64_t padding_idx,
    Tensor& out,
    Tensor& offset2bag,
    Tensor& bag_size,
    Tensor& max_indices,
    EmbeddingBagLayout& layout);

} // namespace executor
} // namespace torch
