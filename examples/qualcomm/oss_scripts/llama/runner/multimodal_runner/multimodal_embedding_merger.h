/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/utils.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <array>
#include <memory>
#include <vector>

namespace example {

class MultimodalEmbeddingMerger {
 public:
  explicit MultimodalEmbeddingMerger(int32_t embedding_dim);

  /**
   * @brief Reset the merger state for a new sequence
   */
  void reset();

  // Append embeddings
  void add_embeddings(const TensorStruct<float>& embedding);
  void add_embeddings(const executorch::aten::Tensor& embedding);

  TensorStruct<float> get_merged_embeddings();
  int32_t get_total_tokens() const noexcept {
    return total_tokens_;
  }

 private:
  // Validates shape [batch size, num tokens, dim] and appends data.
  void append_data(
      const float* data,
      ssize_t ndim,
      const executorch::aten::SizesType* sizes);

  int32_t embedding_dim_;
  int32_t total_tokens_{0};

  // merged embeddings are holded in this vector.
  std::vector<float> embeddings_;
  std::array<executorch::aten::TensorImpl::SizesType, 3> sizes_{};
};

} // namespace example
