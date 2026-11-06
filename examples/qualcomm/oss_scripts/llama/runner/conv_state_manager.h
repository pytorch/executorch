/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/method_meta.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace example {

/**
 * @class ConvStateManager
 * @brief Rolling state of short-convolution layers (e.g. LFM2/LFM3).
 *
 * Each conv layer reads a state of the last `state_len` conv inputs,
 * [1, state_len, dim], and outputs the conv inputs of the current AR tokens,
 * [1, ar_len, dim], in the same quantized encoding. After every step the last
 * `state_len` valid rows of (state ++ output) become the next state.
 */
class ConvStateManager {
 public:
  struct Metadata {
    int64_t num_layers;
    int64_t state_len;
    int64_t dim;
    int32_t max_ar_len;
    size_t element_size;
  };
  explicit ConvStateManager(Metadata metadata);

  /**
   * @brief Create a manager if the model has conv layers (`get_n_conv_layers`
   * metadata), otherwise nullptr. State length, dim and dtype come from the
   * first conv-state input of `method_meta`.
   */
  static executorch::runtime::Result<std::unique_ptr<ConvStateManager>>
  from_module(
      executorch::extension::Module& module,
      const executorch::runtime::MethodMeta& method_meta,
      int32_t max_ar_len);

  void init(IMemAlloc* buffer_manager);

  /**
   * @brief Bind conv states and conv outputs to a method. They must be the
   * last `num_layers` inputs and outputs of the method, appended right after
   * everything already in `input_tensors`/`output_tensors`.
   */
  void bind_io(
      IMemAlloc* buffer_manager,
      const executorch::runtime::MethodMeta& method_meta,
      std::vector<std::unique_ptr<executorch::aten::TensorImpl>>& tensors,
      std::vector<executorch::aten::Tensor>& input_tensors,
      std::vector<executorch::aten::Tensor>& output_tensors);

  /// Clear all states.
  void reset();

  /// Shift the first `n_update` rows of each layer's output into its state.
  void update(int32_t n_update);

  std::byte* state(int64_t layer) const {
    return states_[layer];
  }
  std::byte* output(int64_t layer) const {
    return outputs_[layer];
  }
  int64_t num_layers() const {
    return metadata_.num_layers;
  }
  size_t total_size_in_bytes() const {
    return metadata_.num_layers * (state_nbytes() + output_nbytes());
  }

 private:
  size_t row_nbytes() const {
    return metadata_.dim * metadata_.element_size;
  }
  size_t state_nbytes() const {
    return metadata_.state_len * row_nbytes();
  }
  size_t output_nbytes() const {
    return metadata_.max_ar_len * row_nbytes();
  }

  Metadata metadata_;
  std::vector<std::byte*> states_;
  std::vector<std::byte*> outputs_;
};

} // namespace example
