/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

namespace example {
class AttentionSinkRopeRunner {
 public:
  AttentionSinkRopeRunner(executorch::extension::Module* module);

  /**
   * This function removes evict_batch_size tokens from the KV cache
   * and updates the position shift accordingly.
   * @return The error code.
   */
  executorch::runtime::Error evict_token(
      const std::string& method_name,
      std::vector<executorch::runtime::EValue>& inputs);

  /**
   * Once KV Cache output data pointer change, need to set
   * the output for specify method name in the module.
   * @return The error code.
   */
  executorch::runtime::Error set_outputs(
      const std::string& method_name,
      std::vector<executorch::runtime::EValue> output_values);

  /**
   * Load the Module for attention sink purpose.
   * @return The error code.
   */
  executorch::runtime::Error load(const std::vector<std::string>& method_names);
  /**
   * Check if the required methods in the Module is loaded.
   * @return True if the Module is loaded, false otherwise.
   */
  bool is_method_loaded(const std::vector<std::string>& method_names);

  int get_position_shift() {
    return position_shift_;
  }
  int get_eviction_batch_size() {
    return eviction_batch_size_;
  }

 protected:
  executorch::extension::Module* module_;
  int position_shift_ = 0;
  int64_t eviction_batch_size_ = 0;
};
} // namespace example
