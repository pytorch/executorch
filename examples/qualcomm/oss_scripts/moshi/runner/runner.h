/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <executorch/extension/module/module.h>

namespace example {

class Runner {
 public:
  explicit Runner(
      const std::string& model_path,
      const std::string& output_path);

  struct Stats {
    // Total time to load the model and init the inputs
    long model_load_start_ms;
    long model_load_end_ms;

    // Total time to decode all chunks
    long decode_start_ms = 0;
    long decode_end_ms = 0;
  };

  executorch::runtime::Error parse_input_list(std::string& input_list);
  executorch::runtime::Error init_io();
  executorch::runtime::Error prepare_io();
  executorch::runtime::Error load(std::string& input_list);
  executorch::runtime::Error generate(std::string& input_list);

 private:
  Stats stats_;
  std::unique_ptr<executorch::extension::Module> module_;
  std::string method_name_;
  std::string output_path_;

  // Pair that stores IO data with its TensorImpl pointer
  std::pair<
      std::vector<std::vector<int32_t>>,
      std::shared_ptr<executorch::aten::TensorImpl>>
      encoded_input_list_;
  std::pair<std::vector<float>, std::shared_ptr<executorch::aten::TensorImpl>>
      k_cache_;
  std::pair<std::vector<float>, std::shared_ptr<executorch::aten::TensorImpl>>
      v_cache_;
  std::pair<std::vector<int64_t>, std::shared_ptr<executorch::aten::TensorImpl>>
      end_index_;
  std::pair<std::vector<int64_t>, std::shared_ptr<executorch::aten::TensorImpl>>
      end_offset_;
  std::vector<std::pair<
      std::vector<float>,
      std::shared_ptr<executorch::aten::TensorImpl>>>
      convtr_partials_;
  std::vector<std::pair<
      std::vector<float>,
      std::shared_ptr<executorch::aten::TensorImpl>>>
      conv_previous_;
  std::pair<
      std::vector<std::vector<float>>,
      std::shared_ptr<executorch::aten::TensorImpl>>
      decoded_output_list_;

  std::vector<executorch::runtime::EValue> inputs_;
  std::vector<executorch::aten::Tensor> input_tensors_;
  std::vector<executorch::aten::Tensor> output_tensors_;
};

} // namespace example
