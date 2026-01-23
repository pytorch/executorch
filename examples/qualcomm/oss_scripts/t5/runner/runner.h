/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple t5 runner that includes preprocessing and post processing
// logic.

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/t5/runner/decoder.h>
#include <executorch/examples/qualcomm/oss_scripts/t5/runner/encoder.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/runtime/core/error.h>
#include <pytorch/tokenizers/tokenizer.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace example {

class Runner {
 public:
  explicit Runner(
      const std::string& model_path,
      const std::string& tokenizer_model_path);

  struct Stats {
    // Scaling factor for timestamps - in this case, we use ms.
    const long SCALING_FACTOR_UNITS_PER_SECOND = 1000;
    // Time stamps for the different stages of the execution
    // model_load_start_ms: Model loading time
    long model_load_start_ms;
    long model_load_end_ms;

    // encoder inference time
    long encoder_inference_start_ms = 0;
    long encoder_inference_end_ms = 0;

    // decoder inference time
    long decoder_inference_start_ms = 0;
    long decoder_inference_end_ms = 0;
  };

  bool is_loaded() const;
  executorch::runtime::Error load();
  executorch::runtime::Error generate(
      int32_t seq_len,
      std::vector<std::vector<uint8_t>>& inputs,
      std::function<void(const std::string&)> token_callback = {});

 private:
  executorch::runtime::Error print_performance();
  uint64_t logits_to_token(const executorch::aten::Tensor& logits_tensor);
  // model
  std::unique_ptr<T5Encoder> encoder_;
  std::unique_ptr<T5Decoder> decoder_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<executorch::extension::llm::Sampler> sampler_;
  std::string tokenizer_model_path_;

  std::unordered_map<std::string, int64_t> metadata_;
  std::unique_ptr<std::unordered_set<uint64_t>> eos_ids_;

  int64_t num_generated_token_ = 0;
  Stats stats_;
};

} // namespace example
