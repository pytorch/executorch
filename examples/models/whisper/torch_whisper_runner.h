/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <tokenizers/tokenizers.h>
#include <torch/torch.h>

#include <executorch/extension/asr/runner/runner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include "torch_aoti_model.h"

namespace executorch::examples::whisper {

class TorchWhisperRunner {
 public:
  TorchWhisperRunner(
      std::string encoder_library,
      std::string decoder_library,
      std::string tokenizer_path,
      std::optional<std::string> encoder_weights = std::nullopt,
      std::optional<std::string> decoder_weights = std::nullopt,
      std::string device = "cpu");

  ::executorch::runtime::Error load();

  ::executorch::runtime::Result<std::vector<int64_t>> transcribe(
      const torch::Tensor& preprocessed_features,
      ::executorch::extension::asr::AsrTranscribeConfig config,
      std::function<void(const std::string&)> token_callback = {});

 private:
  ::executorch::runtime::Error load_tokenizer();
  bool is_loaded() const;
  torch::Tensor maybe_cast_encoder_input(const torch::Tensor& tensor);
  torch::Tensor build_decoder_input(int64_t token_id);
  torch::Tensor build_cache_position_tensor(int64_t cache_position);

  std::string encoder_library_;
  std::string decoder_library_;
  std::string tokenizer_path_;
  std::optional<std::string> encoder_weights_;
  std::optional<std::string> decoder_weights_;
  std::string device_;

  std::unique_ptr<TorchAOTIModel> encoder_model_;
  std::unique_ptr<TorchAOTIModel> decoder_model_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;

  std::unordered_set<int64_t> eos_token_ids_;
  ::executorch::extension::llm::Stats stats_;
};

} // namespace executorch::examples::whisper
