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
#include <string>
#include <unordered_set>
#include <vector>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace executorch::extension::asr {

using ::executorch::extension::Module;
using ::executorch::extension::llm::get_eos_ids;
using ::executorch::extension::llm::load_tokenizer;
using ::executorch::extension::llm::print_report;
using ::executorch::extension::llm::Sampler;
using ::executorch::extension::llm::Stats;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

/**
 * Configuration for the ASR transcription loop.
 *
 * max_new_tokens controls the number of tokens generated after the prompt.
 * Temperature controls the randomness of the output.
 */
struct AsrTranscribeConfig {
  int64_t max_new_tokens = 128;
  std::unordered_set<int64_t> eos_token_ids = {};
  float temperature = 0.0f;
  int64_t decoder_start_token_id = 0;
};

/**
 * Runner that owns a ASR model encoder + decoder pair exported as a single
 * ExecuTorch module. A good example is Whisper
 * (https://huggingface.co/openai/whisper-small)
 *
 * The module is expected to expose two callable methods:
 *  - "encoder": processes precomputed audio features into encoder states.
 *  - "text_decoder": consumes the decoder input ids, encoder output and cache
 *    positions to autoregressively generate logits.
 */
class AsrRunner {
 public:
  AsrRunner(
      std::string module_path,
      std::string data_path,
      std::string tokenizer_path);

  /**
   * Returns true when the module and tokenizer are ready for inference.
   */
  bool is_loaded() const;

  /**
   * Loads the module, validates required methods and initialises tokenizer.
   */
  ::executorch::runtime::Error load();

  /**
   * Executes an end-to-end transcription cycle.
   *
   * @param preprocessed_features Audio features already processed by a
   * preprocessor module (see voxtral example).
   * @param config Controls generation length and termination criteria.
   * @param token_callback Optional functor invoked for each decoded piece of
   * text emitted during generation.
   *
   * @returns Result containing the final decoder token ids (including the seed
   * prompt and generated tokens), or an error.
   */
  ::executorch::runtime::Result<std::vector<int64_t>> transcribe(
      ::executorch::extension::TensorPtr preprocessed_features,
      AsrTranscribeConfig config = {},
      std::function<void(const std::string&)> token_callback = {});

 private:
  ::executorch::runtime::Error load_tokenizer();
  inline const std::unordered_set<int64_t>& eos_token_ids() const {
    return eos_token_ids_;
  }

  /**
   * Sample the next token from the logits tensor.
   * @param logits_tensor The logits tensor.
   * @param temperature The temperature parameter used to control randomness in
   * sampling.
   * @return The next token.
   */
  inline int32_t logits_to_token(
      const executorch::aten::Tensor& logits_tensor,
      const float temperature = 0.0f) {
    int32_t result = 0;

    // Create a minimal context for error handling in ET_SWITCH
    struct {
      [[noreturn]] void fail(torch::executor::Error /* error */) {
        ET_CHECK_MSG(false, "Unsupported dtype in logits_to_token");
      }
    } ctx;

    ET_SWITCH_FOUR_TYPES(
        Float,
        Half,
        BFloat16,
        UInt16,
        logits_tensor.scalar_type(),
        ctx,
        "logits_to_token",
        CTYPE,
        [&]() {
          // If the logit_tensor rank is 3, the shape is [batch, seq_length,
          // vocab_size], get the last logits, sample and return. Else the model
          // outputs the last logit, directly sample and return.
          auto* logits = logits_tensor.mutable_data_ptr<CTYPE>();
          ssize_t vocab_size = logits_tensor.size(logits_tensor.dim() - 1);
          if (logits_tensor.dim() == 3) {
            auto num_tokens = logits_tensor.size(1);
            logits += (num_tokens - 1) * vocab_size;
          }
          // @lint-ignore CLANGTIDY facebook-hte-Deprecated
          Sampler sampler(vocab_size, temperature);
          result = sampler.sample(logits);
        });
    return result;
  }

  std::string module_path_;
  std::string data_path_;
  std::string tokenizer_path_;

  std::unique_ptr<Module> module_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unordered_set<int64_t> eos_token_ids_;

  bool encoder_method_loaded_ = false;
  bool decoder_method_loaded_ = false;

  Stats stats_;
};

} // namespace executorch::extension::asr
