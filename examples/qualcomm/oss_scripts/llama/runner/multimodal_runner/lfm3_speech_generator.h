/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/lfm3/speech_policy.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_token_generator.h>
#include <array>
#include <deque>
#include <optional>
#include <random>

namespace example {

/**
 * @class Lfm3SpeechTokenGenerator
 * @brief LFM3-Omni text + speech generation.
 *
 * Each step either samples a text token from the decoder logits or, in a speech
 * phase, a speech frame of `kNumCodebooks` codes from the decoder's final
 * hidden state with the depth decoder (MaskGIT: codebooks revealed in a random
 * order over passes of 4). The next decoder input is the token's embedding or
 * the sum of the frame's codebook embeddings. Mirrors the HF `generate` mode
 * machine: `<|audio_start|>` opens speech, `<|mixed_start|>` opens interleaved
 * text/speech blocks (6 text, 12 speech slots), an end-of-speech code returns
 * to text.
 */
class Lfm3SpeechTokenGenerator : public MultimodalTokenGenerator {
 public:
  static constexpr int kNumCodebooks = 12;
  using Frame = std::array<int32_t, kNumCodebooks>;

  struct SpeechConfig {
    float hidden_scale;
    int32_t hidden_zero_point;
    std::optional<float> audio_temperature;
    std::optional<float> audio_repetition_penalty;
    uint32_t seed = 0;
    int32_t codebook_size = 0;
    int32_t audio_eos_code = 0;
    float logits_scale = 0;
    int32_t logits_zero_point = 0;
  };

  Lfm3SpeechTokenGenerator(
      tokenizers::Tokenizer* tokenizer,
      TokenEmbeddingProcessor* tok_embedding_runner,
      DecoderRunner* decoder_runner,
      KVManager* kv_manager,
      const std::string& method_name,
      std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
      Metadata metadata,
      executorch::llm::Stats* stats,
      std::unique_ptr<executorch::extension::MethodMeta> method_meta,
      executorch::extension::Module* depth,
      executorch::extension::Module* embed_frame,
      SpeechConfig speech_config);

  executorch::runtime::Error load_speech();

  /**
   * @brief Final prompt outputs; the first text slot uses the same policy as
   * decode.
   */
  void set_prompt_outputs(const uint16_t* hidden, const uint16_t* logits);

  executorch::runtime::Result<int64_t> generate(
      std::vector<uint64_t> tokens,
      int64_t start_pos,
      int32_t seq_len,
      std::function<void(const std::string&)> token_callback,
      bool dump_logits,
      AttentionSinkRopeRunner* attention_sink_rope_runner) override;

  const std::vector<Frame>& frames() const {
    return frames_;
  }

 private:
  void dequantize_hidden(const uint16_t* hidden);
  executorch::runtime::Result<Frame> sample_frame();
  int32_t sample_code(const float* logits, int codebook);
  uint64_t sample_text();
  executorch::runtime::Error embed_frame(const Frame& frame);
  void embed_token(uint64_t token);
  executorch::runtime::Error step(int64_t pos);

  uint64_t audio_start_{0};
  uint64_t text_end_{0};
  uint64_t mixed_start_{0};
  uint64_t audio_out_{0};
  std::array<uint64_t, 4> suppressed_{};
  executorch::runtime::Result<uint64_t> special_token(const char* spelling);

  executorch::extension::Module* depth_;
  executorch::extension::Module* embed_frame_;
  SpeechConfig speech_config_;
  std::mt19937 rng_;
  std::vector<float> hidden_f_;
  std::vector<Frame> frames_;
  std::deque<Frame> history_;
  std::deque<int32_t> text_history_;
  speech::SamplingPolicy text_policy_;
  speech::SamplingPolicy audio_policy_;
  size_t text_repetition_window_ = 0;
  size_t audio_repetition_window_ = 0;
};

} // namespace example
