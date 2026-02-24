/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace voxtral_realtime {

// Custom runner rather than MultimodalRunner because Voxtral Realtime sums
// audio and text embeddings at each position (element-wise add), while
// MultimodalRunner concatenates modality segments sequentially.

struct TranscribeConfig {
  int max_new_tokens = 500;
  float temperature = 0.0f; // 0 = greedy
};

using TokenCallback = std::function<void(const std::string&)>;

class StreamingSession;

class VoxtralRealtimeRunner {
 public:
  VoxtralRealtimeRunner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const std::string& preprocessor_path = "",
      bool warmup = true);

  // Offline transcription: full encoder first, then step-by-step decode.
  int transcribe(
      const float* audio_data,
      int64_t num_samples,
      const TranscribeConfig& config,
      TokenCallback token_cb);

  // Streaming transcription: processes raw audio incrementally via
  // StreamingSession. Requires a model exported with --streaming and
  // a streaming preprocessor .pte.
  std::unique_ptr<StreamingSession> create_streaming_session(
      const TranscribeConfig& config,
      TokenCallback token_cb);

  int64_t max_seq_len() const {
    return max_seq_len_;
  }
  int64_t vocab_size() const {
    return vocab_size_;
  }
  bool is_streaming() const {
    return is_streaming_;
  }

 private:
  friend class StreamingSession;

  std::unique_ptr<::executorch::extension::Module> model_;
  std::unique_ptr<::executorch::extension::Module> preprocessor_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

  // From model metadata (constant_methods)
  int64_t max_seq_len_ = 4096;
  int64_t vocab_size_ = 131072;
  int64_t dim_ = 3072;

  // Streaming metadata (from constant_methods, if present)
  bool is_streaming_ = false;
  int64_t num_mel_bins_ = 128;
  int64_t chunk_mel_len_ = 8;
  int64_t max_enc_len_ = 750;
  int64_t enc_dim_ = 1280;
  int64_t conv1_pad_ = 2;
  int64_t conv2_pad_ = 2;

  // Raw audio samples per streaming step (sampling_rate / frame_rate = 1280)
  int64_t step_samples_ = 1280;

  // STFT overlap for streaming mel computation (read from model metadata).
  int64_t stft_left_overlap_ = 320;
  int64_t stft_right_lookahead_ = 40;
  int64_t mel_skip_frames_ = 2;

  // Tokenizer special tokens
  uint64_t bos_id_ = 1;
  uint64_t eos_id_ = 2;

  // Run preprocessor.pte on raw audio -> mel spectrogram tensor.
  ::executorch::extension::TensorPtr run_preprocessor(
      const float* audio,
      int64_t num_samples);
};

// Streaming session: accepts raw audio incrementally via feed_audio(),
// computes mel spectrogram per step, and runs encoder+decoder in real-time.
class StreamingSession {
 public:
  StreamingSession(
      VoxtralRealtimeRunner& runner,
      TranscribeConfig config,
      TokenCallback token_cb);

  // Feed raw audio (16kHz float32). Processes as many complete 80ms steps
  // as possible. Returns number of new tokens generated.
  int feed_audio(const float* data, int64_t num_samples);

  // Signal end of audio. Pads last partial step, then generates remaining
  // text-only tokens until EOS or max_new_tokens. Returns total tokens
  // generated across the entire session.
  int flush();

  int total_tokens() const {
    return num_generated_;
  }

 private:
  VoxtralRealtimeRunner& runner_;
  TranscribeConfig config_;
  TokenCallback token_cb_;

  // Raw audio accumulation buffer
  std::vector<float> audio_buf_;
  int64_t samples_consumed_ = 0;

  // Encoder streaming state
  std::vector<float> conv1_state_;
  std::vector<float> conv2_state_;
  int64_t enc_frame_pos_ = 0;

  // Decoder state
  int64_t dec_pos_ = 0;
  uint64_t prev_token_;
  int num_generated_ = 0;
  bool eos_reached_ = false;
  bool flushed_ = false;

  ::executorch::extension::llm::Sampler sampler_;
  std::vector<float> input_embeds_buf_;

  // Process one 80ms step from the audio buffer.
  bool try_process_step();

  // Run one decoder step (token_embed + optional audio_embed -> logits).
  bool decode_step(const float* audio_embeds);
};

} // namespace voxtral_realtime
