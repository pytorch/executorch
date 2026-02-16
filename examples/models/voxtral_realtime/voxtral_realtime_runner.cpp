/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "voxtral_realtime_runner.h"

#include <cstring>
#include <ctime>
#include <vector>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

namespace voxtral_realtime {

VoxtralRealtimeRunner::VoxtralRealtimeRunner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& preprocessor_path) {
  // Load the main model (.pte with audio_encoder, text_decoder,
  // token_embedding methods). Mmap avoids copying the file into memory.
  ET_LOG(Info, "Loading model from: %s", model_path.c_str());
  model_ = std::make_unique<Module>(model_path, Module::LoadMode::Mmap);
  auto load_error = model_->load();
  ET_CHECK_MSG(load_error == Error::Ok, "Failed to load model.");

  // Model metadata is exported as constant_methods (zero-input methods that
  // return a scalar). These tell the runner the model's dimensions so it can
  // allocate buffers and enforce limits.
  std::vector<EValue> empty;
  auto ms = model_->execute("max_seq_len", empty);
  auto vs = model_->execute("vocab_size", empty);
  auto dm = model_->execute("dim", empty);

  if (ms.ok())
    max_seq_len_ = ms.get()[0].toInt();
  if (vs.ok())
    vocab_size_ = vs.get()[0].toInt();
  if (dm.ok())
    dim_ = dm.get()[0].toInt();

  ET_LOG(
      Info,
      "Model: max_seq_len=%ld, vocab_size=%ld, dim=%ld",
      static_cast<long>(max_seq_len_),
      static_cast<long>(vocab_size_),
      static_cast<long>(dim_));

  // Detect streaming model (exported with --streaming flag).
  auto streaming_val = model_->execute("streaming", empty);
  if (streaming_val.ok() && streaming_val.get()[0].toInt() == 1) {
    is_streaming_ = true;

    auto nmb = model_->execute("num_mel_bins", empty);
    if (nmb.ok())
      num_mel_bins_ = nmb.get()[0].toInt();
    auto cm = model_->execute("chunk_mel_len", empty);
    if (cm.ok())
      chunk_mel_len_ = cm.get()[0].toInt();
    auto me = model_->execute("max_enc_len", empty);
    if (me.ok())
      max_enc_len_ = me.get()[0].toInt();
    auto ed = model_->execute("enc_dim", empty);
    if (ed.ok())
      enc_dim_ = ed.get()[0].toInt();
    auto c1 = model_->execute("conv1_pad", empty);
    if (c1.ok())
      conv1_pad_ = c1.get()[0].toInt();
    auto c2 = model_->execute("conv2_pad", empty);
    if (c2.ok())
      conv2_pad_ = c2.get()[0].toInt();

    auto ss = model_->execute("step_samples", empty);
    if (ss.ok())
      step_samples_ = ss.get()[0].toInt();

    auto slo = model_->execute("stft_left_overlap", empty);
    if (slo.ok())
      stft_left_overlap_ = slo.get()[0].toInt();
    auto srl = model_->execute("stft_right_lookahead", empty);
    if (srl.ok())
      stft_right_lookahead_ = srl.get()[0].toInt();
    auto msf = model_->execute("mel_skip_frames", empty);
    if (msf.ok())
      mel_skip_frames_ = msf.get()[0].toInt();

    ET_LOG(
        Info,
        "Streaming: chunk_mel=%ld, max_enc=%ld, enc_dim=%ld",
        static_cast<long>(chunk_mel_len_),
        static_cast<long>(max_enc_len_),
        static_cast<long>(enc_dim_));
  }

  // Tekken tokenizer (tekken.json) for the Mistral vocabulary.
  ET_LOG(Info, "Loading tokenizer from: %s", tokenizer_path.c_str());
  tokenizer_ = ::executorch::extension::llm::load_tokenizer(tokenizer_path);
  ET_CHECK_MSG(tokenizer_ != nullptr, "Failed to load tokenizer.");
  bos_id_ = tokenizer_->bos_tok();
  eos_id_ = tokenizer_->eos_tok();

  // Separate .pte that converts raw 16kHz audio waveform to mel spectrogram
  // (1, 128, T_mel). Exported from extension/audio/mel_spectrogram.py.
  if (!preprocessor_path.empty()) {
    ET_LOG(Info, "Loading preprocessor from: %s", preprocessor_path.c_str());
    preprocessor_ =
        std::make_unique<Module>(preprocessor_path, Module::LoadMode::Mmap);
    auto pp_error = preprocessor_->load();
    ET_CHECK_MSG(pp_error == Error::Ok, "Failed to load preprocessor.");
  }
}

TensorPtr VoxtralRealtimeRunner::run_preprocessor(
    const float* audio,
    int64_t num_samples) {
  ET_CHECK_MSG(preprocessor_ != nullptr, "No preprocessor loaded.");

  // Preprocessor expects a 1-D waveform: (num_samples,).
  auto audio_tensor = from_blob(
      const_cast<float*>(audio),
      {static_cast<int>(num_samples)},
      ::executorch::aten::ScalarType::Float);

  auto result =
      preprocessor_->execute("forward", std::vector<EValue>{*audio_tensor});
  ET_CHECK_MSG(result.ok(), "Preprocessor forward failed.");

  auto& outputs = result.get();
  ET_CHECK_MSG(
      !outputs.empty() && outputs[0].isTensor(), "Bad preprocessor output.");

  // Output is (1, 128, T_mel) channels-first mel spectrogram.
  // The data lives in the preprocessor Module's internal buffer and remains
  // valid until the next execute("forward") call (which we never make).
  auto mel_ref = outputs[0].toTensor();
  return from_blob(
      mel_ref.mutable_data_ptr<float>(),
      {static_cast<int>(mel_ref.size(0)),
       static_cast<int>(mel_ref.size(1)),
       static_cast<int>(mel_ref.size(2))},
      ::executorch::aten::ScalarType::Float);
}

int VoxtralRealtimeRunner::transcribe(
    const float* audio_data,
    int64_t num_samples,
    const TranscribeConfig& config,
    TokenCallback token_cb) {
  // --- Step 1: Preprocess raw audio to mel spectrogram ---
  ET_CHECK_MSG(preprocessor_ != nullptr, "No preprocessor provided.");
  TensorPtr mel = run_preprocessor(audio_data, num_samples);

  // --- Step 2: Encode mel to audio embeddings ---
  // audio_encoder: (1, 128, T_mel) -> (1, T_audio, 3072)
  // T_audio = T_mel / 8 (conv stride 2, then downsample by 4).
  auto enc_result = model_->execute("audio_encoder", std::vector<EValue>{*mel});
  ET_CHECK_MSG(enc_result.ok(), "audio_encoder failed.");

  auto& enc_outputs = enc_result.get();
  ET_CHECK_MSG(
      !enc_outputs.empty() && enc_outputs[0].isTensor(),
      "Bad audio_encoder output.");

  // audio_embeds data lives in the audio_encoder Method's output buffer,
  // valid until the next audio_encoder call (which we never make).
  auto audio_embeds = enc_outputs[0].toTensor();
  const int64_t t_audio = audio_embeds.size(1);
  ET_LOG(
      Info,
      "Audio: %ld samples -> %ld frames",
      static_cast<long>(num_samples),
      static_cast<long>(t_audio));

  // --- Step 3: Autoregressive decode ---
  // At each position:
  //   - If pos < t_audio: input = audio_embeds[pos] + token_embed(prev_token)
  //   - If pos >= t_audio: input = token_embed(prev_token) (text-only)
  // This element-wise sum is the key difference from standard multimodal
  // models which concatenate modality segments sequentially.
  uint64_t prev_token = bos_id_;
  int num_generated = 0;
  const int64_t max_pos = std::min(
      static_cast<int64_t>(config.max_new_tokens) + t_audio, max_seq_len_);

  std::vector<float> input_embeds_buf(static_cast<size_t>(dim_));

  // Token sampler with xorshift RNG, seeded from wall clock.
  ::executorch::extension::llm::Sampler sampler(
      static_cast<int32_t>(vocab_size_),
      config.temperature,
      ::executorch::extension::llm::kTopp,
      static_cast<unsigned long long>(std::time(nullptr)));

  for (int64_t pos = 0; pos < max_pos; pos++) {
    // a. Look up embedding for the previous token.
    int64_t token_id = static_cast<int64_t>(prev_token);
    auto token_tensor =
        from_blob(&token_id, {1, 1}, ::executorch::aten::ScalarType::Long);

    auto tok_result =
        model_->execute("token_embedding", std::vector<EValue>{*token_tensor});
    ET_CHECK_MSG(tok_result.ok(), "token_embedding failed.");
    auto tok_embed = tok_result.get()[0].toTensor();
    const float* tok_data = tok_embed.const_data_ptr<float>();

    // b. Sum audio + token embeddings (or token-only after audio ends).
    if (pos < t_audio) {
      const float* audio_frame =
          audio_embeds.const_data_ptr<float>() + pos * dim_;
      for (int64_t i = 0; i < dim_; i++) {
        input_embeds_buf[static_cast<size_t>(i)] = audio_frame[i] + tok_data[i];
      }
    } else {
      std::memcpy(
          input_embeds_buf.data(),
          tok_data,
          static_cast<size_t>(dim_) * sizeof(float));
    }

    auto input_embeds = from_blob(
        input_embeds_buf.data(),
        {1, 1, static_cast<int>(dim_)},
        ::executorch::aten::ScalarType::Float);

    // c. Run one decoder step. KV cache is updated internally by the model.
    auto cache_pos = from_blob(&pos, {1}, ::executorch::aten::ScalarType::Long);

    auto dec_result = model_->execute(
        "text_decoder", std::vector<EValue>{*input_embeds, *cache_pos});
    ET_CHECK_MSG(dec_result.ok(), "text_decoder failed.");

    auto logits = dec_result.get()[0].toTensor();

    // d. Sample next token from logits. Safe to mutate the output buffer
    //    since text_decoder overwrites it on the next execute() call.
    float* logits_data =
        logits.mutable_data_ptr<float>() + (logits.numel() - vocab_size_);
    int64_t next_token = static_cast<int64_t>(sampler.sample(logits_data));
    num_generated++;

    // e. Decode token to text and emit via callback.
    auto piece =
        tokenizer_->decode(prev_token, static_cast<uint64_t>(next_token));
    if (piece.ok()) {
      token_cb(*piece);
    }

    // f. Stop on end-of-sequence.
    if (static_cast<uint64_t>(next_token) == eos_id_) {
      break;
    }

    prev_token = static_cast<uint64_t>(next_token);
  }

  return num_generated;
}

// ---------------------------------------------------------------------------
// StreamingSession
// ---------------------------------------------------------------------------

std::unique_ptr<StreamingSession>
VoxtralRealtimeRunner::create_streaming_session(
    const TranscribeConfig& config,
    TokenCallback token_cb) {
  ET_CHECK_MSG(is_streaming_, "Model was not exported with --streaming.");
  ET_CHECK_MSG(
      preprocessor_ != nullptr,
      "No preprocessor loaded. Provide --preprocessor_path.");
  return std::make_unique<StreamingSession>(*this, config, std::move(token_cb));
}

StreamingSession::StreamingSession(
    VoxtralRealtimeRunner& runner,
    TranscribeConfig config,
    TokenCallback token_cb)
    : runner_(runner),
      config_(config),
      token_cb_(std::move(token_cb)),
      prev_token_(runner.bos_id_),
      sampler_(
          static_cast<int32_t>(runner.vocab_size_),
          config.temperature,
          ::executorch::extension::llm::kTopp,
          static_cast<unsigned long long>(std::time(nullptr))),
      input_embeds_buf_(static_cast<size_t>(runner.dim_)) {
  // Initialize conv states to zero (matches offline encoder's left-padding).
  // num_mel_bins=128, conv1_pad_=2 → 128*2 = 256 floats
  conv1_state_.assign(
      static_cast<size_t>(runner.num_mel_bins_ * runner.conv1_pad_), 0.0f);
  // enc_dim_=1280, conv2_pad_=2 → 1280*2 = 2560 floats
  conv2_state_.assign(
      static_cast<size_t>(runner.enc_dim_ * runner.conv2_pad_), 0.0f);
}

int StreamingSession::feed_audio(const float* data, int64_t num_samples) {
  audio_buf_.insert(audio_buf_.end(), data, data + num_samples);

  int new_tokens = 0;
  while (!eos_reached_ && try_process_step()) {
    new_tokens++;
  }

  // Trim consumed audio to bound memory growth. Keep stft_left_overlap_
  // samples before samples_consumed_ for the next step's left context.
  int64_t keep_from = samples_consumed_ - runner_.stft_left_overlap_;
  if (keep_from > 0) {
    audio_buf_.erase(
        audio_buf_.begin(),
        audio_buf_.begin() + static_cast<size_t>(keep_from));
    samples_consumed_ -= keep_from;
  }

  return new_tokens;
}

bool StreamingSession::try_process_step() {
  const int64_t step = runner_.step_samples_;
  const int64_t left_overlap = runner_.stft_left_overlap_;
  const int64_t right_lookahead = runner_.stft_right_lookahead_;
  const int64_t mel_skip = runner_.mel_skip_frames_;
  const int64_t chunk_mel_len = runner_.chunk_mel_len_;

  // Need enough audio for: current step + right look-ahead.
  // Left overlap comes from audio before samples_consumed_ (already in buffer).
  const int64_t need_end = samples_consumed_ + step + right_lookahead;
  if (static_cast<int64_t>(audio_buf_.size()) < need_end) {
    return false;
  }

  // Guard: encoder/decoder cache capacity.
  const int64_t enc_frames_per_chunk = chunk_mel_len / 2;
  if (enc_frame_pos_ + enc_frames_per_chunk > runner_.max_enc_len_ ||
      dec_pos_ >= runner_.max_seq_len_) {
    return false;
  }

  // --- Build the overlapping audio window ---
  // Window: [left_overlap] + [step (1280)] + [right_lookahead (40)] = 1640
  // samples For the first step (samples_consumed_=0), left side is zero-padded.
  const int64_t window_size = left_overlap + step + right_lookahead;
  std::vector<float> window_buf(static_cast<size_t>(window_size), 0.0f);

  // Left overlap: copy from audio_buf_ before samples_consumed_
  int64_t left_start = samples_consumed_ - left_overlap;
  if (left_start >= 0) {
    std::memcpy(
        window_buf.data(),
        audio_buf_.data() + left_start,
        static_cast<size_t>(left_overlap) * sizeof(float));
  } else {
    // Partial left overlap (first step): zero-pad then copy available
    int64_t available_left = samples_consumed_;
    int64_t zero_pad = left_overlap - available_left;
    // window_buf[0..zero_pad) is already 0.0f
    if (available_left > 0) {
      std::memcpy(
          window_buf.data() + zero_pad,
          audio_buf_.data(),
          static_cast<size_t>(available_left) * sizeof(float));
    }
  }

  // Step + right look-ahead
  std::memcpy(
      window_buf.data() + left_overlap,
      audio_buf_.data() + samples_consumed_,
      static_cast<size_t>(step + right_lookahead) * sizeof(float));

  // --- Compute mel spectrogram on the full window ---
  auto audio_tensor = from_blob(
      window_buf.data(),
      {static_cast<int>(window_size)},
      ::executorch::aten::ScalarType::Float);

  auto mel_result = runner_.preprocessor_->execute(
      "forward", std::vector<EValue>{*audio_tensor});
  ET_CHECK_MSG(mel_result.ok(), "Streaming preprocessor failed.");

  auto mel = mel_result.get()[0].toTensor();
  // mel shape: (1, 128, 10) — 10 frames from 1640 samples with center=True
  const int64_t num_mel_bins = mel.size(1);
  const int64_t total_mel_frames = mel.size(2);

  ET_CHECK_MSG(
      total_mel_frames >= mel_skip + chunk_mel_len,
      "Preprocessor produced fewer mel frames than expected.");

  // --- Extract frames [mel_skip, mel_skip+8) = frames 2-9 ---
  // These align exactly with the offline mel frames for this step.
  // Output layout is channels-first: (1, 128, T). For each channel,
  // copy 8 contiguous frames starting at offset mel_skip.
  std::vector<float> mel_chunk_buf(
      static_cast<size_t>(num_mel_bins * chunk_mel_len));
  const float* mel_data = mel.const_data_ptr<float>();
  for (int64_t c = 0; c < num_mel_bins; c++) {
    std::memcpy(
        mel_chunk_buf.data() + c * chunk_mel_len,
        mel_data + c * total_mel_frames + mel_skip,
        static_cast<size_t>(chunk_mel_len) * sizeof(float));
  }

  auto mel_chunk = from_blob(
      mel_chunk_buf.data(),
      {1, static_cast<int>(num_mel_bins), static_cast<int>(chunk_mel_len)},
      ::executorch::aten::ScalarType::Float);

  auto conv1_state = from_blob(
      conv1_state_.data(),
      {1, static_cast<int>(num_mel_bins), static_cast<int>(runner_.conv1_pad_)},
      ::executorch::aten::ScalarType::Float);

  auto conv2_state = from_blob(
      conv2_state_.data(),
      {1,
       static_cast<int>(runner_.enc_dim_),
       static_cast<int>(runner_.conv2_pad_)},
      ::executorch::aten::ScalarType::Float);

  std::vector<int64_t> enc_pos_data(static_cast<size_t>(enc_frames_per_chunk));
  for (int64_t i = 0; i < enc_frames_per_chunk; i++) {
    enc_pos_data[static_cast<size_t>(i)] = enc_frame_pos_ + i;
  }
  auto enc_pos = from_blob(
      enc_pos_data.data(),
      {static_cast<int>(enc_frames_per_chunk)},
      ::executorch::aten::ScalarType::Long);

  // --- Run streaming encoder ---
  auto enc_result = runner_.model_->execute(
      "encode_audio_chunk",
      std::vector<EValue>{*mel_chunk, *conv1_state, *conv2_state, *enc_pos});
  ET_CHECK_MSG(enc_result.ok(), "encode_audio_chunk failed.");

  auto& enc_outputs = enc_result.get();
  auto audio_embeds = enc_outputs[0].toTensor();
  auto new_conv1 = enc_outputs[1].toTensor();
  auto new_conv2 = enc_outputs[2].toTensor();

  std::memcpy(
      conv1_state_.data(),
      new_conv1.const_data_ptr<float>(),
      conv1_state_.size() * sizeof(float));
  std::memcpy(
      conv2_state_.data(),
      new_conv2.const_data_ptr<float>(),
      conv2_state_.size() * sizeof(float));
  enc_frame_pos_ += enc_frames_per_chunk;
  samples_consumed_ += step;

  // --- Decode one step ---
  return decode_step(audio_embeds.const_data_ptr<float>());
}

bool StreamingSession::decode_step(const float* audio_embeds) {
  // Token embedding for previous token.
  int64_t token_id = static_cast<int64_t>(prev_token_);
  auto token_tensor =
      from_blob(&token_id, {1, 1}, ::executorch::aten::ScalarType::Long);

  auto tok_result = runner_.model_->execute(
      "token_embedding", std::vector<EValue>{*token_tensor});
  ET_CHECK_MSG(tok_result.ok(), "token_embedding failed.");
  auto tok_embed = tok_result.get()[0].toTensor();
  const float* tok_data = tok_embed.const_data_ptr<float>();

  // Sum audio + token embeddings (or token-only if audio_embeds is null).
  if (audio_embeds != nullptr) {
    for (int64_t i = 0; i < runner_.dim_; i++) {
      input_embeds_buf_[static_cast<size_t>(i)] = audio_embeds[i] + tok_data[i];
    }
  } else {
    std::memcpy(
        input_embeds_buf_.data(),
        tok_data,
        static_cast<size_t>(runner_.dim_) * sizeof(float));
  }

  auto input_embeds = from_blob(
      input_embeds_buf_.data(),
      {1, 1, static_cast<int>(runner_.dim_)},
      ::executorch::aten::ScalarType::Float);

  auto cache_pos =
      from_blob(&dec_pos_, {1}, ::executorch::aten::ScalarType::Long);

  auto dec_result = runner_.model_->execute(
      "text_decoder", std::vector<EValue>{*input_embeds, *cache_pos});
  ET_CHECK_MSG(dec_result.ok(), "text_decoder failed.");

  auto logits = dec_result.get()[0].toTensor();
  float* logits_data =
      logits.mutable_data_ptr<float>() + (logits.numel() - runner_.vocab_size_);
  int64_t next_token = static_cast<int64_t>(sampler_.sample(logits_data));
  num_generated_++;

  auto piece = runner_.tokenizer_->decode(
      prev_token_, static_cast<uint64_t>(next_token));
  if (piece.ok()) {
    token_cb_(*piece);
  }

  if (static_cast<uint64_t>(next_token) == runner_.eos_id_) {
    eos_reached_ = true;
    return true;
  }

  prev_token_ = static_cast<uint64_t>(next_token);
  dec_pos_++;
  return true;
}

int StreamingSession::flush() {
  if (flushed_) {
    return num_generated_;
  }
  flushed_ = true;

  // Pad with silence so any remaining audio (including partial steps and
  // the right look-ahead for the last complete step) can be processed.
  const int64_t remaining =
      static_cast<int64_t>(audio_buf_.size()) - samples_consumed_;
  if (remaining > 0 && !eos_reached_) {
    const int64_t step = runner_.step_samples_;
    const int64_t right_lookahead = runner_.stft_right_lookahead_;
    // Pad to next full step + right look-ahead
    int64_t pad_to = ((remaining + step - 1) / step) * step + right_lookahead;
    std::vector<float> silence(static_cast<size_t>(pad_to - remaining), 0.0f);
    audio_buf_.insert(audio_buf_.end(), silence.begin(), silence.end());

    while (!eos_reached_ && try_process_step()) {
    }
  }

  // Text-only decoding after audio ends.
  const int64_t max_text_steps = std::min(
      static_cast<int64_t>(config_.max_new_tokens) - num_generated_,
      runner_.max_seq_len_ - dec_pos_);

  for (int64_t i = 0; i < max_text_steps && !eos_reached_; i++) {
    decode_step(nullptr);
  }

  return num_generated_;
}

} // namespace voxtral_realtime
