/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "voxtral_realtime_runner.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <vector>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

namespace voxtral_realtime {
namespace {

using ProfileClock = std::chrono::steady_clock;

double elapsed_milliseconds(ProfileClock::time_point start) {
  return std::chrono::duration<double, std::milli>(ProfileClock::now() - start)
      .count();
}

} // namespace

VoxtralRealtimeRunner::VoxtralRealtimeRunner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& preprocessor_path,
    const std::string& data_path,
    bool warmup)
    : VoxtralRealtimeRunner(
          model_path,
          tokenizer_path,
          preprocessor_path,
          data_path.empty() ? std::vector<std::string>{}
                            : std::vector<std::string>{data_path},
          warmup) {}

VoxtralRealtimeRunner::VoxtralRealtimeRunner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& preprocessor_path,
    const std::vector<std::string>& data_paths,
    bool warmup) {
  // Load the main model (.pte with audio_encoder, text_decoder,
  // token_embedding methods). Mmap avoids copying the file into memory.
  // Delegate data files hold external constants or compiled kernels.
  ET_LOG(Info, "Loading model from: %s", model_path.c_str());
  if (!data_paths.empty()) {
    for (const auto& data_path : data_paths) {
      std::ifstream data_file(data_path, std::ios::binary);
      ET_CHECK_MSG(
          data_file.good(),
          "Delegate data file not found or unreadable: %s",
          data_path.c_str());
      ET_LOG(Info, "Loading data from: %s", data_path.c_str());
    }
    model_ = std::make_unique<Module>(
        model_path, data_paths, Module::LoadMode::Mmap);
  } else {
    model_ = std::make_unique<Module>(model_path, Module::LoadMode::Mmap);
  }
  auto load_error = model_->load();
  ET_CHECK_MSG(
      load_error == Error::Ok,
      "Failed to load model or %zu delegate data file(s); verify every .ptd "
      "belongs to this .pte.",
      data_paths.size());

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

  // Detect model dtype from method metadata (same pattern as ASR runner).
  // Checks the first input tensor's scalar_type of the audio_encoder or
  // encode_audio_chunk method. Falls back to Float for old .pte files.
  for (const char* method : {"audio_encoder", "encode_audio_chunk"}) {
    auto meta_result = model_->method_meta(method);
    if (meta_result.ok()) {
      auto meta = meta_result.get();
      if (meta.num_inputs() > 0) {
        auto input_meta = meta.input_tensor_meta(0);
        if (input_meta.ok()) {
          model_dtype_ = input_meta.get().scalar_type();
        }
      }
      break;
    }
  }

  ET_LOG(
      Info,
      "Model: max_seq_len=%ld, vocab_size=%ld, dim=%ld, dtype=%s",
      static_cast<long>(max_seq_len_),
      static_cast<long>(vocab_size_),
      static_cast<long>(dim_),
      ::executorch::runtime::toString(model_dtype_));

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
    auto ebc = model_->execute("encoder_batch_chunks", empty);
    if (ebc.ok())
      encoder_batch_chunks_ = ebc.get()[0].toInt();
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

    auto dt = model_->execute("delay_tokens", empty);
    if (dt.ok())
      delay_tokens_ = dt.get()[0].toInt();

    auto sw = model_->execute("sliding_window", empty);
    if (sw.ok())
      sliding_window_ = sw.get()[0].toInt();

    ET_LOG(
        Info,
        "Streaming: chunk_mel=%ld, encoder_batch_chunks=%ld, max_enc=%ld, "
        "enc_dim=%ld",
        static_cast<long>(chunk_mel_len_),
        static_cast<long>(encoder_batch_chunks_),
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

  // Warmup: trigger lazy initialization (XNNPACK workspace allocation, etc.).
  // For streaming: run a single dummy step through the full pipeline.
  // For offline: call each method once with minimal inputs (can't use
  // transcribe() because the offline preprocessor pads to 30-second chunks).
  if (warmup && preprocessor_) {
    ET_LOG(Info, "Warming up...");
    std::vector<float> dummy_audio(static_cast<size_t>(step_samples_), 0.0f);

    if (is_streaming_) {
      StreamingTranscribeConfig warmup_config;
      auto session =
          create_streaming_session(warmup_config, [](const std::string&) {});
      session->feed_audio(dummy_audio.data(), step_samples_);
      session->flush();
    } else {
      // Preprocessor (always float32 — runs on CPU)
      auto pp_wav = from_blob(
          dummy_audio.data(),
          {static_cast<int>(step_samples_)},
          ::executorch::aten::ScalarType::Float);
      auto pp_r =
          preprocessor_->execute("forward", std::vector<EValue>{*pp_wav});
      ET_CHECK_MSG(pp_r.ok(), "Warmup: preprocessor failed.");

      // Audio encoder (8 mel frames = minimum valid input)
      // Create fp32 mel then convert to model dtype if needed.
      std::vector<float> dummy_mel_fp32(
          static_cast<size_t>(num_mel_bins_ * 8), 0.0f);
      auto mel_fp32 = from_blob(
          dummy_mel_fp32.data(),
          {1, static_cast<int>(num_mel_bins_), 8},
          ::executorch::aten::ScalarType::Float);
      auto mel_t = convert_to_model_dtype(std::move(mel_fp32));
      auto enc_r =
          model_->execute("audio_encoder", std::vector<EValue>{*mel_t});
      ET_CHECK_MSG(enc_r.ok(), "Warmup: audio_encoder failed.");

      // Token embedding
      int64_t dummy_tok = static_cast<int64_t>(bos_id_);
      auto tok_t =
          from_blob(&dummy_tok, {1, 1}, ::executorch::aten::ScalarType::Long);
      auto tok_r =
          model_->execute("token_embedding", std::vector<EValue>{*tok_t});
      ET_CHECK_MSG(tok_r.ok(), "Warmup: token_embedding failed.");

      // Text decoder — create embeds in model dtype
      auto tok_embed = tok_r.get()[0].toTensor();
      auto emb_t = from_blob(
          tok_embed.mutable_data_ptr(),
          {1, 1, static_cast<int>(dim_)},
          model_dtype_);
      int64_t dummy_pos = 0;
      auto pos_t =
          from_blob(&dummy_pos, {1}, ::executorch::aten::ScalarType::Long);
      auto dec_r =
          model_->execute("text_decoder", std::vector<EValue>{*emb_t, *pos_t});
      ET_CHECK_MSG(dec_r.ok(), "Warmup: text_decoder failed.");
    }
    ET_LOG(Info, "Warmup complete.");
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

// Extract the last vocab_size logits as fp32 for sampling.
// If the tensor is already fp32, returns a pointer into it directly.
// Otherwise converts to fp32 into the provided buffer.
static float* get_logits_fp32(
    ::executorch::aten::Tensor& logits,
    int64_t vocab_size,
    std::vector<float>& buf) {
  const int64_t offset = logits.numel() - vocab_size;
  if (logits.scalar_type() == ::executorch::aten::ScalarType::Float) {
    return logits.mutable_data_ptr<float>() + offset;
  }
  // Convert bf16/half logits to fp32
  buf.resize(static_cast<size_t>(vocab_size));
  if (logits.scalar_type() == ::executorch::aten::ScalarType::BFloat16) {
    const auto* src =
        logits.const_data_ptr<::executorch::aten::BFloat16>() + offset;
    for (int64_t i = 0; i < vocab_size; i++) {
      buf[static_cast<size_t>(i)] = static_cast<float>(src[i]);
    }
  } else if (logits.scalar_type() == ::executorch::aten::ScalarType::Half) {
    const auto* src =
        logits.const_data_ptr<::executorch::aten::Half>() + offset;
    for (int64_t i = 0; i < vocab_size; i++) {
      buf[static_cast<size_t>(i)] = static_cast<float>(src[i]);
    }
  } else {
    ET_CHECK_MSG(false, "Unsupported logits dtype for sampling.");
  }
  return buf.data();
}

TensorPtr VoxtralRealtimeRunner::convert_to_model_dtype(TensorPtr tensor) {
  if (model_dtype_ == ::executorch::aten::ScalarType::Float ||
      tensor->scalar_type() == model_dtype_) {
    return tensor;
  }
  if (model_dtype_ == ::executorch::aten::ScalarType::BFloat16) {
    auto result = ::executorch::extension::llm::convert_to_bfloat16(tensor);
    ET_CHECK_MSG(result.ok(), "Failed to convert tensor to BFloat16.");
    return std::move(result.get());
  }
  ET_CHECK_MSG(false, "Unsupported model dtype conversion.");
  return tensor; // unreachable
}

int VoxtralRealtimeRunner::transcribe(
    const float* audio_data,
    int64_t num_samples,
    const OfflineTranscribeConfig& config,
    TokenCallback token_cb) {
  // --- Step 1: Preprocess raw audio to mel spectrogram ---
  ET_CHECK_MSG(preprocessor_ != nullptr, "No preprocessor provided.");
  const auto preprocessor_start = ProfileClock::now();
  TensorPtr mel_fp32 = run_preprocessor(audio_data, num_samples);
  const double preprocessor_ms = elapsed_milliseconds(preprocessor_start);

  // Convert mel from fp32 (preprocessor) to model dtype (may be bf16)
  TensorPtr mel = convert_to_model_dtype(std::move(mel_fp32));

  // --- Step 2: Encode mel to audio embeddings ---
  // audio_encoder: (1, 128, T_mel) -> (1, T_audio, 3072)
  // T_audio = T_mel / 8 (conv stride 2, then downsample by 4).
  const auto encoder_start = ProfileClock::now();
  auto enc_result = model_->execute("audio_encoder", std::vector<EValue>{*mel});
  const double encoder_ms = elapsed_milliseconds(encoder_start);
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

  ::executorch::extension::llm::Sampler sampler(
      static_cast<int32_t>(vocab_size_),
      config.temperature,
      ::executorch::extension::llm::kTopp,
      static_cast<unsigned long long>(std::time(nullptr)));
  std::vector<float> logits_fp32_buf;
  auto input_embeds = ::executorch::extension::empty(
      {1, 1, static_cast<int>(dim_)}, model_dtype_);
  double embedding_ms = 0.0;
  double add_ms = 0.0;
  double decoder_ms = 0.0;
  double sampling_ms = 0.0;

  for (int64_t pos = 0; pos < max_pos; pos++) {
    // a. Look up embedding for the previous token.
    int64_t token_id = static_cast<int64_t>(prev_token);
    auto token_tensor =
        from_blob(&token_id, {1, 1}, ::executorch::aten::ScalarType::Long);

    const auto embedding_start = ProfileClock::now();
    auto tok_result =
        model_->execute("token_embedding", std::vector<EValue>{*token_tensor});
    if (config.profile_methods) {
      embedding_ms += elapsed_milliseconds(embedding_start);
    }
    ET_CHECK_MSG(tok_result.ok(), "token_embedding failed.");
    auto tok_embed = tok_result.get()[0].toTensor();

    // b. Sum audio + token embeddings (or token-only after audio ends).
    // Both audio_embeds and tok_embed are in model_dtype_ (fp32 or bf16).
    // Reuses pre-allocated input_embeds buffer (no per-token allocation).
    const auto add_start = ProfileClock::now();
    if (pos < t_audio) {
      // Element-wise sum: audio_frame[i] + tok_data[i]
      // Works for any dtype since we operate on raw bytes via BFloat16 type.
      if (model_dtype_ == ::executorch::aten::ScalarType::BFloat16) {
        auto* out =
            input_embeds->mutable_data_ptr<::executorch::aten::BFloat16>();
        const auto* af =
            audio_embeds.const_data_ptr<::executorch::aten::BFloat16>() +
            pos * dim_;
        const auto* tf =
            tok_embed.const_data_ptr<::executorch::aten::BFloat16>();
        for (int64_t i = 0; i < dim_; i++) {
          out[i] = ::executorch::aten::BFloat16(
              static_cast<float>(af[i]) + static_cast<float>(tf[i]));
        }
      } else {
        auto* out = input_embeds->mutable_data_ptr<float>();
        const auto* af = audio_embeds.const_data_ptr<float>() + pos * dim_;
        const auto* tf = tok_embed.const_data_ptr<float>();
        for (int64_t i = 0; i < dim_; i++) {
          out[i] = af[i] + tf[i];
        }
      }
    } else {
      std::memcpy(
          input_embeds->mutable_data_ptr(),
          tok_embed.const_data_ptr(),
          static_cast<size_t>(dim_) * input_embeds->element_size());
    }
    if (config.profile_methods) {
      add_ms += elapsed_milliseconds(add_start);
    }

    // c. Run one decoder step. KV cache is updated internally by the model.
    auto cache_pos = from_blob(&pos, {1}, ::executorch::aten::ScalarType::Long);

    const auto decoder_start = ProfileClock::now();
    auto dec_result = model_->execute(
        "text_decoder", std::vector<EValue>{*input_embeds, *cache_pos});
    if (config.profile_methods) {
      decoder_ms += elapsed_milliseconds(decoder_start);
    }
    ET_CHECK_MSG(dec_result.ok(), "text_decoder failed.");

    auto logits = dec_result.get()[0].toTensor();

    // d. Sample next token (persistent sampler preserves RNG state).
    const auto sampling_start = ProfileClock::now();
    float* logits_data = get_logits_fp32(logits, vocab_size_, logits_fp32_buf);
    int64_t next_token = static_cast<int64_t>(sampler.sample(logits_data));
    num_generated++;

    // e. Decode token to text and emit via callback.
    auto piece =
        tokenizer_->decode(prev_token, static_cast<uint64_t>(next_token));
    if (piece.ok()) {
      token_cb(*piece);
    }
    if (config.profile_methods) {
      sampling_ms += elapsed_milliseconds(sampling_start);
    }

    // f. Stop on end-of-sequence.
    if (static_cast<uint64_t>(next_token) == eos_id_) {
      break;
    }

    prev_token = static_cast<uint64_t>(next_token);
  }

  if (config.profile_methods) {
    ET_LOG(
        Info,
        "VOXTRAL_PROFILE mode=offline steps=%d preprocessor_ms=%.3f "
        "encoder_ms=%.3f embedding_ms=%.3f add_ms=%.3f decoder_ms=%.3f "
        "sampling_ms=%.3f",
        num_generated,
        preprocessor_ms,
        encoder_ms,
        embedding_ms,
        add_ms,
        decoder_ms,
        sampling_ms);
  }

  return num_generated;
}

// ---------------------------------------------------------------------------
// StreamingSession
// ---------------------------------------------------------------------------

std::unique_ptr<StreamingSession>
VoxtralRealtimeRunner::create_streaming_session(
    const StreamingTranscribeConfig& config,
    TokenCallback token_cb) {
  ET_CHECK_MSG(is_streaming_, "Model was not exported with --streaming.");
  ET_CHECK_MSG(
      preprocessor_ != nullptr,
      "No preprocessor loaded. Provide --preprocessor_path.");
  return std::make_unique<StreamingSession>(*this, config, std::move(token_cb));
}

StreamingSession::StreamingSession(
    VoxtralRealtimeRunner& runner,
    StreamingTranscribeConfig config,
    TokenCallback token_cb)
    : runner_(runner),
      token_cb_(std::move(token_cb)),
      prev_token_(runner.bos_id_),
      profile_methods_(config.profile_methods),
      sampler_(
          static_cast<int32_t>(runner.vocab_size_),
          config.temperature,
          ::executorch::extension::llm::kTopp,
          static_cast<unsigned long long>(std::time(nullptr))),
      audio_embeds_bf16_copy_(
          runner.model_dtype_ == ::executorch::aten::ScalarType::BFloat16
              ? static_cast<size_t>(runner.encoder_batch_chunks_ * runner.dim_)
              : 0),
      audio_embeds_fp32_copy_(
          runner.model_dtype_ != ::executorch::aten::ScalarType::BFloat16
              ? static_cast<size_t>(runner.encoder_batch_chunks_ * runner.dim_)
              : 0),
      input_embeds_(
          ::executorch::extension::empty(
              {1, 1, static_cast<int>(runner.dim_)},
              runner.model_dtype_)) {}

int StreamingSession::feed_audio(const float* data, int64_t num_samples) {
  audio_buf_.insert(audio_buf_.end(), data, data + num_samples);

  const int generated_before = num_generated_;
  while (!eos_reached_ && try_process_step()) {
    // num_generated_ is updated inside try_process_step()
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

  return num_generated_ - generated_before;
}

bool StreamingSession::try_process_step() {
  const auto step_start = ProfileClock::now();
  auto framing_start = step_start;
  const int64_t step = runner_.step_samples_;
  const int64_t left_overlap = runner_.stft_left_overlap_;
  const int64_t right_lookahead = runner_.stft_right_lookahead_;
  const int64_t mel_skip = runner_.mel_skip_frames_;
  const int64_t chunk_mel_len = runner_.chunk_mel_len_;
  const int64_t encoder_batch_chunks = runner_.encoder_batch_chunks_;
  const int64_t encoder_chunk_mel_len = chunk_mel_len * encoder_batch_chunks;

  // Need enough audio for: current step + right look-ahead.
  // Left overlap comes from audio before samples_consumed_ (already in buffer).
  const int64_t batched_step = step * encoder_batch_chunks;
  const int64_t need_end = samples_consumed_ + batched_step + right_lookahead;
  if (static_cast<int64_t>(audio_buf_.size()) < need_end) {
    return false;
  }

  // Old .pte files use a flat decoder KV cache bounded by max_seq_len.
  // New .pte files (with sliding_window metadata) use a ring buffer with
  // no position limit.
  const int64_t enc_frames_per_chunk = encoder_chunk_mel_len / 2;
  if (runner_.sliding_window_ == 0 && dec_pos_ >= runner_.max_seq_len_) {
    return false;
  }

  const int64_t num_mel_bins = runner_.num_mel_bins_;
  std::vector<float> mel_chunk_fp32(
      static_cast<size_t>(num_mel_bins * encoder_chunk_mel_len));
  const int64_t window_size = left_overlap + step + right_lookahead;
  for (int64_t batch_idx = 0; batch_idx < encoder_batch_chunks; ++batch_idx) {
    const auto chunk_framing_start = ProfileClock::now();
    const int64_t chunk_start = samples_consumed_ + batch_idx * step;
    std::vector<float> window_buf(static_cast<size_t>(window_size), 0.0f);
    const int64_t left_start = chunk_start - left_overlap;
    if (left_start >= 0) {
      std::memcpy(
          window_buf.data(),
          audio_buf_.data() + left_start,
          static_cast<size_t>(left_overlap) * sizeof(float));
    } else {
      const int64_t available_left = chunk_start;
      const int64_t zero_pad = left_overlap - available_left;
      if (available_left > 0) {
        std::memcpy(
            window_buf.data() + zero_pad,
            audio_buf_.data(),
            static_cast<size_t>(available_left) * sizeof(float));
      }
    }
    std::memcpy(
        window_buf.data() + left_overlap,
        audio_buf_.data() + chunk_start,
        static_cast<size_t>(step + right_lookahead) * sizeof(float));
    auto audio_tensor = from_blob(
        window_buf.data(),
        {static_cast<int>(window_size)},
        ::executorch::aten::ScalarType::Float);
    if (profile_methods_) {
      profile_framing_ms_ += elapsed_milliseconds(chunk_framing_start);
    }

    const auto preprocessor_start = ProfileClock::now();
    auto mel_result = runner_.preprocessor_->execute(
        "forward", std::vector<EValue>{*audio_tensor});
    if (profile_methods_) {
      profile_preprocessor_ms_ += elapsed_milliseconds(preprocessor_start);
    }
    ET_CHECK_MSG(mel_result.ok(), "Streaming preprocessor failed.");

    const auto copy_start = ProfileClock::now();
    auto mel = mel_result.get()[0].toTensor();
    ET_CHECK_MSG(
        mel.size(1) == num_mel_bins && mel.size(2) >= mel_skip + chunk_mel_len,
        "Preprocessor produced an unexpected mel shape.");
    const int64_t total_mel_frames = mel.size(2);
    const float* mel_data = mel.const_data_ptr<float>();
    for (int64_t c = 0; c < num_mel_bins; ++c) {
      std::memcpy(
          mel_chunk_fp32.data() + c * encoder_chunk_mel_len +
              batch_idx * chunk_mel_len,
          mel_data + c * total_mel_frames + mel_skip,
          static_cast<size_t>(chunk_mel_len) * sizeof(float));
    }
    if (profile_methods_) {
      profile_framing_ms_ += elapsed_milliseconds(copy_start);
    }
  }

  framing_start = ProfileClock::now();
  auto mel_chunk_tensor = from_blob(
      mel_chunk_fp32.data(),
      {1,
       static_cast<int>(num_mel_bins),
       static_cast<int>(encoder_chunk_mel_len)},
      ::executorch::aten::ScalarType::Float);

  // Convert to model dtype if needed (e.g., fp32 -> bf16 for CUDA)
  auto mel_chunk = runner_.convert_to_model_dtype(std::move(mel_chunk_tensor));

  std::vector<int64_t> enc_pos_data(static_cast<size_t>(enc_frames_per_chunk));
  for (int64_t i = 0; i < enc_frames_per_chunk; i++) {
    enc_pos_data[static_cast<size_t>(i)] = enc_frame_pos_ + i;
  }
  auto enc_pos = from_blob(
      enc_pos_data.data(),
      {static_cast<int>(enc_frames_per_chunk)},
      ::executorch::aten::ScalarType::Long);
  if (profile_methods_) {
    profile_framing_ms_ += elapsed_milliseconds(framing_start);
  }

  // --- Run streaming encoder ---
  const auto encoder_start = ProfileClock::now();
  auto enc_result = runner_.model_->execute(
      "encode_audio_chunk", std::vector<EValue>{*mel_chunk, *enc_pos});
  if (profile_methods_) {
    profile_encoder_ms_ += elapsed_milliseconds(encoder_start);
  }
  ET_CHECK_MSG(enc_result.ok(), "encode_audio_chunk failed.");

  auto audio_embeds = enc_result.get()[0].toTensor();
  ET_CHECK_MSG(
      audio_embeds.dim() == 3 && audio_embeds.size(1) == encoder_batch_chunks &&
          audio_embeds.size(2) == runner_.dim_,
      "Unexpected batched encoder output shape.");

  enc_frame_pos_ += enc_frames_per_chunk;
  samples_consumed_ += batched_step;

  int64_t decode_count = encoder_batch_chunks;
  if (pending_flush_decode_steps_ >= 0) {
    decode_count = std::min(decode_count, pending_flush_decode_steps_);
  }
  int64_t decoded = 0;
  // Decoder execution may reuse module-managed output storage, so preserve the
  // batched encoder result across all decoder calls in this step.
  if (runner_.model_dtype_ == ::executorch::aten::ScalarType::BFloat16) {
    auto& copied_audio = audio_embeds_bf16_copy_;
    std::memcpy(
        copied_audio.data(),
        audio_embeds.const_data_ptr<::executorch::aten::BFloat16>(),
        copied_audio.size() * sizeof(::executorch::aten::BFloat16));
    for (; decoded < decode_count && !eos_reached_; ++decoded) {
      auto audio_embed = from_blob(
          copied_audio.data() + decoded * runner_.dim_,
          {1, 1, static_cast<int>(runner_.dim_)},
          ::executorch::aten::ScalarType::BFloat16);
      decode_step(audio_embed);
    }
  } else {
    auto& copied_audio = audio_embeds_fp32_copy_;
    std::memcpy(
        copied_audio.data(),
        audio_embeds.const_data_ptr<float>(),
        copied_audio.size() * sizeof(float));
    for (; decoded < decode_count && !eos_reached_; ++decoded) {
      auto audio_embed = from_blob(
          copied_audio.data() + decoded * runner_.dim_,
          {1, 1, static_cast<int>(runner_.dim_)},
          ::executorch::aten::ScalarType::Float);
      decode_step(audio_embed);
    }
  }
  if (pending_flush_decode_steps_ >= 0) {
    pending_flush_decode_steps_ -= decoded;
  }
  if (profile_methods_) {
    profile_total_ms_ += elapsed_milliseconds(step_start);
    profile_steps_ += decoded;
  }
  return true;
}

bool StreamingSession::decode_step(const TensorPtr& audio_embeds_tensor) {
  const int64_t dim = runner_.dim_;
  const auto model_dtype = runner_.model_dtype_;

  // Token embedding for previous token.
  int64_t token_id = static_cast<int64_t>(prev_token_);
  auto token_tensor =
      from_blob(&token_id, {1, 1}, ::executorch::aten::ScalarType::Long);

  const auto embedding_start = ProfileClock::now();
  auto tok_result = runner_.model_->execute(
      "token_embedding", std::vector<EValue>{*token_tensor});
  if (profile_methods_) {
    profile_embedding_ms_ += elapsed_milliseconds(embedding_start);
  }
  ET_CHECK_MSG(tok_result.ok(), "token_embedding failed.");
  auto tok_embed = tok_result.get()[0].toTensor();

  // Sum audio + token embeddings.
  // Reuses pre-allocated input_embeds_ buffer (no per-token allocation).
  const auto add_start = ProfileClock::now();
  auto& audio_embeds = *audio_embeds_tensor;
  if (model_dtype == ::executorch::aten::ScalarType::BFloat16) {
    auto* out = input_embeds_->mutable_data_ptr<::executorch::aten::BFloat16>();
    const auto* af =
        audio_embeds.const_data_ptr<::executorch::aten::BFloat16>();
    const auto* tf = tok_embed.const_data_ptr<::executorch::aten::BFloat16>();
    for (int64_t i = 0; i < dim; i++) {
      out[i] = ::executorch::aten::BFloat16(
          static_cast<float>(af[i]) + static_cast<float>(tf[i]));
    }
  } else {
    auto* out = input_embeds_->mutable_data_ptr<float>();
    const auto* af = audio_embeds.const_data_ptr<float>();
    const auto* tf = tok_embed.const_data_ptr<float>();
    for (int64_t i = 0; i < dim; i++) {
      out[i] = af[i] + tf[i];
    }
  }
  if (profile_methods_) {
    profile_add_ms_ += elapsed_milliseconds(add_start);
  }

  auto cache_pos =
      from_blob(&dec_pos_, {1}, ::executorch::aten::ScalarType::Long);

  const auto decoder_start = ProfileClock::now();
  auto dec_result = runner_.model_->execute(
      "text_decoder", std::vector<EValue>{*input_embeds_, *cache_pos});
  if (profile_methods_) {
    profile_decoder_ms_ += elapsed_milliseconds(decoder_start);
  }
  ET_CHECK_MSG(dec_result.ok(), "text_decoder failed.");

  auto logits = dec_result.get()[0].toTensor();

  // Sample next token (persistent sampler preserves RNG state).
  const auto sampling_start = ProfileClock::now();
  float* logits_data =
      get_logits_fp32(logits, runner_.vocab_size_, logits_fp32_buf_);
  int64_t next_token = static_cast<int64_t>(sampler_.sample(logits_data));
  num_generated_++;

  auto piece = runner_.tokenizer_->decode(
      prev_token_, static_cast<uint64_t>(next_token));
  if (piece.ok()) {
    token_cb_(*piece);
  }
  if (profile_methods_) {
    profile_sampling_ms_ += elapsed_milliseconds(sampling_start);
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

  const int64_t remaining =
      static_cast<int64_t>(audio_buf_.size()) - samples_consumed_;
  if (remaining > 0 && !eos_reached_) {
    const int64_t step = runner_.step_samples_;
    const int64_t right_lookahead = runner_.stft_right_lookahead_;
    const int64_t right_pad_audio_steps = runner_.delay_tokens_;

    // Stay on the normal audio-conditioned path through the final partial
    // step, the preprocessor look-ahead, and the model's transcription delay.
    // Matches vLLM flush behavior:
    // https://github.com/vllm-project/vllm/blob/2f9f946/vllm/model_executor/models/voxtral_realtime.py#L239-L270
    const int64_t remaining_audio_steps = (remaining + step - 1) / step;
    const int64_t remaining_decode_steps =
        remaining_audio_steps + right_pad_audio_steps;
    const int64_t padded_audio_steps =
        ((remaining_decode_steps + runner_.encoder_batch_chunks_ - 1) /
         runner_.encoder_batch_chunks_) *
        runner_.encoder_batch_chunks_;
    const int64_t pad_to = padded_audio_steps * step + right_lookahead;
    const int64_t silence_padded_samples = pad_to - remaining;
    std::vector<float> silence(
        static_cast<size_t>(silence_padded_samples), 0.0f);
    audio_buf_.insert(audio_buf_.end(), silence.begin(), silence.end());

    // Guaranteed to terminate b/c each call to try_process_step() consumes a
    // fixed number of audio samples and the padded audio buffer is finite.
    pending_flush_decode_steps_ = remaining_decode_steps;
    while (!eos_reached_ && pending_flush_decode_steps_ > 0 &&
           try_process_step()) {
    }
    pending_flush_decode_steps_ = -1;
  }

  print_profile();

  return num_generated_;
}

void StreamingSession::print_profile() const {
  if (!profile_methods_) {
    return;
  }
  const double accounted_ms = profile_framing_ms_ + profile_preprocessor_ms_ +
      profile_encoder_ms_ + profile_embedding_ms_ + profile_add_ms_ +
      profile_decoder_ms_ + profile_sampling_ms_;
  ET_LOG(
      Info,
      "VOXTRAL_PROFILE mode=streaming steps=%ld total_ms=%.3f "
      "framing_ms=%.3f preprocessor_ms=%.3f encoder_ms=%.3f "
      "embedding_ms=%.3f add_ms=%.3f decoder_ms=%.3f sampling_ms=%.3f "
      "other_ms=%.3f",
      static_cast<long>(profile_steps_),
      profile_total_ms_,
      profile_framing_ms_,
      profile_preprocessor_ms_,
      profile_encoder_ms_,
      profile_embedding_ms_,
      profile_add_ms_,
      profile_decoder_ms_,
      profile_sampling_ms_,
      profile_total_ms_ - accounted_ms);
}

} // namespace voxtral_realtime
