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

} // namespace voxtral_realtime
