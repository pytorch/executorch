/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "voxtral_tts_runner.h"

#include <cstring>
#include <ctime>
#include <fstream>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>

using ::executorch::extension::empty;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

namespace voxtral_tts {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

VoxtralTTSRunner::VoxtralTTSRunner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& codec_path,
    bool warmup) {
  // Load model.pte
  ET_LOG(Info, "Loading model from: %s", model_path.c_str());
  model_ = std::make_unique<Module>(model_path, Module::LoadMode::Mmap);
  auto load_err = model_->load();
  ET_CHECK_MSG(load_err == Error::Ok, "Failed to load model.");

  // Read metadata constants
  std::vector<EValue> empty_args;
  auto ms = model_->execute("max_seq_len", empty_args);
  auto vs = model_->execute("vocab_size", empty_args);
  auto dm = model_->execute("dim", empty_args);
  auto nac = model_->execute("n_acoustic_codebook", empty_args);

  if (ms.ok())
    max_seq_len_ = ms.get()[0].toInt();
  if (vs.ok())
    vocab_size_ = vs.get()[0].toInt();
  if (dm.ok())
    dim_ = dm.get()[0].toInt();
  if (nac.ok())
    n_acoustic_codebook_ = nac.get()[0].toInt();
  n_codebooks_ = 1 + n_acoustic_codebook_;

  // Detect model dtype from text_decoder input
  auto meta_result = model_->method_meta("text_decoder");
  if (meta_result.ok()) {
    auto meta = meta_result.get();
    if (meta.num_inputs() > 0) {
      auto input_meta = meta.input_tensor_meta(0);
      if (input_meta.ok()) {
        model_dtype_ = input_meta.get().scalar_type();
      }
    }
  }

  ET_LOG(
      Info,
      "Model: max_seq_len=%ld, vocab=%ld, dim=%ld, n_codebooks=%ld, dtype=%s",
      static_cast<long>(max_seq_len_),
      static_cast<long>(vocab_size_),
      static_cast<long>(dim_),
      static_cast<long>(n_codebooks_),
      ::executorch::runtime::toString(model_dtype_));

  // Load codec.pte (optional)
  if (!codec_path.empty()) {
    ET_LOG(Info, "Loading codec from: %s", codec_path.c_str());
    codec_ = std::make_unique<Module>(codec_path, Module::LoadMode::Mmap);
    auto codec_err = codec_->load();
    ET_CHECK_MSG(codec_err == Error::Ok, "Failed to load codec.");

    auto sr = codec_->execute("sampling_rate", empty_args);
    auto cs = codec_->execute("chunk_size", empty_args);
    auto nc = codec_->execute("n_codebooks", empty_args);
    if (sr.ok())
      sample_rate_ = sr.get()[0].toInt();
    if (cs.ok())
      codec_chunk_size_ = cs.get()[0].toInt();
    if (nc.ok())
      n_codebooks_ = nc.get()[0].toInt();
    ET_LOG(
        Info,
        "Codec: sample_rate=%ld, chunk_size=%ld",
        static_cast<long>(sample_rate_),
        static_cast<long>(codec_chunk_size_));
  }

  // Load tokenizer
  tokenizer_ = ::executorch::extension::llm::load_tokenizer(tokenizer_path);
  bos_id_ = tokenizer_->bos_tok();
  eos_id_ = tokenizer_->eos_tok();
  ET_LOG(Info, "Tokenizer: bos=%lu, eos=%lu", bos_id_, eos_id_);

  // Audio token ID from metadata, or default to 24 (Tekken special_ids.audio).
  auto at = model_->execute("audio_tok_id", empty_args);
  if (at.ok()) {
    audio_tok_id_ = static_cast<uint64_t>(at.get()[0].toInt());
  } else {
    audio_tok_id_ = 24; // Mistral Tekken default
  }
  ET_LOG(Info, "Audio token ID: %lu", audio_tok_id_);

  // Warmup: call each method once with minimal inputs
  if (warmup) {
    ET_LOG(Info, "Warming up...");
    // token_embedding
    {
      int64_t tok = 0;
      auto t = from_blob(&tok, {1, 1}, ::executorch::aten::ScalarType::Long);
      model_->execute("token_embedding", std::vector<EValue>{*t});
    }
    // text_decoder
    {
      auto e = empty({1, 4, static_cast<int>(dim_)}, model_dtype_);
      std::vector<int64_t> pos_data = {0, 1, 2, 3};
      auto p =
          from_blob(pos_data.data(), {4}, ::executorch::aten::ScalarType::Long);
      model_->execute("text_decoder", std::vector<EValue>{*e, *p});
    }
    // lm_head
    {
      auto h = empty({1, 1, static_cast<int>(dim_)}, model_dtype_);
      model_->execute("lm_head", std::vector<EValue>{*h});
    }
    // decode_audio_frame
    {
      auto h = empty({1, static_cast<int>(dim_)}, model_dtype_);
      auto n = empty({1, static_cast<int>(n_acoustic_codebook_)}, model_dtype_);
      model_->execute("decode_audio_frame", std::vector<EValue>{*h, *n});
    }
    // audio_token_embedding
    {
      auto c = empty(
          {1, static_cast<int>(n_codebooks_), 1},
          ::executorch::aten::ScalarType::Long);
      model_->execute("audio_token_embedding", std::vector<EValue>{*c});
    }
    ET_LOG(Info, "Warmup complete.");
  }
}

// ---------------------------------------------------------------------------
// Voice embedding
// ---------------------------------------------------------------------------

void VoxtralTTSRunner::load_voice(const std::string& voice_bin_path) {
  std::ifstream f(voice_bin_path, std::ios::binary);
  ET_CHECK_MSG(
      f.is_open(), "Cannot open voice file: %s", voice_bin_path.c_str());

  int64_t n_frames, dim;
  f.read(reinterpret_cast<char*>(&n_frames), sizeof(n_frames));
  f.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  ET_CHECK_MSG(
      dim == dim_,
      "Voice dim %ld != model dim %ld",
      static_cast<long>(dim),
      static_cast<long>(dim_));

  voice_data_.resize(static_cast<size_t>(n_frames * dim));
  f.read(
      reinterpret_cast<char*>(voice_data_.data()),
      static_cast<std::streamsize>(n_frames * dim * sizeof(float)));
  n_voice_frames_ = n_frames;
  ET_LOG(
      Info,
      "Loaded voice: %ld frames x %ld dim from %s",
      static_cast<long>(n_frames),
      static_cast<long>(dim),
      voice_bin_path.c_str());
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

std::vector<float> VoxtralTTSRunner::generate(
    const std::string& prompt,
    const GenerateConfig& config,
    TokenCallback token_cb,
    std::vector<int64_t>* audio_codes_out) {
  // Encode prompt
  auto encode_result = tokenizer_->encode(prompt);
  ET_CHECK_MSG(encode_result.ok(), "Failed to encode prompt.");
  auto prompt_tokens = encode_result.get();

  // Build TTS prompt matching mistral_common's encode_speech_request format:
  // [BOS] [BEGIN_AUDIO] [AUDIO]×n_voice [/INST] text [INST] [BEGIN_AUDIO]
  // Tekken special token IDs: INST=35, /INST=36, BEGIN_AUDIO=25, AUDIO=24
  constexpr int64_t INST_TOK = 35;
  constexpr int64_t END_INST_TOK = 36;
  constexpr int64_t BEGIN_AUDIO_TOK = 25;

  // Voice context: use loaded voice embedding frames, or 10 dummy tokens
  const int64_t n_voice = n_voice_frames_ > 0 ? n_voice_frames_ : 10;

  std::vector<int64_t> tokens;
  tokens.push_back(static_cast<int64_t>(bos_id_));
  tokens.push_back(BEGIN_AUDIO_TOK);
  const int64_t voice_start = static_cast<int64_t>(tokens.size());
  for (int64_t i = 0; i < n_voice; i++) {
    tokens.push_back(static_cast<int64_t>(audio_tok_id_));
  }
  const int64_t voice_end = static_cast<int64_t>(tokens.size());
  tokens.push_back(END_INST_TOK);
  for (auto t : prompt_tokens) {
    tokens.push_back(static_cast<int64_t>(t));
  }
  tokens.push_back(INST_TOK);
  tokens.push_back(BEGIN_AUDIO_TOK);
  const int64_t prompt_len = static_cast<int64_t>(tokens.size());
  ET_LOG(
      Info,
      "Prompt: %ld tokens (including BOS)",
      static_cast<long>(prompt_len));

  // Setup sampler and noise RNG
  ::executorch::extension::llm::Sampler sampler(
      static_cast<int32_t>(vocab_size_),
      config.temperature,
      ::executorch::extension::llm::kTopp,
      static_cast<unsigned long long>(config.seed));
  std::mt19937 noise_rng(static_cast<unsigned int>(config.seed));
  std::normal_distribution<float> noise_dist(0.0f, 1.0f);

  // Accumulated audio codes: flat buffer, n_codebooks values per frame
  std::vector<int64_t> all_codes;
  int64_t num_audio_frames = 0;

  int64_t dec_pos = 0;
  int num_generated = 0;

  // --- Prefill: process all prompt tokens ---
  ET_LOG(Info, "Prefilling %ld tokens...", static_cast<long>(prompt_len));
  {
    auto ids_tensor = from_blob(
        tokens.data(),
        {1, static_cast<int>(prompt_len)},
        ::executorch::aten::ScalarType::Long);
    auto tok_result =
        model_->execute("token_embedding", std::vector<EValue>{*ids_tensor});
    ET_CHECK_MSG(tok_result.ok(), "token_embedding (prefill) failed.");
    auto embeds = tok_result.get()[0].toTensor();

    // Inject voice embeddings at audio token positions [voice_start, voice_end)
    if (n_voice_frames_ > 0 && !voice_data_.empty()) {
      const size_t elem_size = embeds.element_size();
      for (int64_t i = voice_start; i < voice_end; i++) {
        const int64_t voice_idx = i - voice_start;
        const float* src = voice_data_.data() + voice_idx * dim_;
        char* dst = static_cast<char*>(embeds.mutable_data_ptr()) +
            static_cast<size_t>(i * dim_) * elem_size;
        if (model_dtype_ == ::executorch::aten::ScalarType::BFloat16) {
          auto* dst_bf16 = reinterpret_cast<::executorch::aten::BFloat16*>(dst);
          for (int64_t d = 0; d < dim_; d++) {
            dst_bf16[d] = ::executorch::aten::BFloat16(src[d]);
          }
        } else {
          std::memcpy(dst, src, static_cast<size_t>(dim_) * sizeof(float));
        }
      }
      ET_LOG(
          Info,
          "Injected %ld voice frames at positions [%ld, %ld)",
          static_cast<long>(n_voice_frames_),
          static_cast<long>(voice_start),
          static_cast<long>(voice_end));
    }

    // Position tensor [0, 1, ..., prompt_len-1]
    std::vector<int64_t> pos_vec(static_cast<size_t>(prompt_len));
    for (int64_t i = 0; i < prompt_len; i++)
      pos_vec[static_cast<size_t>(i)] = i;
    auto pos_tensor = from_blob(
        pos_vec.data(),
        {static_cast<int>(prompt_len)},
        ::executorch::aten::ScalarType::Long);

    auto dec_result = model_->execute(
        "text_decoder", std::vector<EValue>{embeds, *pos_tensor});
    ET_CHECK_MSG(dec_result.ok(), "text_decoder (prefill) failed.");

    dec_pos = prompt_len;
    ET_LOG(Info, "Prefill done.");
  }

  // --- Audio generation loop ---
  // After the prompt ends with [BEGIN_AUDIO], every step:
  // 1. Extract hidden state from text_decoder
  // 2. Run decode_audio_frame → audio codes + check semantic code for EOS
  // 3. Feed audio token (24) back to text_decoder for next step
  // The LM head is NOT used — the acoustic transformer drives generation.
  // END_AUDIO (semantic code == 1) signals end of audio.
  constexpr int64_t END_AUDIO_SEMANTIC = 1;

  ET_LOG(Info, "Generating audio (max %d frames)...", config.max_tokens);

  // Pad to 3 tokens for Metal SDPA compatibility (requires seq_len >= 3).
  // Positions [dec_pos, dec_pos+1, dec_pos+2] — we advance by 1 per step
  // but present 3 positions to satisfy Metal's min seq_len. Only position 0's
  // hidden state is used. The padding positions get overwritten next step.
  constexpr int DECODE_SEQ_LEN = 3;

  // First step uses plain token_embedding(audio_tok) since there are no
  // previous audio codes yet. Subsequent steps use audio_token_embedding
  // on the previous step's codes (multi-codebook sum), matching vLLM's
  // tts_preprocess which calls embed_multimodal(audio_tokens).
  int64_t audio_tok = static_cast<int64_t>(audio_tok_id_);
  auto audio_tok_t =
      from_blob(&audio_tok, {1, 1}, ::executorch::aten::ScalarType::Long);
  auto audio_tok_result =
      model_->execute("token_embedding", std::vector<EValue>{*audio_tok_t});
  ET_CHECK_MSG(audio_tok_result.ok(), "token_embedding(audio) failed.");
  auto first_embed = audio_tok_result.get()[0].toTensor();
  const size_t embed_bytes =
      static_cast<size_t>(dim_) * first_embed.element_size();

  // Build padded embedding buffer (updated each step)
  auto audio_step_embeds =
      empty({1, DECODE_SEQ_LEN, static_cast<int>(dim_)}, model_dtype_);

  // Helper to fill all DECODE_SEQ_LEN slots with the same embedding
  auto fill_step_embeds = [&](const void* src) {
    for (int i = 0; i < DECODE_SEQ_LEN; i++) {
      std::memcpy(
          static_cast<char*>(audio_step_embeds->mutable_data_ptr()) +
              static_cast<size_t>(i) * embed_bytes,
          src,
          embed_bytes);
    }
  };

  // Initialize with token_embedding(24) for the first step
  fill_step_embeds(first_embed.const_data_ptr());

  for (int step = 0; step < config.max_tokens && dec_pos < max_seq_len_;
       step++) {
    // 1. Run text_decoder with current step's embedding
    std::vector<int64_t> pos_data = {dec_pos, dec_pos + 1, dec_pos + 2};
    auto pos_tensor = from_blob(
        pos_data.data(),
        {DECODE_SEQ_LEN},
        ::executorch::aten::ScalarType::Long);
    auto dec_result = model_->execute(
        "text_decoder", std::vector<EValue>{*audio_step_embeds, *pos_tensor});
    ET_CHECK_MSG(
        dec_result.ok(),
        "text_decoder failed at pos %ld.",
        static_cast<long>(dec_pos));
    auto hidden = dec_result.get()[0].toTensor();
    dec_pos++;

    // 2. Extract first position's hidden state: (1, DECODE_SEQ_LEN, dim) -> (1,
    // dim)
    auto h_2d = empty({1, static_cast<int>(dim_)}, model_dtype_);
    std::memcpy(
        h_2d->mutable_data_ptr(),
        hidden.const_data_ptr(),
        static_cast<size_t>(dim_) * hidden.element_size());

    // Generate noise
    auto noise_tensor =
        empty({1, static_cast<int>(n_acoustic_codebook_)}, model_dtype_);
    if (model_dtype_ == ::executorch::aten::ScalarType::BFloat16) {
      auto* ptr =
          noise_tensor->mutable_data_ptr<::executorch::aten::BFloat16>();
      for (int64_t i = 0; i < n_acoustic_codebook_; i++) {
        ptr[i] = ::executorch::aten::BFloat16(noise_dist(noise_rng));
      }
    } else {
      auto* ptr = noise_tensor->mutable_data_ptr<float>();
      for (int64_t i = 0; i < n_acoustic_codebook_; i++) {
        ptr[i] = noise_dist(noise_rng);
      }
    }

    // 3. Run acoustic transformer
    auto frame_result = model_->execute(
        "decode_audio_frame", std::vector<EValue>{*h_2d, *noise_tensor});
    ET_CHECK_MSG(frame_result.ok(), "decode_audio_frame failed.");
    auto codes_tensor = frame_result.get()[0].toTensor();
    const auto* codes_ptr = codes_tensor.const_data_ptr<int64_t>();

    // Check semantic code (first element) for END_AUDIO
    int64_t semantic_code = codes_ptr[0];
    if (semantic_code == END_AUDIO_SEMANTIC) {
      ET_LOG(Info, "END_AUDIO at frame %d", step);
      break;
    }

    // Accumulate codes
    for (int64_t i = 0; i < n_codebooks_; i++) {
      all_codes.push_back(codes_ptr[i]);
    }
    num_audio_frames++;
    num_generated++;

    if (token_cb) {
      token_cb("");
    }

    // 4. Prepare next step's embedding via audio_token_embedding.
    // Reshape codes from (1, n_codebooks) to (1, n_codebooks, 1) for the
    // multi-codebook embedding lookup, then pad to DECODE_SEQ_LEN.
    std::vector<int64_t> codes_col(static_cast<size_t>(n_codebooks_));
    for (int64_t i = 0; i < n_codebooks_; i++) {
      codes_col[static_cast<size_t>(i)] = codes_ptr[i];
    }
    auto codes_3d = from_blob(
        codes_col.data(),
        {1, static_cast<int>(n_codebooks_), 1},
        ::executorch::aten::ScalarType::Long);
    auto ate_result = model_->execute(
        "audio_token_embedding", std::vector<EValue>{*codes_3d});
    ET_CHECK_MSG(ate_result.ok(), "audio_token_embedding failed.");
    auto next_embed = ate_result.get()[0].toTensor(); // (1, 1, dim)
    fill_step_embeds(next_embed.const_data_ptr());
  }

  ET_LOG(
      Info,
      "Generated %d tokens, %ld audio frames",
      num_generated,
      static_cast<long>(num_audio_frames));

  // Output audio codes if requested
  if (audio_codes_out) {
    *audio_codes_out = all_codes;
  }

  // Decode audio codes to waveform via codec
  if (codec_ && num_audio_frames > 0) {
    return decode_audio(all_codes, num_audio_frames);
  }

  return {};
}

// ---------------------------------------------------------------------------
// Codec decode
// ---------------------------------------------------------------------------

std::vector<float> VoxtralTTSRunner::decode_audio(
    const std::vector<int64_t>& codes,
    int64_t n_frames) {
  ET_CHECK_MSG(codec_ != nullptr, "No codec loaded.");

  std::vector<float> waveform;
  const int64_t chunk = codec_chunk_size_;

  for (int64_t start = 0; start < n_frames; start += chunk) {
    int64_t this_chunk = std::min(chunk, n_frames - start);

    // Pad to full chunk if needed.
    // Strip _N_AUDIO_SPECIAL_TOKENS (2) offset from all codes:
    // decode_one_frame outputs codes with special token offset (semantic
    // argmax includes EMPTY=0/END=1 positions, acoustic codes are shifted
    // by +2). The codec expects raw indices: semantic in [0, codebook_size)
    // and acoustic in [0, levels-1].
    constexpr int64_t N_SPECIAL = 2;
    std::vector<int64_t> chunk_codes(
        static_cast<size_t>(chunk * n_codebooks_), 0);
    for (int64_t f = 0; f < this_chunk; f++) {
      for (int64_t c = 0; c < n_codebooks_; c++) {
        int64_t code =
            codes[static_cast<size_t>((start + f) * n_codebooks_ + c)];
        code -= N_SPECIAL;
        chunk_codes[static_cast<size_t>(c * chunk + f)] = code;
      }
    }

    // Shape: (1, n_codebooks, chunk)
    auto codes_tensor = from_blob(
        chunk_codes.data(),
        {1, static_cast<int>(n_codebooks_), static_cast<int>(chunk)},
        ::executorch::aten::ScalarType::Long);

    auto result =
        codec_->execute("audio_decoder", std::vector<EValue>{*codes_tensor});
    ET_CHECK_MSG(result.ok(), "audio_decoder failed.");

    auto wav_tensor = result.get()[0].toTensor();
    const int64_t wav_samples = wav_tensor.numel();

    // Only take the valid portion (trim padding)
    const int64_t valid_samples = this_chunk * (wav_samples / chunk);

    const float* wav_data = wav_tensor.const_data_ptr<float>();
    waveform.insert(waveform.end(), wav_data, wav_data + valid_samples);
  }

  ET_LOG(
      Info,
      "Decoded %ld audio frames -> %zu waveform samples",
      static_cast<long>(n_frames),
      waveform.size());
  return waveform;
}

// ---------------------------------------------------------------------------
// WAV writer
// ---------------------------------------------------------------------------

bool write_wav(
    const std::string& path,
    const float* samples,
    int64_t num_samples,
    int32_t sample_rate) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    ET_LOG(Error, "Cannot open %s for writing.", path.c_str());
    return false;
  }

  const int16_t num_channels = 1;
  const int16_t bits_per_sample = 16;
  const int32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
  const int16_t block_align = num_channels * bits_per_sample / 8;
  const int32_t data_size = static_cast<int32_t>(num_samples) * block_align;
  const int32_t chunk_size = 36 + data_size;

  // RIFF header
  file.write("RIFF", 4);
  file.write(reinterpret_cast<const char*>(&chunk_size), 4);
  file.write("WAVE", 4);

  // fmt sub-chunk
  file.write("fmt ", 4);
  int32_t fmt_size = 16;
  int16_t audio_format = 1; // PCM
  file.write(reinterpret_cast<const char*>(&fmt_size), 4);
  file.write(reinterpret_cast<const char*>(&audio_format), 2);
  file.write(reinterpret_cast<const char*>(&num_channels), 2);
  file.write(reinterpret_cast<const char*>(&sample_rate), 4);
  file.write(reinterpret_cast<const char*>(&byte_rate), 4);
  file.write(reinterpret_cast<const char*>(&block_align), 2);
  file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

  // data sub-chunk
  file.write("data", 4);
  file.write(reinterpret_cast<const char*>(&data_size), 4);

  // Convert float32 [-1, 1] to int16 and write
  for (int64_t i = 0; i < num_samples; i++) {
    float s = std::max(-1.0f, std::min(1.0f, samples[i]));
    int16_t pcm = static_cast<int16_t>(s * 32767.0f);
    file.write(reinterpret_cast<const char*>(&pcm), 2);
  }

  file.close();
  ET_LOG(
      Info,
      "Wrote %s: %ld samples, %.1f seconds",
      path.c_str(),
      static_cast<long>(num_samples),
      static_cast<double>(num_samples) / sample_rate);
  return true;
}

} // namespace voxtral_tts
