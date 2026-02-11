/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sortformer_runner.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>

using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

namespace sortformer {
namespace {

constexpr int64_t kMaxMelFrames = 4000; // pre_encode static input shape
constexpr int kMelBins = 128; // Mel spectrogram frequency bins
constexpr int64_t kSampleRate = 16000;
constexpr int64_t kMaxAudioSamples = kSampleRate * 120; // 120s preprocessor max

// Compress the speaker cache when it exceeds max_size frames.
//
// Scores each frame by its max speaker probability and keeps the top max_size
// frames (sorted by original time order). This is a simplification of NeMo's
// log-odds importance scoring, which uses per-speaker AOSC (Average Online
// Speaker Confusion) scores. The simplified version works well in practice
// because high max-probability frames tend to be the same ones NeMo would keep.
void compress_cache(
    std::vector<float>& embs,
    std::vector<float>& preds,
    int64_t& size,
    int64_t max_size,
    int64_t d_model,
    int64_t max_spks) {
  if (size <= max_size)
    return;

  std::vector<std::pair<float, int64_t>> scored;
  scored.reserve(static_cast<size_t>(size));
  for (int64_t i = 0; i < size; i++) {
    float max_p = 0.0f;
    for (int64_t s = 0; s < max_spks; s++) {
      max_p = std::max(max_p, preds[static_cast<size_t>(i * max_spks + s)]);
    }
    scored.push_back({max_p, i});
  }

  std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
    return a.first > b.first;
  });

  std::vector<int64_t> keep;
  keep.reserve(static_cast<size_t>(max_size));
  for (int64_t i = 0; i < max_size; i++) {
    keep.push_back(scored[static_cast<size_t>(i)].second);
  }
  std::sort(keep.begin(), keep.end());

  std::vector<float> new_embs(static_cast<size_t>(max_size * d_model));
  std::vector<float> new_preds(static_cast<size_t>(max_size * max_spks));
  for (int64_t i = 0; i < max_size; i++) {
    int64_t src = keep[static_cast<size_t>(i)];
    std::memcpy(
        &new_embs[static_cast<size_t>(i * d_model)],
        &embs[static_cast<size_t>(src * d_model)],
        static_cast<size_t>(d_model) * sizeof(float));
    std::memcpy(
        &new_preds[static_cast<size_t>(i * max_spks)],
        &preds[static_cast<size_t>(src * max_spks)],
        static_cast<size_t>(max_spks) * sizeof(float));
  }

  embs = std::move(new_embs);
  preds = std::move(new_preds);
  size = max_size;
}

} // namespace

// Read model parameters from .pte constant_methods. These are baked into the
// exported model by export_sortformer.py and describe the preprocessing config
// and architecture dimensions needed to set up the streaming pipeline.
SortformerRunner::SortformerRunner(const std::string& model_path) {
  ET_LOG(Info, "Loading model from: %s", model_path.c_str());
  model_ = std::make_unique<Module>(model_path, Module::LoadMode::Mmap);
  auto load_error = model_->load();
  if (load_error != Error::Ok) {
    ET_LOG(Error, "Failed to load model.");
    return;
  }

  std::vector<EValue> empty;
  auto ws = model_->execute("window_stride", empty);
  auto ss = model_->execute("subsampling_factor", empty);
  auto sc = model_->execute("spkcache_len", empty);
  auto ms = model_->execute("max_num_of_spks", empty);

  window_stride_ = ws.ok() ? ws.get()[0].toDouble() : 0.01;
  subsampling_factor_ = ss.ok() ? ss.get()[0].toInt() : 8;
  spkcache_len_ = sc.ok() ? sc.get()[0].toInt() : 188;
  max_spks_ = ms.ok() ? ms.get()[0].toInt() : 4;
  frame_duration_ = window_stride_ * static_cast<double>(subsampling_factor_);
}

std::pair<std::vector<float>, int64_t> SortformerRunner::run_preprocessor(
    const float* audio,
    int64_t num_samples) {
  auto audio_tensor = from_blob(
      const_cast<float*>(audio),
      {static_cast<::executorch::aten::SizesType>(num_samples)},
      ::executorch::aten::ScalarType::Float);
  std::vector<int64_t> len_data = {num_samples};
  auto len_tensor =
      from_blob(len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  auto result = model_->execute(
      "preprocessor", std::vector<EValue>{audio_tensor, len_tensor});
  if (!result.ok()) {
    ET_LOG(Error, "Preprocessor failed.");
    return {{}, 0};
  }

  auto& outputs = result.get();
  auto mel = outputs[0].toTensor();
  int64_t mel_len = outputs[1].toTensor().const_data_ptr<int64_t>()[0];
  int64_t mel_T = mel.sizes()[2]; // (1, 128, T_mel)
  int64_t valid_mel = std::min(mel_T, mel_len);

  // Transpose from (1, 128, T) channels-first to (T, 128) channels-last.
  // The preprocessor outputs channels-first but pre_encode expects
  // channels-last.
  const float* mel_ptr = mel.const_data_ptr<float>();
  std::vector<float> transposed(static_cast<size_t>(valid_mel) * kMelBins);
  for (int64_t t = 0; t < valid_mel; t++) {
    for (int f = 0; f < kMelBins; f++) {
      transposed[static_cast<size_t>(t * kMelBins + f)] =
          mel_ptr[static_cast<size_t>(f * mel_T + t)];
    }
  }

  return {std::move(transposed), valid_mel};
}

int64_t SortformerRunner::run_pre_encode(
    const float* mel_transposed,
    int64_t valid_mel_len,
    std::vector<float>& all_embs) {
  // Pad to kMaxMelFrames — pre_encode requires a static input shape because
  // the conv-derived time expression 1+((L-1)//8) creates a guard that
  // torch.export's solver can't prove against nn.Linear (see model.md #5).
  std::vector<float> padded(
      static_cast<size_t>(kMaxMelFrames) * kMelBins, 0.0f);
  int64_t copy_len = std::min(valid_mel_len, kMaxMelFrames);
  std::memcpy(
      padded.data(),
      mel_transposed,
      static_cast<size_t>(copy_len * kMelBins) * sizeof(float));

  auto chunk_tensor = from_blob(
      padded.data(),
      {1,
       static_cast<::executorch::aten::SizesType>(kMaxMelFrames),
       static_cast<::executorch::aten::SizesType>(kMelBins)},
      ::executorch::aten::ScalarType::Float);
  std::vector<int64_t> len_data = {copy_len};
  auto len_tensor =
      from_blob(len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

  auto result = model_->execute(
      "pre_encode", std::vector<EValue>{chunk_tensor, len_tensor});
  if (!result.ok()) {
    ET_LOG(Error, "pre_encode failed.");
    return 0;
  }

  auto& outputs = result.get();
  auto embs = outputs[0].toTensor();
  int64_t emb_len = outputs[1].toTensor().const_data_ptr<int64_t>()[0];

  if (d_model_ == 0) {
    d_model_ = embs.sizes()[2]; // 512
  }

  const float* emb_ptr = embs.const_data_ptr<float>();
  all_embs.insert(
      all_embs.end(),
      emb_ptr,
      emb_ptr + static_cast<size_t>(emb_len * d_model_));
  return emb_len;
}

// Streaming encode loop. For each chunk of embeddings:
//   1. Concatenate [speaker_cache | FIFO | current_chunk] → encode →
//   predictions
//   2. Extract only the current chunk's predictions as output
//   3. Update FIFO (push oldest frames to speaker cache when it overflows)
//   4. Compress speaker cache if it exceeds spkcache_len_ (188 frames)
//
// The cache and FIFO exist purely to give the conformer+transformer context
// from earlier audio. The model itself is stateless per call.
SortformerRunner::Result SortformerRunner::run_streaming_encode(
    const std::vector<float>& all_embs,
    int64_t total_emb_len,
    float threshold,
    const StreamingConfig& config,
    SegmentCallback segment_cb) {
  // Speaker cache: long-term memory of the most speaker-discriminative frames.
  std::vector<float> cache_embs;
  std::vector<float> cache_preds;
  int64_t cache_size = 0;

  // FIFO: short-term sliding window of recent embeddings.
  std::vector<float> fifo_embs;
  int64_t fifo_size = 0;

  // Per-speaker activity state for segment emission.
  std::vector<bool> spk_active(static_cast<size_t>(max_spks_), false);
  std::vector<int64_t> spk_start_frame(static_cast<size_t>(max_spks_), 0);
  std::vector<int64_t> spk_active_frames(static_cast<size_t>(max_spks_), 0);
  int64_t num_output_frames = 0;
  int num_segments = 0;

  for (int64_t offset = 0; offset < total_emb_len; offset += config.chunk_len) {
    int64_t cur_chunk_len = std::min(config.chunk_len, total_emb_len - offset);
    int64_t total_len = cache_size + fifo_size + cur_chunk_len;

    // Build [cache | fifo | chunk]
    std::vector<float> concat(static_cast<size_t>(total_len * d_model_));
    size_t dst = 0;
    if (cache_size > 0) {
      std::memcpy(
          concat.data() + dst,
          cache_embs.data(),
          static_cast<size_t>(cache_size * d_model_) * sizeof(float));
      dst += static_cast<size_t>(cache_size * d_model_);
    }
    if (fifo_size > 0) {
      std::memcpy(
          concat.data() + dst,
          fifo_embs.data(),
          static_cast<size_t>(fifo_size * d_model_) * sizeof(float));
      dst += static_cast<size_t>(fifo_size * d_model_);
    }
    std::memcpy(
        concat.data() + dst,
        all_embs.data() + static_cast<size_t>(offset * d_model_),
        static_cast<size_t>(cur_chunk_len * d_model_) * sizeof(float));

    auto enc_tensor = from_blob(
        concat.data(),
        {1,
         static_cast<::executorch::aten::SizesType>(total_len),
         static_cast<::executorch::aten::SizesType>(d_model_)},
        ::executorch::aten::ScalarType::Float);
    std::vector<int64_t> enc_len_data = {total_len};
    auto enc_len_tensor = from_blob(
        enc_len_data.data(), {1}, ::executorch::aten::ScalarType::Long);

    auto enc_result = model_->execute(
        "encode", std::vector<EValue>{enc_tensor, enc_len_tensor});
    if (!enc_result.ok()) {
      ET_LOG(Error, "encode failed.");
      break;
    }
    auto enc_preds = enc_result.get()[0].toTensor();
    const float* preds_ptr = enc_preds.const_data_ptr<float>();

    // Extract the cache region's predictions from the encode output. Because
    // the cache frames were encoded alongside the new chunk, their predictions
    // now reflect updated context. These are used as scoring input for
    // compress_cache.
    if (cache_size > 0) {
      cache_preds.resize(static_cast<size_t>(cache_size * max_spks_));
      std::memcpy(
          cache_preds.data(),
          preds_ptr,
          static_cast<size_t>(cache_size * max_spks_) * sizeof(float));
    }

    // Extract chunk predictions (skip cache and FIFO regions) and emit
    // segments via threshold-based activity detection. Each speaker slot's
    // probability is compared against the threshold independently.
    size_t chunk_pred_start =
        static_cast<size_t>((cache_size + fifo_size) * max_spks_);
    for (int64_t t = 0; t < cur_chunk_len; t++) {
      int64_t global_frame = num_output_frames + t;
      for (int spk = 0; spk < static_cast<int>(max_spks_); spk++) {
        float prob = preds_ptr
            [chunk_pred_start + static_cast<size_t>(t * max_spks_ + spk)];
        size_t si = static_cast<size_t>(spk);
        if (prob > threshold) {
          spk_active_frames[si]++;
          if (!spk_active[si]) {
            spk_active[si] = true;
            spk_start_frame[si] = global_frame;
          }
        } else if (spk_active[si]) {
          spk_active[si] = false;
          segment_cb(
              {spk_start_frame[si] * frame_duration_,
               global_frame * frame_duration_,
               spk});
          num_segments++;
        }
      }
    }
    num_output_frames += cur_chunk_len;

    // Update FIFO: append current chunk's embeddings. When the FIFO exceeds
    // its configured size, the oldest frames overflow into the speaker cache.
    int64_t combined = fifo_size + cur_chunk_len;
    int64_t overflow = std::max(int64_t(0), combined - config.fifo_len);

    if (overflow > 0) {
      int64_t from_fifo = std::min(overflow, fifo_size);
      int64_t from_chunk = overflow - from_fifo;

      if (from_fifo > 0) {
        cache_embs.insert(
            cache_embs.end(),
            fifo_embs.begin(),
            fifo_embs.begin() + static_cast<ptrdiff_t>(from_fifo * d_model_));
        size_t fifo_pred_start = static_cast<size_t>(cache_size * max_spks_);
        cache_preds.insert(
            cache_preds.end(),
            preds_ptr + fifo_pred_start,
            preds_ptr + fifo_pred_start +
                static_cast<size_t>(from_fifo * max_spks_));
        cache_size += from_fifo;

        fifo_embs.erase(
            fifo_embs.begin(),
            fifo_embs.begin() + static_cast<ptrdiff_t>(from_fifo * d_model_));
        fifo_size -= from_fifo;
      }

      if (from_chunk > 0) {
        size_t chunk_emb_start = static_cast<size_t>(offset * d_model_);
        cache_embs.insert(
            cache_embs.end(),
            all_embs.data() + chunk_emb_start,
            all_embs.data() + chunk_emb_start +
                static_cast<size_t>(from_chunk * d_model_));
        size_t orig_chunk_pred_start = chunk_pred_start;
        cache_preds.insert(
            cache_preds.end(),
            preds_ptr + orig_chunk_pred_start,
            preds_ptr + orig_chunk_pred_start +
                static_cast<size_t>(from_chunk * max_spks_));
        cache_size += from_chunk;
      }

      int64_t chunk_keep = cur_chunk_len - from_chunk;
      if (chunk_keep > 0) {
        size_t keep_start =
            static_cast<size_t>((offset + from_chunk) * d_model_);
        fifo_embs.insert(
            fifo_embs.end(),
            all_embs.data() + keep_start,
            all_embs.data() + keep_start +
                static_cast<size_t>(chunk_keep * d_model_));
        fifo_size += chunk_keep;
      }
    } else {
      size_t emb_start = static_cast<size_t>(offset * d_model_);
      fifo_embs.insert(
          fifo_embs.end(),
          all_embs.data() + emb_start,
          all_embs.data() + emb_start +
              static_cast<size_t>(cur_chunk_len * d_model_));
      fifo_size += cur_chunk_len;
    }

    // Compress speaker cache if it exceeds the limit (188 frames). Keeps the
    // most speaker-discriminative frames ranked by max speaker probability.
    if (cache_size > spkcache_len_) {
      compress_cache(
          cache_embs,
          cache_preds,
          cache_size,
          spkcache_len_,
          d_model_,
          max_spks_);
    }
  }

  // Close any still-active segments
  for (int spk = 0; spk < static_cast<int>(max_spks_); spk++) {
    size_t si = static_cast<size_t>(spk);
    if (spk_active[si]) {
      segment_cb(
          {spk_start_frame[si] * frame_duration_,
           num_output_frames * frame_duration_,
           spk});
      num_segments++;
    }
  }

  return {num_output_frames, num_segments, std::move(spk_active_frames)};
}

// Orchestrates the full diarization pipeline for arbitrary-length audio.
// Stages 1-2 (preprocessor → pre_encode) run first to collect all embeddings,
// then stage 3 (streaming encode) processes them in chunks with cache/FIFO
// context. Audio longer than 120s is automatically chunked for the
// preprocessor.
SortformerRunner::Result SortformerRunner::diarize(
    const float* audio_data,
    int64_t num_samples,
    float threshold,
    const StreamingConfig& config,
    SegmentCallback segment_cb) {
  // Stages 1-2: preprocessor → pre_encode, chunked for arbitrary length
  std::vector<float> all_embs;
  int64_t total_emb_len = 0;

  for (int64_t audio_offset = 0; audio_offset < num_samples;
       audio_offset += kMaxAudioSamples) {
    int64_t chunk_samples =
        std::min(kMaxAudioSamples, num_samples - audio_offset);

    ET_LOG(
        Info,
        "Running preprocessor (%.1fs-%.1fs)...",
        static_cast<double>(audio_offset) / kSampleRate,
        static_cast<double>(audio_offset + chunk_samples) / kSampleRate);

    auto [mel_transposed, valid_mel] =
        run_preprocessor(audio_data + audio_offset, chunk_samples);

    // Run pre_encode in sub-chunks of kMaxMelFrames
    for (int64_t mel_offset = 0; mel_offset < valid_mel;
         mel_offset += kMaxMelFrames) {
      int64_t sub_len = std::min(kMaxMelFrames, valid_mel - mel_offset);
      int64_t added = run_pre_encode(
          mel_transposed.data() + static_cast<size_t>(mel_offset * kMelBins),
          sub_len,
          all_embs);
      total_emb_len += added;
    }
  }

  ET_LOG(
      Info,
      "Total embeddings: %lld frames x %lld dims",
      static_cast<long long>(total_emb_len),
      static_cast<long long>(d_model_));

  ET_LOG(
      Info,
      "Streaming: chunk=%lld, fifo=%lld, cache=%lld, frame=%.3fs",
      static_cast<long long>(config.chunk_len),
      static_cast<long long>(config.fifo_len),
      static_cast<long long>(spkcache_len_),
      frame_duration_);

  // Stage 3: streaming encode
  return run_streaming_encode(
      all_embs, total_emb_len, threshold, config, std::move(segment_cb));
}

} // namespace sortformer
