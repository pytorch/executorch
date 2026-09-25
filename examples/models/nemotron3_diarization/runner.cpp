/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * The audio frontend and streaming/cache behavior in this example are adapted
 * from mlx-audio's Nemotron Diarization and Sortformer
 * implementations (https://github.com/Blaizzy/mlx-audio).
 *
 * MIT License
 *
 * Copyright (c) 2024 Prince Canuma
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "runner.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <executorch/extension/tensor/tensor.h>

// MLX links a different pocketfft version; keep these template symbols
// separate.
#define pocketfft nemotron3_pocketfft
#define POCKETFFT_CACHE_SIZE 1
#include <pocketfft_hdronly.h>
#undef POCKETFFT_CACHE_SIZE
#undef pocketfft

namespace nemotron3 {
namespace {
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::extension::from_blob;
using executorch::runtime::Error;
using executorch::runtime::EValue;

constexpr int64_t kMinEncoderFrames = 2;

std::vector<EValue> execute(
    executorch::extension::Module& model,
    const char* method,
    const std::vector<EValue>& inputs = {}) {
  auto result = model.execute(method, inputs);
  if (!result.ok()) {
    throw std::runtime_error(
        std::string(method) +
        " failed: " + executorch::runtime::to_string(result.error()));
  }
  return std::move(result.get());
}

std::vector<float> copy_floats(const EValue& value) {
  const auto& tensor = value.toTensor();
  if (tensor.scalar_type() != ScalarType::Float) {
    throw std::runtime_error("Expected float32 model output");
  }
  const auto* data = tensor.const_data_ptr<float>();
  return {data, data + tensor.numel()};
}
} // namespace

StreamingConfig StreamingConfig::from_preset(const std::string& preset) {
  if (preset == "offline") {
    return {340, 40, 40, 300};
  }
  if (preset == "low") {
    return {9, 4, 264, 222};
  }
  if (preset == "very_low") {
    return {6, 2, 264, 222};
  }
  if (preset == "ultra_low") {
    return {3, 1, 264, 222};
  }
  throw std::invalid_argument("Unknown streaming preset: " + preset);
}

Runner::Runner(const std::string& model_path, StreamingConfig config)
    : model_(model_path, executorch::extension::Module::LoadMode::Mmap),
      config_(config) {
  if (model_.load() != Error::Ok) {
    throw std::runtime_error("Could not load " + model_path);
  }
  const auto integer = [this](const char* name) {
    return execute(model_, name).at(0).toInt();
  };
  const auto real = [this](const char* name) {
    return static_cast<float>(execute(model_, name).at(0).toDouble());
  };
  if (integer("format_version") != 1) {
    throw std::runtime_error("Unsupported Nemotron model format");
  }
  sample_rate_ = integer("sample_rate");
  hop_ = integer("hop_length");
  n_fft_ = integer("n_fft");
  num_mels_ = integer("num_mels");
  d_model_ = integer("d_model");
  num_speakers_ = integer("num_speakers");
  factor_ = integer("subsampling_factor");
  pad_to_ = integer("pad_to");
  cache_capacity_ = integer("spkcache_len");
  silence_frames_ = integer("silence_frames");
  max_encoder_ = integer("max_encoder_frames");
  max_features_ = integer("max_feature_frames");
  preemphasis_ = real("preemphasis");
  score_threshold_ = real("pred_score_threshold");
  latest_boost_ = real("scores_boost_latest");
  strong_boost_ = real("strong_boost_rate");
  weak_boost_ = real("weak_boost_rate");
  min_positive_ = real("min_pos_scores_rate");
  if (sample_rate_ <= 0 || hop_ <= 0 || hop_ > n_fft_ / 2 || num_mels_ <= 0 ||
      d_model_ <= 0 || num_speakers_ <= 0 || factor_ != 8 || pad_to_ < 0 ||
      silence_frames_ < 0 ||
      cache_capacity_ < (silence_frames_ + 1) * num_speakers_ ||
      config_.chunk <= 0 || config_.right_context < 0 || config_.fifo < 0 ||
      config_.update_period <= 0 ||
      config_.chunk + config_.right_context < kMinEncoderFrames ||
      (config_.chunk + config_.right_context) * factor_ > max_features_ ||
      cache_capacity_ + config_.fifo + config_.chunk + config_.right_context >
          max_encoder_ ||
      !(preemphasis_ >= 0 && preemphasis_ <= 1) ||
      !(score_threshold_ > 0 && score_threshold_ < 1) ||
      !(min_positive_ >= 0 && min_positive_ <= 1) ||
      !std::isfinite(latest_boost_) || latest_boost_ < 0 ||
      !std::isfinite(strong_boost_) || strong_boost_ < 0 ||
      !std::isfinite(weak_boost_) || weak_boost_ < 0) {
    throw std::invalid_argument(
        "Invalid model metadata or streaming configuration");
  }
  auto constants = execute(model_, "frontend_constants");
  if (constants.size() != 3) {
    throw std::runtime_error(
        "Expected window, mel filters, and silence embedding");
  }
  auto window = copy_floats(constants[0]);
  mel_filters_ = copy_floats(constants[1]);
  silence_ = copy_floats(constants[2]);
  if (window.empty() || window.size() > static_cast<size_t>(n_fft_) ||
      mel_filters_.size() !=
          static_cast<size_t>(num_mels_ * (n_fft_ / 2 + 1)) ||
      silence_.size() != static_cast<size_t>(d_model_)) {
    throw std::runtime_error("Unexpected frontend constant dimensions");
  }
  window_.resize(n_fft_, 0.0f);
  std::copy(
      window.begin(),
      window.end(),
      window_.begin() + (n_fft_ - window.size()) / 2);
}

void Runner::reset() {
  audio_.clear();
  cache_.clear();
  cache_probs_.clear();
  fifo_.clear();
  samples_received_ = sample_offset_ = frames_processed_ = 0;
  compressed_ = finished_ = false;
}

std::vector<float> Runner::features(int64_t count) const {
  const int64_t bins = n_fft_ / 2 + 1;
  std::vector<float> output(count * num_mels_, 0.0f);
  std::vector<float> frame(n_fft_);
  std::vector<std::complex<float>> spectrum(bins);
  const auto sample = [this](int64_t position) {
    if (position < 0 || position >= samples_received_) {
      return 0.0f;
    }
    if (position < sample_offset_) {
      throw std::logic_error("Missing PCM history for STFT");
    }
    return audio_.at(position - sample_offset_);
  };
  for (int64_t t = 0; t < count; ++t) {
    const int64_t global = frames_processed_ + t;
    if (global >= samples_received_ / hop_) {
      continue;
    }
    for (int64_t j = 0; j < n_fft_; ++j) {
      const int64_t position = global * hop_ + j - n_fft_ / 2;
      frame[j] = position >= 0 && position < samples_received_
          ? (sample(position) - preemphasis_ * sample(position - 1)) *
              window_[j]
          : 0.0f;
    }
    nemotron3_pocketfft::r2c<float>(
        {static_cast<size_t>(n_fft_)},
        {sizeof(float)},
        {sizeof(std::complex<float>)},
        0,
        true,
        frame.data(),
        spectrum.data(),
        1.0f);
    for (int64_t mel = 0; mel < num_mels_; ++mel) {
      double power = 0.0;
      for (int64_t j = 0; j < bins; ++j) {
        power += std::norm(spectrum[j]) * mel_filters_[mel * bins + j];
      }
      output[t * num_mels_ + mel] =
          std::log(static_cast<float>(power) + 0x1p-24f);
    }
  }
  return output;
}

std::vector<float>
Runner::step(int64_t feature_frames, int64_t valid, int64_t central) {
  const int64_t cache_len = cache_.size() / d_model_;
  const int64_t fifo_len = fifo_.size() / d_model_;
  const int64_t prefix = cache_len + fifo_len;
  if (prefix < kMinEncoderFrames) {
    feature_frames =
        std::max(feature_frames, (kMinEncoderFrames - prefix) * factor_);
  }
  // Keep the physical padded window: its masked queries still affect the
  // neighboring valid frame through the subpixel convolution, as in NeMo.
  feature_frames += (factor_ - feature_frames % factor_) % factor_;
  if (feature_frames > max_features_) {
    throw std::runtime_error("Feature window exceeds export bound");
  }
  auto mel = features(feature_frames);
  auto input = from_blob(
      mel.data(),
      {1,
       static_cast<SizesType>(feature_frames),
       static_cast<SizesType>(num_mels_)},
      ScalarType::Float);
  auto pre = execute(model_, "pre_encode", {input});
  auto chunk = copy_floats(pre.at(0));
  if (chunk.size() !=
      static_cast<size_t>(feature_frames / factor_ * d_model_)) {
    throw std::runtime_error("Unexpected pre_encode output dimensions");
  }
  const int64_t total = prefix + feature_frames / factor_;
  if (total < kMinEncoderFrames || total > max_encoder_) {
    throw std::runtime_error("Encoder window outside export bounds");
  }
  std::vector<float> combined;
  combined.reserve(total * d_model_);
  combined.insert(combined.end(), cache_.begin(), cache_.end());
  combined.insert(combined.end(), fifo_.begin(), fifo_.end());
  combined.insert(combined.end(), chunk.begin(), chunk.end());
  int64_t length = prefix + (valid + factor_ - 1) / factor_;
  auto embeddings = from_blob(
      combined.data(),
      {1, static_cast<SizesType>(total), static_cast<SizesType>(d_model_)},
      ScalarType::Float);
  auto lengths = from_blob(&length, {1}, ScalarType::Long);
  auto encoded = execute(model_, "encode", {embeddings, lengths});
  auto high = copy_floats(encoded.at(0));
  if (high.size() != static_cast<size_t>(total * factor_ * num_speakers_)) {
    throw std::runtime_error("Unexpected encode output dimensions");
  }
  std::vector<float> low(total * num_speakers_, 0.0f);
  for (int64_t t = 0; t < length; ++t) {
    for (int64_t s = 0; s < num_speakers_; ++s) {
      float sum = 0;
      for (int64_t k = 0; k < factor_; ++k) {
        sum += high[(t * factor_ + k) * num_speakers_ + s];
      }
      low[t * num_speakers_ + s] = sum / factor_;
    }
  }
  const int64_t n = (central + factor_ - 1) / factor_;
  fifo_.insert(fifo_.end(), chunk.begin(), chunk.begin() + n * d_model_);
  const int64_t new_fifo_len = fifo_len + n;
  if (new_fifo_len > config_.fifo) {
    const int64_t pop = std::min(
        new_fifo_len,
        std::max(config_.update_period, new_fifo_len - config_.fifo));
    if (!compressed_) {
      cache_probs_.assign(low.begin(), low.begin() + cache_len * num_speakers_);
    }
    cache_.insert(cache_.end(), fifo_.begin(), fifo_.begin() + pop * d_model_);
    cache_probs_.insert(
        cache_probs_.end(),
        low.begin() + cache_len * num_speakers_,
        low.begin() + (cache_len + pop) * num_speakers_);
    fifo_.erase(fifo_.begin(), fifo_.begin() + pop * d_model_);
    if (cache_.size() / d_model_ > static_cast<size_t>(cache_capacity_)) {
      compress_cache();
      compressed_ = true;
    }
  }
  frames_processed_ += central;
  return {
      high.begin() + prefix * factor_ * num_speakers_,
      high.begin() + (prefix * factor_ + central) * num_speakers_};
}

void Runner::compress_cache() {
  const int64_t count = cache_.size() / d_model_;
  const int64_t budget = cache_capacity_ / num_speakers_ - silence_frames_;
  const int64_t min_positive = std::floor(budget * min_positive_);
  const int64_t scored = count + silence_frames_;
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  std::vector<float> scores(scored * num_speakers_, negative_infinity);
  for (int64_t t = 0; t < count; ++t) {
    float sum = 0;
    for (int64_t s = 0; s < num_speakers_; ++s) {
      sum += std::log(std::max(
          1.0f - cache_probs_[t * num_speakers_ + s], score_threshold_));
    }
    for (int64_t s = 0; s < num_speakers_; ++s) {
      const float p = cache_probs_[t * num_speakers_ + s];
      if (p > 0.5f) {
        scores[s * scored + t] = std::log(std::max(p, score_threshold_)) -
            std::log(std::max(1.0f - p, score_threshold_)) + sum -
            std::log(0.5f);
      }
    }
  }
  for (int64_t s = 0; s < num_speakers_; ++s) {
    auto first = scores.begin() + s * scored;
    const int64_t positive =
        std::count_if(first, first + count, [](float v) { return v > 0; });
    for (int64_t t = 0; t < count; ++t) {
      if (positive >= min_positive && first[t] <= 0) {
        first[t] = negative_infinity;
      }
      if (t >= cache_capacity_) {
        first[t] += latest_boost_;
      }
    }
    std::vector<int64_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    for (const auto& boost :
         {std::make_pair(strong_boost_, 2.0f),
          std::make_pair(weak_boost_, 1.0f)}) {
      const int64_t k = static_cast<int64_t>(std::min(
          static_cast<double>(count),
          std::floor(static_cast<double>(budget) * boost.first)));
      // Prefer earlier frames on ties; torch.topk does not guarantee tie order.
      std::partial_sort(
          order.begin(), order.begin() + k, order.end(), [&](auto a, auto b) {
            return first[a] == first[b] ? a < b : first[a] > first[b];
          });
      for (int64_t j = 0; j < k; ++j) {
        first[order[j]] -= boost.second * std::log(0.5f);
      }
    }
    std::fill(
        first + count, first + scored, std::numeric_limits<float>::infinity());
  }
  std::vector<int64_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(
      order.begin(),
      order.begin() + cache_capacity_,
      order.end(),
      [&](auto a, auto b) {
        return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
      });
  order.resize(cache_capacity_);
  for (auto& i : order) {
    if (scores[i] == negative_infinity) {
      i = scores.size();
    }
  }
  std::sort(order.begin(), order.end());
  std::vector<float> embeddings(cache_capacity_ * d_model_);
  std::vector<float> probs(cache_capacity_ * num_speakers_, 0.0f);
  for (int64_t t = 0; t < cache_capacity_; ++t) {
    const int64_t index = order[t] % scored;
    if (order[t] < static_cast<int64_t>(scores.size()) && index < count) {
      std::copy_n(
          cache_.data() + index * d_model_,
          d_model_,
          embeddings.data() + t * d_model_);
      std::copy_n(
          cache_probs_.data() + index * num_speakers_,
          num_speakers_,
          probs.data() + t * num_speakers_);
    } else {
      std::copy(
          silence_.begin(), silence_.end(), embeddings.begin() + t * d_model_);
    }
  }
  cache_.swap(embeddings);
  cache_probs_.swap(probs);
}

std::vector<float> Runner::feed(const float* audio, size_t count, bool final) {
  if (finished_) {
    throw std::logic_error(
        "Stream is finished; call reset before feeding more audio");
  }
  if (count && audio == nullptr) {
    throw std::invalid_argument("Missing audio samples");
  }
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(audio[i])) {
      throw std::invalid_argument("Audio contains non-finite samples");
    }
  }
  try {
    if (count) {
      audio_.insert(audio_.end(), audio, audio + count);
    }
    samples_received_ += count;
    std::vector<float> output;
    const int64_t central = config_.chunk * factor_;
    const int64_t right = config_.right_context * factor_;
    while (true) {
      const int64_t available = samples_received_ / hop_ - frames_processed_;
      const int64_t needed =
          (frames_processed_ + central + right - 1) * hop_ + n_fft_ / 2;
      if (available <= 0 || (!final && samples_received_ < needed)) {
        break;
      }
      const int64_t n = std::min(central, available);
      int64_t window = central + right;
      if (final) {
        int64_t total_frames = samples_received_ / hop_ + 1;
        if (pad_to_) {
          total_frames += (pad_to_ - total_frames % pad_to_) % pad_to_;
        }
        window = std::min(window, total_frames - frames_processed_);
      }
      auto probs = step(window, std::min(window, available), n);
      output.insert(output.end(), probs.begin(), probs.end());
      const int64_t keep_from =
          std::max<int64_t>(0, frames_processed_ * hop_ - n_fft_ / 2 - 1);
      audio_.erase(
          audio_.begin(), audio_.begin() + (keep_from - sample_offset_));
      sample_offset_ = keep_from;
    }
    finished_ = final;
    if (final) {
      audio_.clear();
    }
    return output;
  } catch (...) {
    finished_ = true;
    throw;
  }
}

} // namespace nemotron3
