/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "silero_vad_runner.h"

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

namespace silero_vad {

SileroVadRunner::SileroVadRunner(const std::string& model_path) {
  ET_LOG(Info, "Loading model from: %s", model_path.c_str());
  model_ = std::make_unique<Module>(model_path, Module::LoadMode::Mmap);
  auto load_error = model_->load();
  if (load_error != Error::Ok) {
    ET_LOG(Error, "Failed to load model.");
    return;
  }

  std::vector<EValue> empty;
  auto sr = model_->execute("sample_rate", empty);
  auto ws = model_->execute("window_size", empty);
  auto cs = model_->execute("context_size", empty);

  sample_rate_ = sr.ok() ? sr.get()[0].toInt() : 16000;
  window_size_ = ws.ok() ? ws.get()[0].toInt() : 512;
  context_size_ = cs.ok() ? cs.get()[0].toInt() : 64;
  input_size_ = window_size_ + context_size_;
  frame_duration_ = static_cast<double>(window_size_) / sample_rate_;
  reset_stream();
}

void SileroVadRunner::reset_stream() {
  stream_state_data_.assign(static_cast<size_t>(2 * kHiddenDim), 0.0f);
  stream_context_.assign(static_cast<size_t>(context_size_), 0.0f);
  stream_input_.assign(static_cast<size_t>(input_size_), 0.0f);
  stream_frame_index_ = 0;
}

float SileroVadRunner::process_frame(
    const float* audio_data,
    int64_t num_samples) {
  int64_t chunk_len = std::min(window_size_, num_samples);

  std::memcpy(
      stream_input_.data(),
      stream_context_.data(),
      static_cast<size_t>(context_size_) * sizeof(float));

  if (chunk_len > 0) {
    std::memcpy(
        stream_input_.data() + context_size_,
        audio_data,
        static_cast<size_t>(chunk_len) * sizeof(float));
  }
  if (chunk_len < window_size_) {
    std::memset(
        stream_input_.data() + context_size_ + chunk_len,
        0,
        static_cast<size_t>(window_size_ - chunk_len) * sizeof(float));
  }

  auto input_tensor = from_blob(
      stream_input_.data(),
      {1, static_cast<::executorch::aten::SizesType>(input_size_)},
      ::executorch::aten::ScalarType::Float);
  auto state_tensor = from_blob(
      stream_state_data_.data(),
      {2, 1, static_cast<::executorch::aten::SizesType>(kHiddenDim)},
      ::executorch::aten::ScalarType::Float);

  auto result = model_->execute(
      "forward", std::vector<EValue>{input_tensor, state_tensor});
  ET_CHECK_MSG(result.ok(), "Silero VAD forward failed.");

  auto& outputs = result.get();
  float prob = outputs[0].toTensor().const_data_ptr<float>()[0];

  auto new_state = outputs[1].toTensor();
  std::memcpy(
      stream_state_data_.data(),
      new_state.const_data_ptr<float>(),
      static_cast<size_t>(2 * kHiddenDim) * sizeof(float));

  if (chunk_len >= context_size_) {
    std::memcpy(
        stream_context_.data(),
        audio_data + chunk_len - context_size_,
        static_cast<size_t>(context_size_) * sizeof(float));
  } else if (chunk_len > 0) {
    int64_t keep = context_size_ - chunk_len;
    std::memmove(
        stream_context_.data(),
        stream_context_.data() + chunk_len,
        static_cast<size_t>(keep) * sizeof(float));
    std::memcpy(
        stream_context_.data() + keep,
        audio_data,
        static_cast<size_t>(chunk_len) * sizeof(float));
  }

  stream_frame_index_++;
  return prob;
}

SileroVadRunner::Result SileroVadRunner::detect(
    const float* audio_data,
    int64_t num_samples,
    float threshold,
    SegmentCallback segment_cb) {
  reset_stream();

  bool speech_active = false;
  int64_t speech_start_frame = 0;
  int64_t num_frames = 0;
  int64_t speech_frames = 0;
  int num_segments = 0;

  for (int64_t offset = 0; offset < num_samples; offset += window_size_) {
    float prob = process_frame(audio_data + offset, num_samples - offset);

    // Threshold-based speech detection
    if (prob > threshold) {
      speech_frames++;
      if (!speech_active) {
        speech_active = true;
        speech_start_frame = num_frames;
      }
    } else if (speech_active) {
      speech_active = false;
      segment_cb(
          {speech_start_frame * frame_duration_, num_frames * frame_duration_});
      num_segments++;
    }

    num_frames++;
  }

  // Close any still-active segment
  if (speech_active) {
    segment_cb(
        {speech_start_frame * frame_duration_, num_frames * frame_duration_});
    num_segments++;
  }

  return {num_frames, num_segments, speech_frames};
}

} // namespace silero_vad
