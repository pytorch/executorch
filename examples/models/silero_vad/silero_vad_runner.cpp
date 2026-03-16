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
}

SileroVadRunner::Result SileroVadRunner::detect(
    const float* audio_data,
    int64_t num_samples,
    float threshold,
    SegmentCallback segment_cb) {
  // LSTM state: (2, 1, 128) — [h, c]
  std::vector<float> state_data(static_cast<size_t>(2 * kHiddenDim), 0.0f);

  // Context: previous chunk's last context_size_ samples
  std::vector<float> context(static_cast<size_t>(context_size_), 0.0f);

  // Input buffer: [context | chunk] = input_size_ samples
  std::vector<float> input(static_cast<size_t>(input_size_));

  bool speech_active = false;
  int64_t speech_start_frame = 0;
  int64_t num_frames = 0;
  int64_t speech_frames = 0;
  int num_segments = 0;

  for (int64_t offset = 0; offset < num_samples; offset += window_size_) {
    int64_t chunk_len = std::min(window_size_, num_samples - offset);

    // Build input: [context | chunk]
    std::memcpy(
        input.data(),
        context.data(),
        static_cast<size_t>(context_size_) * sizeof(float));

    if (chunk_len == window_size_) {
      std::memcpy(
          input.data() + context_size_,
          audio_data + offset,
          static_cast<size_t>(window_size_) * sizeof(float));
    } else {
      // Pad the last partial chunk with zeros
      std::memcpy(
          input.data() + context_size_,
          audio_data + offset,
          static_cast<size_t>(chunk_len) * sizeof(float));
      std::memset(
          input.data() + context_size_ + chunk_len,
          0,
          static_cast<size_t>(window_size_ - chunk_len) * sizeof(float));
    }

    auto input_tensor = from_blob(
        input.data(),
        {1, static_cast<::executorch::aten::SizesType>(input_size_)},
        ::executorch::aten::ScalarType::Float);
    auto state_tensor = from_blob(
        state_data.data(),
        {2, 1, static_cast<::executorch::aten::SizesType>(kHiddenDim)},
        ::executorch::aten::ScalarType::Float);

    auto result = model_->execute(
        "forward", std::vector<EValue>{input_tensor, state_tensor});
    if (!result.ok()) {
      ET_LOG(
          Error,
          "forward failed at offset %lld.",
          static_cast<long long>(offset));
      break;
    }

    auto& outputs = result.get();
    float prob = outputs[0].toTensor().const_data_ptr<float>()[0];

    // Update LSTM state
    auto new_state = outputs[1].toTensor();
    std::memcpy(
        state_data.data(),
        new_state.const_data_ptr<float>(),
        static_cast<size_t>(2 * kHiddenDim) * sizeof(float));

    // Update context from current chunk
    if (chunk_len >= context_size_) {
      std::memcpy(
          context.data(),
          audio_data + offset + chunk_len - context_size_,
          static_cast<size_t>(context_size_) * sizeof(float));
    } else {
      // Shift existing context and append partial chunk
      int64_t keep = context_size_ - chunk_len;
      std::memmove(
          context.data(),
          context.data() + chunk_len,
          static_cast<size_t>(keep) * sizeof(float));
      std::memcpy(
          context.data() + keep,
          audio_data + offset,
          static_cast<size_t>(chunk_len) * sizeof(float));
    }

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
