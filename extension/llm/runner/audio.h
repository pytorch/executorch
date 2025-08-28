/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple audio struct.

#pragma once
#include <executorch/runtime/platform/compiler.h>
#include <cstdint>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {

/**
 * Audio inputs as a raw audio tensor, for use when the audio processing
 * into a mel spectrogram is baked into the audio encoder with torch.export.
 */
struct ET_EXPERIMENTAL RawAudio {
  std::vector<uint8_t> data;
  int32_t batch_size;
  int32_t n_channels; // For mono, use n_channels = 1.
  int32_t n_samples;
};

/**
 * Pre-processed audio inputs, ready to feed directly into an audio
 * encoder.
 */
struct ET_EXPERIMENTAL Audio {
  std::vector<uint8_t> data;
  int32_t batch_size;
  int32_t n_bins;
  int32_t n_frames;
};

} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::Audio;
} // namespace executor
} // namespace torch
