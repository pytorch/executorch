/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace supertonic {

struct WavLayout {
  uint32_t data_bytes;
  uint32_t riff_size;
  uint16_t block_align;
  uint32_t byte_rate;
};

WavLayout
validate_wav_layout(size_t sample_count, int sample_rate, int channels);

bool write_pcm16_wav(
    const std::string& path,
    const std::vector<float>& samples,
    int sample_rate,
    int channels = 1);

} // namespace supertonic
