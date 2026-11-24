/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

namespace reverb {

struct Audio {
  std::vector<float> samples; // mono, in [-1, 1]
  int sample_rate = 0;
};

// Reads a PCM (8, 16, 24 or 32 bit) or float WAV file, averaging channels.
Audio read_wav(const std::string& path);

// Writes mono samples as a 16-bit PCM WAV file.
void write_wav(
    const std::string& path,
    const std::vector<float>& samples,
    int sample_rate);

// Band-limited sinc resampling with a Kaiser window, matching
// torchaudio.functional.resample(..., resampling_method="sinc_interp_kaiser")
// with the parameters of librosa's kaiser_best.
std::vector<float>
resample(const std::vector<float>& samples, int from_rate, int to_rate);

// Raises a recording quieter than -40 dBFS RMS to -20 dBFS, keeping its peak
// at most 0.99. Quiet references make quiet takes that trimming then cuts.
void raise_quiet(std::vector<float>& samples);

// Cuts leading and trailing silence (10 ms windows under -50 dBFS), keeping
// `keep` seconds of it on each side. A short burst alone at either edge,
// separated by over 200 ms of silence, counts as silence.
std::vector<float> trim_silence(
    const std::vector<float>& samples,
    int sample_rate,
    float keep = 0.1f);

// Seconds of a reference recording that set the speaker's pace: without its
// silent edges and with every pause counted as at most `max_pause` seconds.
double speech_length(
    const std::vector<float>& samples,
    int sample_rate,
    double max_pause = 0.5,
    double lead = 0.1,
    double trail = 0.2);

} // namespace reverb
