/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace reverb {

constexpr int kFrameRate = 25; // codec frames per second

// A reference voice: its codec codes and the transcript of the recording.
struct Voice {
  std::vector<int64_t> codes; // [8, frames] row-major, each below 1024
  std::string transcript; // already prepared (verbalized) text
  std::string language;
  std::optional<int> speech_frames; // frames spent speaking, for pacing

  int frames() const {
    return static_cast<int>(codes.size() / 8);
  }
  // The reference length the target length is estimated from.
  int pace_frames(bool whole_recording) const;
};

// Reads a voice JSON file: {"codes", "transcript", "language"} with an
// optional "speech_frames". Codes are 8 x frames integers, flat row-major or
// nested. The preset voices of the Reverb-TTS release use this format.
Voice load_voice(const std::string& path);

void save_voice(const Voice& voice, const std::string& path);

} // namespace reverb
