/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "reverb_sampler.h"
#include "reverb_voice.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace reverb {

struct GenerateOptions {
  std::string text; // already prepared (verbalized) text
  std::string language = "en";
  uint64_t seed = 42; // consecutive seeds go to consecutive pieces
  float speed = 1.0f; // a factor on the estimated length
  float duration = 0.0f; // seconds; 0 estimates it from the reference's pace
  bool pace_whole_recording = false; // pace on every frame, pauses included
  bool trim = true; // cut the take's silent edges
  SamplerConfig sampler;
};

struct Limits {
  int64_t prompt_tokens = 0;
  int64_t reference_frames = 0;
  int64_t target_frames = 0;
  int64_t min_encode_frames = 0;
  int64_t max_encode_frames = 0;
  int64_t sample_rate = 0;
  int64_t hop_length = 0;
};

// Seconds spent in each program call during the last generate().
struct Timings {
  double prefill = 0.0;
  double steps = 0.0;
  double polish = 0.0;
  double decode = 0.0;
};

class ReverbRunner {
 public:
  // `backbone_path` and `codec_path` are the reverb.pte and codec.pte files
  // written by export_reverb; `tokenizer_path` is the release's tokenizer.json.
  ReverbRunner(
      const std::string& backbone_path,
      const std::string& codec_path,
      const std::string& tokenizer_path);
  ~ReverbRunner();

  ReverbRunner(const ReverbRunner&) = delete;
  ReverbRunner& operator=(const ReverbRunner&) = delete;

  const Limits& limits() const;

  // Loads every method, so the first take is not slowed by loading.
  void load(bool with_encoder);

  // A voice from a recording of one speaker (3 to 15 s of clean speech) and
  // its exact, prepared transcript.
  Voice clone_voice(
      const std::string& wav_path,
      const std::string& transcript,
      const std::string& language);

  // Speaks `options.text` in `voice`. Returns mono samples at the codec's
  // sample rate. A text estimated over 30 s is spoken in pieces of about 15 s.
  std::vector<float> generate(
      const GenerateOptions& options,
      const Voice& voice);

  const Timings& last_timings() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace reverb
