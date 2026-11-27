/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "reverb_audio.h"
#include "reverb_runner.h"
#include "reverb_text.h"
#include "reverb_voice.h"

#include <gflags/gflags.h>

#include <chrono>
#include <cstdio>
#include <exception>
#include <string>

DEFINE_string(backbone, "", "reverb.pte written by export_reverb.");
DEFINE_string(codec, "", "codec.pte written by export_reverb.");
DEFINE_string(tokenizer, "", "tokenizer.json of the Reverb-TTS release.");
DEFINE_string(
    voice,
    "",
    "Voice JSON: a preset from the release's voices/ folder, or one saved with --save_voice.");
DEFINE_string(
    reference_audio,
    "",
    "WAV recording to clone a voice from, in place of --voice.");
DEFINE_string(
    reference_text,
    "",
    "Exact, prepared transcript of --reference_audio.");
DEFINE_string(
    reference_language,
    "",
    "Language of --reference_audio; defaults to --language.");
DEFINE_string(
    save_voice,
    "",
    "Writes the cloned voice to this JSON file for reuse with --voice.");
DEFINE_string(
    text,
    "",
    "Prepared text to speak: numbers, dates and symbols written out as words.");
DEFINE_string(
    language,
    "en",
    "Language of --text: de, en, es, fr, it, pl or pt.");
DEFINE_string(output, "reverb.wav", "Output WAV path.");
DEFINE_uint64(
    seed,
    42,
    "Seed of the take; consecutive seeds go to consecutive pieces.");
DEFINE_double(
    speed,
    1.0,
    "Factor on the estimated length: above 1 speaks faster.");
DEFINE_double(duration, 0.0, "Length of the take in seconds; 0 estimates it.");
DEFINE_bool(
    pace_whole_recording,
    false,
    "Estimate the length from the whole reference, pauses included.");
DEFINE_bool(trim, true, "Cut the take's silent edges.");
DEFINE_int32(steps, 8, "Main passes, which decide codebooks 0-1.");
DEFINE_bool(
    warmup,
    false,
    "Run one discarded take first, so the timing excludes first-run setup.");

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Reverb-TTS on ExecuTorch: voice-cloning text to speech.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  const bool cloning = !FLAGS_reference_audio.empty();
  if (FLAGS_backbone.empty() || FLAGS_codec.empty() ||
      FLAGS_tokenizer.empty() || (FLAGS_voice.empty() == !cloning) ||
      (FLAGS_text.empty() && FLAGS_save_voice.empty())) {
    std::fprintf(
        stderr,
        "Required: --backbone, --codec, --tokenizer, one of --voice or --reference_audio, and --text.\n");
    return 2;
  }
  if (cloning && FLAGS_reference_text.empty()) {
    std::fprintf(stderr, "--reference_audio needs --reference_text.\n");
    return 2;
  }

  try {
    using Clock = std::chrono::steady_clock;
    const auto seconds = [](Clock::time_point since) {
      return std::chrono::duration<double>(Clock::now() - since).count();
    };

    auto start = Clock::now();
    reverb::ReverbRunner runner(FLAGS_backbone, FLAGS_codec, FLAGS_tokenizer);
    runner.load(cloning);
    std::fprintf(stderr, "Loaded in %.2f s\n", seconds(start));

    reverb::Voice voice;
    if (cloning) {
      start = Clock::now();
      const std::string language = FLAGS_reference_language.empty()
          ? FLAGS_language
          : FLAGS_reference_language;
      voice = runner.clone_voice(
          FLAGS_reference_audio, FLAGS_reference_text, language);
      std::fprintf(
          stderr,
          "Encoded a %.2f s reference in %.2f s\n",
          voice.frames() / 25.0,
          seconds(start));
      if (!FLAGS_save_voice.empty()) {
        reverb::save_voice(voice, FLAGS_save_voice);
        std::fprintf(
            stderr, "Saved the voice to %s\n", FLAGS_save_voice.c_str());
      }
    } else {
      voice = reverb::load_voice(FLAGS_voice);
    }
    if (FLAGS_text.empty()) {
      return 0;
    }

    reverb::GenerateOptions options;
    options.text = FLAGS_text;
    options.language = FLAGS_language;
    options.seed = FLAGS_seed;
    options.speed = static_cast<float>(FLAGS_speed);
    options.duration = static_cast<float>(FLAGS_duration);
    options.pace_whole_recording = FLAGS_pace_whole_recording;
    options.trim = FLAGS_trim;
    options.sampler.steps = FLAGS_steps;

    if (FLAGS_warmup) {
      start = Clock::now();
      reverb::GenerateOptions warmup = options;
      warmup.text = "Warming up.";
      runner.generate(warmup, voice);
      std::fprintf(stderr, "Warmed up in %.2f s\n", seconds(start));
    }

    start = Clock::now();
    const std::vector<float> audio = runner.generate(options, voice);
    const double elapsed = seconds(start);
    const int sample_rate = static_cast<int>(runner.limits().sample_rate);
    reverb::write_wav(FLAGS_output, audio, sample_rate);
    const double audio_seconds =
        static_cast<double>(audio.size()) / sample_rate;
    std::fprintf(
        stderr,
        "Wrote %s: %.2f s of audio in %.2f s (%.1fx real time)\n",
        FLAGS_output.c_str(),
        audio_seconds,
        elapsed,
        elapsed > 0 ? audio_seconds / elapsed : 0.0);
    const reverb::Timings& timings = runner.last_timings();
    std::fprintf(
        stderr,
        "  prefill %.3f s, main passes %.3f s, polish %.3f s, codec decode %.3f s\n",
        timings.prefill,
        timings.steps,
        timings.polish,
        timings.decode);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Error: %s\n", error.what());
    return 1;
  }
  return 0;
}
