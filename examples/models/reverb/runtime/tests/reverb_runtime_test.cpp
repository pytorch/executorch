/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Tests the runner's pure helpers. Expected values come from the reference
// implementation (the reverb-tts Python package and torchaudio).

#include "reverb_audio.h"
#include "reverb_sampler.h"
#include "reverb_text.h"
#include "reverb_voice.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
  if (!condition) {
    std::fprintf(stderr, "FAILED: %s\n", what.c_str());
    ++failures;
  }
}

template <typename T>
void check_eq(const T& actual, const T& expected, const std::string& what) {
  check(actual == expected, what);
}

void check_near(
    double actual,
    double expected,
    double tolerance,
    const std::string& what) {
  check(
      std::fabs(actual - expected) <= tolerance,
      what + ": got " + std::to_string(actual));
}

std::vector<float> tone(double seconds, double amplitude, int rate) {
  std::vector<float> out(static_cast<size_t>(seconds * rate));
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] =
        static_cast<float>(amplitude * std::sin(2.0 * M_PI * 220.0 * i / rate));
  }
  return out;
}

std::vector<float> silence(double seconds, int rate) {
  return std::vector<float>(static_cast<size_t>(seconds * rate), 0.0f);
}

std::vector<float> concat(const std::vector<std::vector<float>>& parts) {
  std::vector<float> out;
  for (const auto& part : parts) {
    out.insert(out.end(), part.begin(), part.end());
  }
  return out;
}

void test_text() {
  const std::string reference = "Guten Morgen! Ach, wie schön, dich zu sehen!";
  check_eq(
      reverb::estimate_frames("Hi.", reference, 120),
      27,
      "short text estimate");
  check_eq(
      reverb::estimate_frames(
          "Hello there, this is a test of twenty-one words.", reference, 120),
      132,
      "sentence estimate");
  check_eq(
      reverb::estimate_frames("Zażółć gęślą jaźń — 42 razy?", reference, 120),
      92,
      "Polish letters, digits and a dash");
  check_eq(
      reverb::estimate_frames("été €5", reference, 120),
      39,
      "combining marks and a currency sign");

  std::string text;
  for (int i = 0; i < 30; ++i) {
    text += "One sentence here. ";
  }
  text += "The end.";
  const auto pieces = reverb::split_text(text, reference, 120, 750, 375);
  const std::vector<size_t> expected = {132, 132, 132, 132, 46};
  check_eq(pieces.size(), expected.size(), "piece count");
  for (size_t i = 0; i < std::min(pieces.size(), expected.size()); ++i) {
    check_eq(
        pieces[i].size(),
        expected[i],
        "piece " + std::to_string(i) + " length");
  }
  check_eq(
      reverb::split_text("Short.", reference, 120, 750, 375).size(),
      size_t{1},
      "short text stays whole");

  check(reverb::is_supported_language("pl"), "pl is supported");
  check(!reverb::is_supported_language("ja"), "ja is not supported");
  check_eq(
      reverb::build_prompt("Hi.", "Ref.", "en"),
      std::string(
          "<|denoise|><|lang_start|>en<|lang_end|><|instruct_start|>None"
          "<|instruct_end|><|text_start|>Ref. Hi.<|text_end|>"),
      "prompt format");
}

void test_schedule() {
  reverb::SamplerConfig config;
  const auto schedule = reverb::commit_schedule(config);
  const std::vector<int> expected = {3, 4, 5, 7, 11, 18, 37, 118};
  for (int s = 0; s < config.steps; ++s) {
    const int quota = static_cast<int>(std::ceil(
        2.0 * 100 * (static_cast<double>(schedule[s + 1]) - schedule[s])));
    check_eq(quota, expected[s], "quota of pass " + std::to_string(s));
  }
}

void test_main_passes() {
  reverb::SamplerConfig config;
  const int frames = 100;
  const int slots = reverb::kMainCodebooks * frames;
  const auto schedule = reverb::commit_schedule(config);
  std::vector<int64_t> tokens(reverb::kCodebooks * frames, reverb::kMask);
  std::vector<float> confidence(slots), held(slots);
  std::vector<int64_t> prediction(slots);
  reverb::Noise noise(7);
  int64_t previous = 0;
  for (int step = 0; step < config.steps; ++step) {
    for (int i = 0; i < slots; ++i) {
      confidence[i] = -0.01f * ((i * 37 + step) % 101);
      held[i] = -0.02f * ((i * 53 + step) % 97);
      prediction[i] = (i * 31 + step) % 1024;
    }
    reverb::apply_main_pass(
        tokens,
        frames,
        {confidence.data(), prediction.data(), held.data()},
        step,
        config,
        schedule,
        noise);
    int64_t revealed = 0;
    for (int i = 0; i < slots; ++i) {
      revealed += tokens[i] != reverb::kMask;
    }
    check(
        revealed >= previous,
        "pass " + std::to_string(step) + " reveals more than it withdraws");
    previous = revealed;
  }
  check_eq(
      previous,
      static_cast<int64_t>(slots),
      "every slot of codebooks 0-1 is decided");
  bool fine_masked = true;
  for (size_t i = slots; i < tokens.size(); ++i) {
    fine_masked &= tokens[i] == reverb::kMask;
  }
  check(fine_masked, "codebooks 2-7 stay masked for polish");

  // Same seed, same take.
  std::vector<int64_t> a(reverb::kCodebooks * frames, reverb::kMask), b = a;
  reverb::Noise na(3), nb(3);
  reverb::apply_main_pass(
      a,
      frames,
      {confidence.data(), prediction.data(), held.data()},
      0,
      config,
      schedule,
      na);
  reverb::apply_main_pass(
      b,
      frames,
      {confidence.data(), prediction.data(), held.data()},
      0,
      config,
      schedule,
      nb);
  check(a == b, "the sampler is deterministic for a seed");
}

void test_audio() {
  const int rate = 24000;
  const auto x = concat(
      {silence(0.5, rate),
       tone(1.0, 0.3, rate),
       silence(0.4, rate),
       tone(0.08, 0.5, rate),
       silence(0.6, rate)});
  check_eq(
      reverb::trim_silence(x, rate).size(),
      size_t{28800},
      "trim drops the edge burst");
  const auto y = concat(
      {silence(0.3, rate),
       tone(0.5, 0.3, rate),
       silence(1.5, rate),
       tone(0.7, 0.3, rate),
       silence(0.9, rate)});
  check_near(
      reverb::speech_length(y, rate), 2.0, 1e-9, "speech length caps pauses");

  std::vector<float> wave(441);
  for (size_t i = 0; i < wave.size(); ++i) {
    wave[i] = static_cast<float>(std::sin(i * 0.05));
  }
  const auto resampled = reverb::resample(wave, 44100, 24000);
  check_eq(resampled.size(), size_t{240}, "resampled length");
  const std::vector<std::pair<int, double>> expected = {
      {0, 0.007581},
      {7, 0.599186},
      {100, 0.235058},
      {200, -0.456936},
      {239, 0.028474}};
  for (const auto& [index, value] : expected) {
    check_near(
        resampled[index],
        value,
        2e-6,
        "resampled sample " + std::to_string(index));
  }

  std::vector<float> quiet(24000, 0.001f);
  quiet[10] = 0.5f;
  reverb::raise_quiet(quiet);
  check_near(quiet[10], 0.99, 1e-6, "raise stops at the peak");

  const std::string path = "reverb_runtime_test.wav";
  reverb::write_wav(path, tone(0.1, 0.5, rate), rate);
  const auto audio = reverb::read_wav(path);
  check_eq(audio.sample_rate, rate, "WAV sample rate round trip");
  check_eq(audio.samples.size(), size_t{2400}, "WAV length round trip");
  check_near(
      audio.samples[100],
      tone(0.1, 0.5, rate)[100],
      1.0 / 32767,
      "WAV sample round trip");
  std::remove(path.c_str());
}

void test_voice() {
  const std::string path = "reverb_runtime_test_voice.json";
  {
    std::ofstream out(path);
    out << R"({"codes": [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]],)"
        << R"( "transcript": "Hi.", "language": "en", "speech_frames": 1})";
  }
  const auto voice = reverb::load_voice(path);
  check_eq(voice.frames(), 2, "nested codes load as [8, frames]");
  check_eq(voice.codes[2], int64_t{3}, "codes are row-major");
  check_eq(voice.pace_frames(false), 1, "pace follows speech frames");
  check_eq(voice.pace_frames(true), 2, "pace can follow the whole recording");
  reverb::save_voice(voice, path);
  const auto reloaded = reverb::load_voice(path);
  check(
      reloaded.codes == voice.codes && reloaded.transcript == "Hi.",
      "voice round trip");
  std::remove(path.c_str());
}

} // namespace

int main() {
  test_text();
  test_schedule();
  test_main_passes();
  test_audio();
  test_voice();
  if (failures) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("All Reverb runtime helper tests passed\n");
  return EXIT_SUCCESS;
}
