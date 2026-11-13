/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/lfm3/speech_policy.h>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef LFM3_SPEECH_POLICY_STANDALONE
#include <gtest/gtest.h>
#endif

namespace {
using namespace example::speech;
using Frame = std::array<int32_t, 12>;

void check(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void signed_penalty() {
  std::vector<float> scores = {4, -2, 3};
  penalize(scores, {0, 1}, 2);
  check(scores == std::vector<float>({2, -4, 3}), "signed penalty");
  std::mt19937 rng(0);
  check(sample_scores(scores, {}, rng) == 2, "penalty changes greedy winner");
}

void dequantized_penalty() {
  const std::array<uint16_t, 3> logits = {96, 95, 0};
  auto scores = dequantize_logits(logits.data(), logits.size(), 0.5f, 100);
  check(scores == std::vector<float>({-2, -2.5f, -50}), "zero-point and scale");
  penalize(scores, {0}, 2);
  std::mt19937 rng(0);
  check(
      sample_scores(scores, {}, rng) == 1,
      "negative decoded logit is multiplied");
}

void greedy_and_suppression() {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  std::mt19937 rng(0);
  SamplingPolicy policy;
  policy.top_k = 1;
  policy.top_p = 0;
  policy.min_p = 1;
  check(
      sample_scores({neg_inf, -3, -2}, policy, rng) == 2, "greedy suppression");
  check(
      sample_scores({neg_inf, 0, 0}, policy, rng) == 1,
      "suppressed zero cannot win tie");
}

void temperature_and_top_k() {
  SamplingPolicy policy;
  policy.temperature = 0.5f;
  policy.top_k = 2;
  const auto weights = sampling_weights({1, 1, 0, -1}, policy);
  check(
      weights == std::vector<double>({1, 1, 0, 0}), "top-k cutoff keeps ties");
  policy.top_k = 0;
  const auto unfiltered = sampling_weights({0, -1}, policy);
  check(
      std::abs(unfiltered[1] - std::exp(-2.0)) < 1e-7,
      "temperature scales logits");
}

void top_p_and_min_p() {
  SamplingPolicy policy;
  policy.temperature = 1;
  policy.top_p = 0.79f;
  const std::vector<float> logits = {
      std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};
  auto weights = sampling_weights(logits, policy);
  check(
      weights[0] > 0 && weights[1] > 0 && weights[2] == 0 && weights[3] == 0,
      "top-p removes low cumulative mass");
  policy.min_p = 0.7f;
  weights = sampling_weights(logits, policy);
  check(weights == std::vector<double>({1, 0, 0, 0}), "min-p runs after top-p");
}

void categorical_boundaries() {
  const std::vector<double> weights = {0, 0.25, 0, 0.75, 0};
  check(categorical(weights, 0) == 1, "zero draw skips zero mass");
  check(categorical(weights, 0.25) == 3, "boundary skips exhausted mass");
  check(
      categorical(weights, std::nextafter(1.0, 2.0)) == 3,
      "rounding residual must not choose final EOS index");
  check(categorical({0, 0}, 0) == -1, "empty support is an error");
}

void bounded_history() {
  std::deque<Frame> history;
  Frame first{};
  first.fill(7);
  append_history(history, first, 64);
  for (int i = 0; i < 64; ++i) {
    Frame next{};
    next.fill(2);
    next[1] = 3;
    append_history(history, next, 64);
  }
  check(
      history.size() == 64 && history.front()[0] == 2,
      "HF audio window, not full history");
  std::unordered_set<int32_t> seen;
  for (const Frame& frame : history) {
    seen.insert(frame[0]);
  }
  std::vector<float> scores(8, 4);
  penalize(scores, seen, 2);
  check(
      scores[2] == 2 && scores[3] == 4 && scores[7] == 4,
      "per-codebook history only");
  append_history(history, first, 0);
  check(history.empty(), "zero window disables history");
  std::deque<int32_t> text;
  for (int32_t i = 0; i < 33; ++i) {
    append_history(text, i, 32);
  }
  check(text.size() == 32 && text.front() == 1, "HF generated-text window");
}

void eos_any_codebook() {
  Frame frame{};
  frame.fill(3);
  check(
      !normalize_audio_eos(frame, 8) && frame[0] == 3,
      "ordinary frame unchanged");
  for (size_t q = 0; q < frame.size(); ++q) {
    frame.fill(3);
    frame[q] = 8;
    check(
        normalize_audio_eos(frame, 8), "EOS in each codebook terminates frame");
    check(
        std::all_of(
            frame.begin(), frame.end(), [](int32_t code) { return code == 8; }),
        "EOS frame is canonicalized before embedding");
  }
}

void mode_transitions() {
  constexpr uint64_t text_end = 1, audio_start = 2, mixed_start = 3;
  ModeState state;
  check(
      !state.text_token(mixed_start, false, text_end, audio_start, mixed_start),
      "enter mixed");
  state.advance(false);
  for (int i = 0; i < 6; ++i) {
    check(!state.audio_slot(), "six text slots");
    // Markers inside a mixed block remain ordinary text.
    check(
        !state.text_token(
            audio_start, false, text_end, audio_start, mixed_start),
        "mixed marker");
    state.advance(true);
  }
  for (int i = 0; i < 12; ++i) {
    check(state.audio_slot(), "twelve audio slots");
    state.advance(true);
  }
  check(!state.audio_slot(), "back to mixed text");
  check(
      !state.text_token(text_end, true, text_end, audio_start, mixed_start),
      "text_end precedes eos in mixed");
  check(
      state.audio_slot() && state.text_done,
      "text_end starts unbounded audio tail");
  state.advance(true);
  check(state.audio_slot(), "audio tail does not toggle");
  // The generator assigns text mode after a canonical EOS frame.
  state.mode = Mode::kText;
  check(
      !state.text_token(audio_start, false, text_end, audio_start, mixed_start),
      "HF speech reentry allowed");
  check(state.audio_slot(), "reentered speech");
  state.mode = Mode::kText;
  check(
      state.text_token(text_end, false, text_end, audio_start, mixed_start),
      "text_end closes plain text");
  check(
      state.text_token(9, true, text_end, audio_start, mixed_start),
      "turn EOS closes any text slot");
}

void omitted_and_explicit_audio_overrides() {
  SamplingPolicy exported;
  exported.temperature = 0;
  exported.repetition_penalty = 2;
  exported.top_k = 7;
  auto kept = audio_overrides(exported, std::nullopt, std::nullopt);
  check(
      kept.temperature == 0 && kept.repetition_penalty == 2 && kept.top_k == 7,
      "omitted flags retain exported policy");
  auto changed = audio_overrides(exported, 0.4f, 1.2f);
  check(
      changed.temperature == 0.4f && changed.repetition_penalty == 1.2f,
      "explicit flags override even at CLI default values");
  changed = audio_overrides(exported, 0.0f, std::nullopt);
  check(
      changed.temperature == 0 && changed.repetition_penalty == 2,
      "explicit zero temperature is not omitted");
}

void prefill_handoff_and_reentry_history() {
  // Four prefill output rows, but only the first two belong to the prompt.
  const std::array<uint16_t, 20> prefill = {0,  99, 0, 0, 0, 0,  1, 2, 3, 80,
                                            99, 0,  0, 0, 0, 99, 0, 0, 0, 0};
  const auto* selected = last_logits_row(prefill.data(), 2, 5);
  check(
      selected == prefill.data() + 5,
      "partial final prefill selects last valid row");
  check(
      last_logits_row(prefill.data(), 0, 5) == nullptr,
      "no valid row rejected");
  SamplingPolicy policy;
  policy.repetition_penalty = 2;
  std::deque<int32_t> text;
  std::mt19937 rng(0);
  const std::array<uint64_t, 0> suppressed{};
  auto sample = [&](const uint16_t* logits) {
    return sample_text_logits(
        logits, 5, 0.5f, 50, policy, text, 32, suppressed, rng);
  };
  check(
      sample(selected) == 4 && text == std::deque<int32_t>({4}),
      "prefill token sampled and recorded once by decode policy");
  const std::array<uint16_t, 5> open = {60, 40, 90, 30, 80};
  ModeState state;
  const int32_t marker = sample(open.data());
  check(
      marker == 2 && !state.text_token(marker, false, 1, 2, 3),
      "text opens audio");
  Frame eos{};
  eos[11] = 8;
  check(normalize_audio_eos(eos, 8), "terminal speech frame");
  state.mode = Mode::kText;
  const std::array<uint16_t, 5> after_eos = {70, 40, 84, 30, 60};
  check(
      sample(after_eos.data()) == 0,
      "text penalty retains pre-EOS generated history");
  check(sample(open.data()) == 2, "model may reopen speech");
  check(
      !state.text_token(2, false, 1, 2, 3) && state.audio_slot(),
      "speech reentry");
  check(
      text == std::deque<int32_t>({4, 2, 0, 2}),
      "history not reset or double-appended");
}

void invalid_distributions() {
  std::mt19937 rng(0);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  check(sample_scores({}, {}, rng) == -1, "empty logits");
  check(sample_scores({nan, 1}, {}, rng) == -1, "nonfinite logits");
  check(
      sample_scores({-std::numeric_limits<float>::infinity()}, {}, rng) == -1,
      "all suppressed");
  SamplingPolicy policy;
  policy.repetition_penalty = 0;
  check(!policy.valid(), "zero penalty rejected");
  policy.repetition_penalty = 1;
  policy.temperature = -1;
  check(!policy.valid(), "negative temperature rejected");
}
} // namespace

#ifdef LFM3_SPEECH_POLICY_STANDALONE
int main() {
  try {
    signed_penalty();
    dequantized_penalty();
    greedy_and_suppression();
    temperature_and_top_k();
    top_p_and_min_p();
    categorical_boundaries();
    bounded_history();
    eos_any_codebook();
    mode_transitions();
    omitted_and_explicit_audio_overrides();
    prefill_handoff_and_reentry_history();
    invalid_distributions();
    std::cout << "12 speech policy cases passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
#else
TEST(SpeechPolicyTest, SignedPenalty) {
  signed_penalty();
}
TEST(SpeechPolicyTest, DequantizedPenalty) {
  dequantized_penalty();
}
TEST(SpeechPolicyTest, GreedyAndSuppression) {
  greedy_and_suppression();
}
TEST(SpeechPolicyTest, TemperatureAndTopK) {
  temperature_and_top_k();
}
TEST(SpeechPolicyTest, TopPAndMinP) {
  top_p_and_min_p();
}
TEST(SpeechPolicyTest, CategoricalBoundaries) {
  categorical_boundaries();
}
TEST(SpeechPolicyTest, BoundedHistory) {
  bounded_history();
}
TEST(SpeechPolicyTest, EosAnyCodebook) {
  eos_any_codebook();
}
TEST(SpeechPolicyTest, ModeTransitions) {
  mode_transitions();
}
TEST(SpeechPolicyTest, OmittedAndExplicitAudioOverrides) {
  omitted_and_explicit_audio_overrides();
}
TEST(SpeechPolicyTest, PrefillHandoffAndReentryHistory) {
  prefill_handoff_and_reentry_history();
}
TEST(SpeechPolicyTest, InvalidDistributions) {
  invalid_distributions();
}
#endif
