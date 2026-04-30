/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

namespace executorch::examples::gemma4 {

/**
 * Performance instrumentation for the Gemma 4 pipeline.
 *
 * Tracks wall-clock time for each stage:
 *   - Model loading (PTE + tokenizer)
 *   - Speech transform (waveform -> mel spectrogram)
 *   - Audio encoding (mel -> audio embeddings)
 *   - Prefill (first forward pass through text decoder)
 *   - Token generation (autoregressive decode loop)
 */
struct Gemma4Stats {
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  double load_ms = 0;
  double speech_transform_ms = 0;
  double audio_encode_ms = 0;
  double vision_encode_ms = 0;
  double prefill_ms = 0;
  double generation_ms = 0;

  int32_t num_prompt_tokens = 0;
  int32_t num_generated_tokens = 0;
  double audio_duration_ms = 0;

  int64_t rss_before_load_kb = 0;
  int64_t rss_after_load_kb = 0;
  int64_t rss_before_gen_kb = 0;
  int64_t rss_peak_gen_kb = 0;
  int64_t rss_after_gen_kb = 0;

  void on_load_begin() {
    load_begin_ = Clock::now();
    rss_before_load_kb = read_rss_kb();
  }
  void on_load_end() {
    load_ms = elapsed_ms(load_begin_);
    rss_after_load_kb = read_rss_kb();
  }

  void on_speech_transform_begin() {
    speech_transform_begin_ = Clock::now();
  }
  void on_speech_transform_end() {
    speech_transform_ms = elapsed_ms(speech_transform_begin_);
  }

  void on_audio_encode_begin() {
    audio_encode_begin_ = Clock::now();
  }
  void on_audio_encode_end() {
    audio_encode_ms = elapsed_ms(audio_encode_begin_);
  }

  void on_vision_encode_begin() {
    vision_encode_begin_ = Clock::now();
  }
  void on_vision_encode_end() {
    vision_encode_ms = elapsed_ms(vision_encode_begin_);
  }

  void on_prefill_begin() {
    prefill_begin_ = Clock::now();
  }
  void on_prefill_end() {
    prefill_ms = elapsed_ms(prefill_begin_);
  }

  void on_generation_begin() {
    generation_begin_ = Clock::now();
  }
  void on_generation_end() {
    generation_ms = elapsed_ms(generation_begin_);
  }

  double speech_encoder_ms() const {
    return speech_transform_ms + audio_encode_ms;
  }

  double total_inference_ms() const {
    return speech_transform_ms + audio_encode_ms + vision_encode_ms +
        prefill_ms + generation_ms;
  }

  double tokens_per_second() const {
    if (generation_ms <= 0 || num_generated_tokens <= 0) {
      return 0;
    }
    return num_generated_tokens / (generation_ms / 1000.0);
  }

  double time_to_first_token_ms() const {
    return speech_transform_ms + audio_encode_ms + vision_encode_ms +
        prefill_ms;
  }

  double prefill_tokens_per_second() const {
    if (prefill_ms <= 0 || num_prompt_tokens <= 0) {
      return 0;
    }
    return num_prompt_tokens / (prefill_ms / 1000.0);
  }

  /// Real-time factor: total_inference / audio_duration. <1.0 = faster than
  /// real-time.
  double rtf() const {
    if (audio_duration_ms <= 0) {
      return 0;
    }
    return total_inference_ms() / audio_duration_ms;
  }

  void reset() {
    load_ms = 0;
    speech_transform_ms = 0;
    audio_encode_ms = 0;
    vision_encode_ms = 0;
    prefill_ms = 0;
    generation_ms = 0;
    num_prompt_tokens = 0;
    num_generated_tokens = 0;
    audio_duration_ms = 0;
    rss_before_load_kb = 0;
    rss_after_load_kb = 0;
    rss_before_gen_kb = 0;
    rss_peak_gen_kb = 0;
    rss_after_gen_kb = 0;
  }

  std::string report() const {
    std::string s;
    s += "=== Gemma 4 Performance Report ===\n";
    if (load_ms > 0) {
      s += "  Model load:        " + fmt_ms(load_ms) + "\n";
    }
    if (speech_transform_ms > 0) {
      s += "  Speech transform:  " + fmt_ms(speech_transform_ms) + "\n";
    }
    if (audio_encode_ms > 0) {
      s += "  Audio encode:      " + fmt_ms(audio_encode_ms) + "\n";
    }
    if (vision_encode_ms > 0) {
      s += "  Vision encode:     " + fmt_ms(vision_encode_ms) + "\n";
    }
    if (speech_encoder_ms() > 0) {
      s += "  Speech encoder:    " + fmt_ms(speech_encoder_ms()) + " (total)\n";
    }
    s += "  Prefill:           " + fmt_ms(prefill_ms);
    if (num_prompt_tokens > 0) {
      s += " (" + std::to_string(num_prompt_tokens) + " tokens, " +
          fmt_f(prefill_tokens_per_second(), 0) + " tok/s)";
    }
    s += "\n";
    s += "  Generation:        " + fmt_ms(generation_ms);
    if (num_generated_tokens > 0) {
      s += " (" + std::to_string(num_generated_tokens) + " tokens, " +
          fmt_f(tokens_per_second(), 0) + " tok/s)";
    }
    s += "\n";
    s += "  TTFT:              " + fmt_ms(time_to_first_token_ms()) + "\n";
    s += "  Total:             " + fmt_ms(total_inference_ms()) + "\n";
    if (rss_after_load_kb > 0) {
      s += "  Memory (load):     " + fmt_f(rss_after_load_kb / 1024.0, 0) +
          " MB\n";
    }
    if (rss_peak_gen_kb > 0) {
      s += "  Memory (peak):     " + fmt_f(rss_peak_gen_kb / 1024.0, 0) +
          " MB\n";
    }
    return s;
  }

  std::string to_json() const {
    // @lint-ignore CLANGTIDY facebook-hte-CArray
    char buf[1024];
    snprintf(
        buf,
        sizeof(buf),
        "{\"load_ms\":%.1f,\"speech_transform_ms\":%.1f,"
        "\"audio_encode_ms\":%.1f,\"vision_encode_ms\":%.1f,"
        "\"prefill_ms\":%.1f,\"generation_ms\":%.1f,"
        "\"num_prompt_tokens\":%d,\"num_generated_tokens\":%d,"
        "\"prefill_tok_per_s\":%.1f,\"gen_tok_per_s\":%.1f,"
        "\"ttft_ms\":%.1f,\"total_ms\":%.1f,"
        "\"rss_after_load_mb\":%.0f,\"rss_peak_gen_mb\":%.0f}",
        load_ms,
        speech_transform_ms,
        audio_encode_ms,
        vision_encode_ms,
        prefill_ms,
        generation_ms,
        num_prompt_tokens,
        num_generated_tokens,
        prefill_tokens_per_second(),
        tokens_per_second(),
        time_to_first_token_ms(),
        total_inference_ms(),
        rss_after_load_kb / 1024.0,
        rss_peak_gen_kb / 1024.0);
    return std::string(buf);
  }

  static int64_t read_rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) {
      return 0;
    }
    // @lint-ignore CLANGTIDY facebook-hte-CArray
    char line[256];
    int64_t rss_kb = 0;
    while (fgets(line, sizeof(line), f)) {
      if (sscanf(line, "VmRSS: %ld kB", &rss_kb) == 1) {
        break;
      }
    }
    fclose(f);
    return rss_kb;
  }

 private:
  TimePoint load_begin_;
  TimePoint speech_transform_begin_;
  TimePoint audio_encode_begin_;
  TimePoint vision_encode_begin_;
  TimePoint prefill_begin_;
  TimePoint generation_begin_;

  static double elapsed_ms(TimePoint start) {
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

  static std::string fmt_ms(double ms) {
    if (ms >= 1000.0) {
      return fmt_f(ms / 1000.0, 2) + " s";
    }
    return fmt_f(ms, 1) + " ms";
  }

  static std::string fmt_f(double val, int precision) {
    // @lint-ignore CLANGTIDY facebook-hte-CArray
    char buf[64];
    snprintf(buf, sizeof(buf), "%.*f", precision, val);
    return std::string(buf);
  }
};

} // namespace executorch::examples::gemma4
