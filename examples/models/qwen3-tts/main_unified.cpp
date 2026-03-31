/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Generated with assistance from Claude.

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/runtime/platform/log.h>

#include "qwen3_tts_unified_runner.h"

DEFINE_string(
    model_path,
    "model.pte",
    "Path to unified qwen3-tts model (.pte).");
DEFINE_string(
    tokenizer_path,
    "",
    "Path to tokenizer.json (for text-to-audio mode).");
DEFINE_string(
    codes_path,
    "",
    "Path to pre-generated codec ids (.bin). "
    "If provided, runs decode-only mode.");
DEFINE_string(output_wav, "output.wav", "Path to output wav file.");
DEFINE_string(
    output_dir,
    "",
    "Optional directory for per-prompt wav outputs in --prompts_path mode.");
DEFINE_string(
    text,
    "",
    "Text for synthesis (requires --tokenizer_path).");
DEFINE_string(
    prompts_path,
    "",
    "Optional newline-delimited prompt file for warm multi-prompt benchmarking.");
DEFINE_string(language, "English", "Language for synthesis.");

DEFINE_int32(max_new_tokens, 200, "Max codec tokens to generate.");
DEFINE_double(temperature, 0.9, "Sampling temperature.");
DEFINE_int32(top_k, 50, "Top-k sampling.");
DEFINE_double(top_p, 1.0, "Top-p sampling. Values <= 0 disable nucleus filtering.");
DEFINE_double(repetition_penalty, 1.05, "Repetition penalty for talker code_0 sampling.");
DEFINE_int32(repeat, 1, "Repeat count for each prompt in --prompts_path mode.");
DEFINE_uint64(seed, 42, "Base RNG seed for text synthesis.");
DEFINE_bool(
    disable_fused_cp_generate,
    false,
    "Force the legacy host-side code predictor loop for validation.");
DEFINE_string(
    instruct,
    "",
    "VoiceDesign instruct text (e.g. 'A cheerful young female voice').");
DEFINE_bool(
    non_streaming_mode,
    false,
    "Disable chunk emission during generation and only decode final audio.");
DEFINE_double(
    streaming_interval,
    2.0,
    "Streaming emit interval in seconds (0 = disabled unless --streaming_chunk_steps is set).");
DEFINE_int32(
    streaming_chunk_steps,
    0,
    "Deprecated alias for the emit interval expressed in codec steps.");
DEFINE_int32(
    streaming_chunk_size,
    300,
    "Maximum codec steps per overlap-context decode window.");
DEFINE_int32(
    streaming_left_context_size,
    25,
    "Left-context codec steps preserved for overlap-context decode.");
DEFINE_bool(
    disable_streaming_decoder_surface,
    false,
    "Force the runner-side overlap decode path even when decode_audio_stream is available.");
DEFINE_bool(
    force_streaming_decoder_surface,
    false,
    "Override export metadata and force decode_audio_stream when it is available.");
DEFINE_bool(
    use_legacy_cumulative_streaming_decode,
    false,
    "For benchmarking only: re-decode the full accumulated prefix on each chunk.");
DEFINE_bool(
    trim_silence,
    true,
    "Trim leading silence from output audio.");
DEFINE_double(
    trim_threshold,
    0.005,
    "RMS threshold for silence trimming.");

namespace {

bool trim_leading_silence(
    std::vector<float>* waveform,
    int sample_rate,
    double threshold,
    double* trimmed_ms) {
  if (waveform == nullptr || waveform->empty()) {
    if (trimmed_ms != nullptr) {
      *trimmed_ms = 0.0;
    }
    return true;
  }
  size_t speech_start = 0;
  const float threshold_f = static_cast<float>(threshold);
  for (size_t i = 0; i < waveform->size(); ++i) {
    if (std::abs((*waveform)[i]) > threshold_f) {
      const size_t margin = static_cast<size_t>(0.05 * sample_rate);
      speech_start = (i > margin) ? i - margin : 0;
      break;
    }
  }
  if (trimmed_ms != nullptr) {
    *trimmed_ms = 1000.0 * static_cast<double>(speech_start) / sample_rate;
  }
  if (speech_start > 0) {
    waveform->erase(waveform->begin(), waveform->begin() + speech_start);
  }
  return true;
}

bool read_prompts_file(const std::string& prompts_path, std::vector<std::string>* prompts) {
  std::ifstream in(prompts_path);
  if (!in.good()) {
    ET_LOG(Error, "Could not open prompts file: %s", prompts_path.c_str());
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      prompts->push_back(line);
    }
  }
  if (prompts->empty()) {
    ET_LOG(Error, "No non-empty prompts found in: %s", prompts_path.c_str());
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!FLAGS_codes_path.empty() &&
      (!FLAGS_text.empty() || !FLAGS_prompts_path.empty())) {
    ET_LOG(
        Error,
        "Provide either --codes_path or text synthesis inputs, not both.");
    return 1;
  }
  if (!FLAGS_text.empty() && !FLAGS_prompts_path.empty()) {
    ET_LOG(Error, "Provide either --text or --prompts_path, not both.");
    return 1;
  }
  if (FLAGS_codes_path.empty() && FLAGS_text.empty() && FLAGS_prompts_path.empty()) {
    ET_LOG(Error, "Either --codes_path, --text, or --prompts_path must be provided.");
    return 1;
  }
  if ((!FLAGS_text.empty() || !FLAGS_prompts_path.empty()) &&
      FLAGS_tokenizer_path.empty()) {
    ET_LOG(Error, "Text synthesis requires --tokenizer_path.");
    return 1;
  }
  if (FLAGS_repeat <= 0) {
    ET_LOG(Error, "--repeat must be positive.");
    return 1;
  }

  const auto t_construct_start = std::chrono::steady_clock::now();
  qwen3_tts::Qwen3TTSUnifiedRunner runner(
      FLAGS_model_path, FLAGS_tokenizer_path);
  const auto t_construct_end = std::chrono::steady_clock::now();
  const double construct_ms =
      std::chrono::duration<double, std::milli>(
          t_construct_end - t_construct_start)
          .count();

  const auto t_warmup_start = std::chrono::steady_clock::now();
  if (!FLAGS_codes_path.empty()) {
    runner.warmup_decode();
  } else {
    runner.warmup_all();
  }
  const auto t_warmup_end = std::chrono::steady_clock::now();
  const double warmup_ms =
      std::chrono::duration<double, std::milli>(t_warmup_end - t_warmup_start)
          .count();
  ET_LOG(Info, "Runner construction: %.1f ms", construct_ms);
  ET_LOG(Info, "Warmup: %.1f ms", warmup_ms);

  std::vector<float> waveform;

  if (!FLAGS_codes_path.empty()) {
    // Decode-only mode: use precomputed codes.
    ET_LOG(Info, "Decode-only mode: %s", FLAGS_codes_path.c_str());
    auto t0 = std::chrono::steady_clock::now();
    if (!runner.decode_codes_file(FLAGS_codes_path, &waveform)) {
      ET_LOG(Error, "decode_codes_file failed.");
      return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double decode_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    double audio_sec =
        static_cast<double>(waveform.size()) / runner.output_sample_rate();
    ET_LOG(
        Info,
        "Decoded %zu samples (%.2fs audio) in %.1f ms (%.2fx realtime)",
        waveform.size(),
        audio_sec,
        decode_ms,
        audio_sec / (decode_ms / 1000.0));
  } else {
    std::vector<std::string> prompts;
    if (!FLAGS_text.empty()) {
      prompts.push_back(FLAGS_text);
    } else if (!read_prompts_file(FLAGS_prompts_path, &prompts)) {
      return 1;
    }

    qwen3_tts::SynthesizeConfig config;
    config.max_new_tokens = FLAGS_max_new_tokens;
    config.temperature = static_cast<float>(FLAGS_temperature);
    config.top_k = FLAGS_top_k;
    config.top_p = static_cast<float>(FLAGS_top_p);
    config.repetition_penalty = static_cast<float>(FLAGS_repetition_penalty);
    config.seed = FLAGS_seed;
    config.use_fused_cp_generate = !FLAGS_disable_fused_cp_generate;
    config.instruct = FLAGS_instruct;
    config.non_streaming_mode = FLAGS_non_streaming_mode;
    config.streaming_interval_sec = static_cast<float>(FLAGS_streaming_interval);
    config.streaming_chunk_steps = FLAGS_streaming_chunk_steps;
    config.streaming_chunk_size = FLAGS_streaming_chunk_size;
    config.streaming_left_context_size = FLAGS_streaming_left_context_size;
    config.disable_streaming_decoder_surface =
        FLAGS_disable_streaming_decoder_surface;
    config.force_streaming_decoder_surface = FLAGS_force_streaming_decoder_surface;
    config.use_legacy_cumulative_streaming_decode =
        FLAGS_use_legacy_cumulative_streaming_decode;

    if (!FLAGS_prompts_path.empty() && !FLAGS_disable_fused_cp_generate &&
        FLAGS_top_k <= 0) {
      config.top_k = 50;
      ET_LOG(
          Info,
          "Benchmark mode defaulting top_k to %d so cp_generate fast path is exercised.",
          config.top_k);
    }

    if (!FLAGS_output_dir.empty()) {
      std::filesystem::create_directories(FLAGS_output_dir);
    }

    for (int repeat_idx = 0; repeat_idx < FLAGS_repeat; ++repeat_idx) {
      for (size_t prompt_idx = 0; prompt_idx < prompts.size(); ++prompt_idx) {
        waveform.clear();
        qwen3_tts::SynthesizeConfig prompt_config = config;
        prompt_config.seed =
            FLAGS_seed + static_cast<uint64_t>(repeat_idx * prompts.size() + prompt_idx);
        auto session = runner.create_synthesis_session(prompt_config);
        qwen3_tts::SynthesisTiming timing;
        int streaming_chunks_received = 0;
        qwen3_tts::AudioChunkCallback stream_cb = nullptr;
        const bool streaming_enabled =
            !prompt_config.non_streaming_mode &&
            (prompt_config.streaming_chunk_steps > 0 ||
             prompt_config.streaming_interval_sec > 0.0f);
        if (streaming_enabled) {
          stream_cb = [&](const std::vector<float>& chunk, bool is_final) {
            ++streaming_chunks_received;
            double chunk_sec =
                static_cast<double>(chunk.size()) / runner.output_sample_rate();
            ET_LOG(
                Info,
                "Stream chunk %d: %.2fs (%zu samples)%s",
                streaming_chunks_received,
                chunk_sec,
                chunk.size(),
                is_final ? " [final]" : "");
          };
        }
        if (!session->synthesize(
                prompts[prompt_idx], FLAGS_language, &waveform, &timing,
                std::move(stream_cb))) {
          ET_LOG(
              Error,
              "Synthesis failed for prompt %zu repeat %d.",
              prompt_idx,
              repeat_idx);
          return 1;
        }

        const size_t raw_sample_count = waveform.size();
        const double raw_audio_sec =
            static_cast<double>(raw_sample_count) / runner.output_sample_rate();
        double postprocess_ms = 0.0;
        double trimmed_ms = 0.0;
        const auto t_postprocess = std::chrono::steady_clock::now();
        if (FLAGS_trim_silence) {
          trim_leading_silence(
              &waveform,
              runner.output_sample_rate(),
              FLAGS_trim_threshold,
              &trimmed_ms);
        }
        std::string output_path;
        const bool should_write_single =
            prompts.size() == 1 && FLAGS_prompts_path.empty() &&
            !FLAGS_output_wav.empty();
        const bool should_write_batch =
            prompts.size() > 1 && !FLAGS_output_dir.empty();
        if (should_write_single) {
          output_path = FLAGS_output_wav;
        } else if (should_write_batch) {
          output_path = FLAGS_output_dir + "/prompt_" +
              std::to_string(prompt_idx) + "_repeat_" +
              std::to_string(repeat_idx) + ".wav";
        }
        if (!output_path.empty() && !runner.write_wav_file(output_path, waveform)) {
          ET_LOG(Error, "Failed to write wav: %s", output_path.c_str());
          return 1;
        }
        postprocess_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_postprocess)
                .count();

        const double trimmed_audio_sec =
            static_cast<double>(waveform.size()) / runner.output_sample_rate();
        const double generation_sec = timing.total_generation_ms / 1000.0;
        const double raw_rtf =
            generation_sec > 0.0 ? raw_audio_sec / generation_sec : 0.0;
        const double trimmed_rtf =
            generation_sec > 0.0 ? trimmed_audio_sec / generation_sec : 0.0;
        ET_LOG(
            Info,
            "prompt=%zu repeat=%d tokens=%d steps=%d audio=%.2fs "
            "trimmed_audio=%.2fs "
            "prep=%.1fms prefill=%.1fms codegen=%.1fms first_audio=%.1fms "
            "chunk_decode=%.1fms final_decode=%.1fms decode=%.1fms "
            "generation=%.1fms post=%.1fms trimmed=%.1fms "
            "rtf=%.2fx rtf_trimmed=%.2fx",
            prompt_idx,
            repeat_idx,
            timing.prompt_token_count,
            timing.generated_codec_steps,
            raw_audio_sec,
            trimmed_audio_sec,
            timing.prompt_prep_ms,
            timing.talker_prefill_ms,
            timing.codegen_ms,
            timing.first_audio_ms,
            timing.chunk_decode_ms,
            timing.final_decode_ms,
            timing.decode_audio_ms,
            timing.total_generation_ms,
            postprocess_ms,
            trimmed_ms,
            raw_rtf,
            trimmed_rtf);
        if (!output_path.empty()) {
          ET_LOG(Info, "Wrote wav: %s", output_path.c_str());
        }
      }
    }
  }

  return 0;
}
