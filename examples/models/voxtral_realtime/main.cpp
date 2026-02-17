/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CLI entry point for the Voxtral Realtime transcriber.
//
// Loads a .pte model, a preprocessor .pte, and a Tekken tokenizer.
// Processes a WAV file or live microphone audio.
//
// Modes:
//   Default:     Offline transcription (full encoder, then decode)
//   --streaming: Streaming transcription from WAV file
//   --mic:       Live microphone transcription (reads raw f32le PCM from stdin)
//
// Mic usage (pipe from ffmpeg or any audio capture tool):
//   ffmpeg -f avfoundation -i ":0" -ar 16000 -ac 1 -f f32le -nostats -loglevel
//   error pipe:1 | \
//     ./voxtral_realtime_runner --mic ...

#include <csignal>
#include <cstdio>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/runtime/platform/log.h>

#include "voxtral_realtime_runner.h"

DEFINE_string(
    model_path,
    "model.pte",
    "Path to Voxtral Realtime model (.pte).");
DEFINE_string(tokenizer_path, "tekken.json", "Path to Tekken tokenizer file.");
DEFINE_string(preprocessor_path, "", "Path to mel preprocessor (.pte).");
DEFINE_string(audio_path, "", "Path to input audio file (.wav).");
DEFINE_double(temperature, 0.0, "Sampling temperature (0 = greedy).");
DEFINE_int32(max_new_tokens, 500, "Maximum number of tokens to generate.");
DEFINE_bool(streaming, false, "Use streaming transcription mode.");
DEFINE_bool(
    mic,
    false,
    "Live microphone mode: read raw 16kHz float32 PCM from stdin.");

namespace {
volatile sig_atomic_t g_interrupted = 0;
void sigint_handler(int) {
  g_interrupted = 1;
}
} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_audio_path.empty() && !FLAGS_mic) {
    ET_LOG(Error, "Provide --audio_path or --mic.");
    return 1;
  }

  if (!FLAGS_audio_path.empty() && FLAGS_mic) {
    ET_LOG(Error, "--mic and --audio_path are mutually exclusive.");
    return 1;
  }

  if (FLAGS_preprocessor_path.empty()) {
    ET_LOG(Error, "preprocessor_path flag must be provided.");
    return 1;
  }

  // Install signal handler early so Ctrl+C is caught during load/warmup.
  std::signal(SIGINT, sigint_handler);

  ::executorch::extension::llm::Stats stats;
  stats.model_load_start_ms = ::executorch::extension::llm::time_in_ms();

  voxtral_realtime::VoxtralRealtimeRunner runner(
      FLAGS_model_path, FLAGS_tokenizer_path, FLAGS_preprocessor_path);

  stats.model_load_end_ms = ::executorch::extension::llm::time_in_ms();
  stats.inference_start_ms = ::executorch::extension::llm::time_in_ms();

  voxtral_realtime::TranscribeConfig config;
  config.temperature = static_cast<float>(FLAGS_temperature);
  config.max_new_tokens = FLAGS_max_new_tokens;

  stats.num_prompt_tokens = 0;
  bool first_token = true;

  // Set to true for green-colored output.
  const bool use_color = false;

  auto token_cb = [&](const std::string& piece) {
    if (first_token) {
      stats.first_token_ms = ::executorch::extension::llm::time_in_ms();
      stats.prompt_eval_end_ms = stats.first_token_ms;
      first_token = false;
    }
    if (!piece.empty() && piece.front() == '[' && piece.back() == ']') {
      // Uncomment to print special tokens
      // ::executorch::extension::llm::safe_printf(piece.c_str());
    } else {
      if (use_color) {
        printf("\033[32m");
      }
      ::executorch::extension::llm::safe_printf(piece.c_str());
      if (use_color) {
        printf("\033[0m");
      }
    }
    fflush(stdout);
  };

  int num_generated;
  if (FLAGS_mic) {
    // Live microphone: read raw 16kHz float32 PCM from stdin.
    ET_CHECK_MSG(
        runner.is_streaming(),
        "Model was not exported with --streaming. Re-export with --streaming flag.");
    auto session = runner.create_streaming_session(config, token_cb);

    // Drain any audio that buffered in stdin during model loading/warmup.
    // Without this, piped audio (e.g., from ffmpeg) accumulates while the
    // model loads and gets processed in a burst, breaking real-time behavior.
#ifndef _WIN32
    {
      int fd = fileno(stdin);
      int flags = fcntl(fd, F_GETFL, 0);
      if (flags >= 0) {
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        char drain[4096];
        while (read(fd, drain, sizeof(drain)) > 0) {
        }
        clearerr(stdin);
        fcntl(fd, F_SETFL, flags);
      }
    }
#endif

    fprintf(stdout, "Listening (Ctrl+C to stop)...\n");
    fflush(stdout);

    // Read 80ms chunks (1280 samples * 4 bytes = 5120 bytes).
    // StreamingSession internally processes in 80ms steps.
    const size_t chunk_samples = 1280;
    const size_t chunk_bytes = chunk_samples * sizeof(float);
    std::vector<float> buf(chunk_samples);

    while (!g_interrupted) {
      size_t bytes_read = fread(buf.data(), 1, chunk_bytes, stdin);
      if (bytes_read == 0) {
        break;
      }
      // Discard trailing bytes not aligned to sizeof(float).
      size_t samples_read = bytes_read / sizeof(float);
      if (samples_read == 0) {
        continue;
      }
      if (samples_read < chunk_samples) {
        std::fill(buf.begin() + samples_read, buf.end(), 0.0f);
      }
      session->feed_audio(buf.data(), static_cast<int64_t>(samples_read));
    }
    num_generated = session->flush();
  } else if (FLAGS_streaming) {
    ET_CHECK_MSG(
        runner.is_streaming(),
        "Model was not exported with --streaming. Re-export with --streaming flag.");
    ET_LOG(Info, "Loading audio from: %s", FLAGS_audio_path.c_str());
    auto audio_data =
        ::executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
    auto session = runner.create_streaming_session(config, token_cb);

    const int64_t chunk_size = 1280;
    for (int64_t offset = 0; offset < static_cast<int64_t>(audio_data.size());
         offset += chunk_size) {
      int64_t n = std::min(
          chunk_size, static_cast<int64_t>(audio_data.size()) - offset);
      session->feed_audio(audio_data.data() + offset, n);
    }
    num_generated = session->flush();
  } else {
    ET_LOG(Info, "Loading audio from: %s", FLAGS_audio_path.c_str());
    auto audio_data =
        ::executorch::extension::llm::load_wav_audio_data(FLAGS_audio_path);
    num_generated = runner.transcribe(
        audio_data.data(),
        static_cast<int64_t>(audio_data.size()),
        config,
        token_cb);
  }

  printf("\n");

  stats.num_generated_tokens = num_generated;
  stats.inference_end_ms = ::executorch::extension::llm::time_in_ms();

  ::executorch::extension::llm::print_report(stats);

  return 0;
}
