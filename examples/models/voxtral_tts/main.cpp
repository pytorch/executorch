/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Voxtral TTS runner CLI.
 *
 * Usage:
 *   voxtral_tts_runner --model model.pte --codec codec_decoder.pte \
 *       --tokenizer tekken.json --text "Hello world" --output output.wav
 */

#include "voxtral_tts_runner.h"

#include <chrono>
#include <csignal>
#include <iostream>

#include <gflags/gflags.h>

DEFINE_string(model, "model.pte", "Path to model.pte (LLM + acoustic head)");
DEFINE_string(codec, "codec_decoder.pte", "Path to codec_decoder.pte");
DEFINE_string(tokenizer, "tekken.json", "Path to tokenizer JSON");
DEFINE_string(text, "", "Text to synthesize");
DEFINE_string(
    voice,
    "",
    "Voice preset name or path to .pt/.bin voice embedding "
    "(default: neutral_female).");
DEFINE_string(output, "output.wav", "Output WAV file path");
DEFINE_string(
    trace_json,
    "",
    "Optional path to write a structured parity trace JSON.");
DEFINE_int32(seed, 42, "Random seed for semantic sampling and flow noise");
DEFINE_double(temperature, 0.0, "Sampling temperature (0 = greedy)");
DEFINE_int32(max_new_tokens, 2048, "Max audio frames to generate");
DEFINE_bool(streaming, false, "Use streaming mode with chunked codec decoding");

static volatile bool g_interrupted = false;
static void signal_handler(int) {
  g_interrupted = true;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_text.empty()) {
    std::cerr << "Error: --text is required" << std::endl;
    return 1;
  }

  std::signal(SIGINT, signal_handler);

  std::cout << "Voxtral TTS" << std::endl;
  std::cout << "  Model: " << FLAGS_model << std::endl;
  std::cout << "  Codec: " << FLAGS_codec << std::endl;
  std::cout << "  Tokenizer: " << FLAGS_tokenizer << std::endl;
  std::cout << "  Text: \"" << FLAGS_text << "\"" << std::endl;
  std::cout << "  Output: " << FLAGS_output << std::endl;
  std::cout << "  Seed: " << FLAGS_seed << std::endl;
  std::cout << "  Mode: " << (FLAGS_streaming ? "streaming" : "offline")
            << std::endl;

  auto load_start = std::chrono::high_resolution_clock::now();

  voxtral_tts::VoxtralTTSRunner runner(
      FLAGS_model, FLAGS_codec, FLAGS_tokenizer);
  runner.set_trace_output_path(FLAGS_trace_json);
  runner.set_seed(static_cast<uint32_t>(FLAGS_seed));

  auto load_end = std::chrono::high_resolution_clock::now();
  auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     load_end - load_start)
                     .count();
  std::cout << "Model loaded in " << load_ms << "ms" << std::endl;

  if (FLAGS_streaming) {
    runner.synthesize_streaming(
        FLAGS_text,
        FLAGS_voice,
        FLAGS_output,
        [](const float* samples, std::size_t count) {
          std::cout << "  Chunk: " << count << " samples" << std::endl;
        },
        static_cast<float>(FLAGS_temperature),
        FLAGS_max_new_tokens);
  } else {
    runner.synthesize_offline(
        FLAGS_text,
        FLAGS_voice,
        FLAGS_output,
        static_cast<float>(FLAGS_temperature),
        FLAGS_max_new_tokens);
  }

  return 0;
}
