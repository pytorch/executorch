/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// CLI entry point for the Voxtral TTS runner.
//
// Loads model.pte (LLM + acoustic transformer), codec.pte (audio decoder),
// and a tokenizer. Generates speech from text and writes a WAV file.
//
// Usage:
//   ./voxtral_tts_runner \
//       --model_path model.pte \
//       --codec_path codec.pte \
//       --tokenizer_path tekken.json \
//       --prompt "Hello, how are you?" \
//       --output_path output.wav

#include <cstdio>

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>

#include "voxtral_tts_runner.h"

DEFINE_string(model_path, "model.pte", "Path to Voxtral TTS model (.pte).");
DEFINE_string(
    codec_path,
    "",
    "Path to audio codec decoder (.pte). If empty, outputs audio codes only.");
DEFINE_string(tokenizer_path, "tekken.json", "Path to Tekken tokenizer file.");
DEFINE_string(prompt, "", "Input text to synthesize.");
DEFINE_string(
    output_path,
    "output.wav",
    "Output WAV file path (default: output.wav).");
DEFINE_int32(max_tokens, 2048, "Maximum tokens to generate.");
DEFINE_double(temperature, 0.0, "Sampling temperature (0 = greedy).");
DEFINE_int32(seed, 42, "Random seed for flow matching noise.");
DEFINE_string(
    voice_path,
    "",
    "Path to voice embedding file (.bin). Generate with: "
    "python -c \"import torch,struct; v=torch.load('voice.pt',map_location='cpu',weights_only=True).float(); "
    "f=open('voice.bin','wb'); f.write(struct.pack('<qq',*v.shape)); f.write(v.numpy().tobytes())\"");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_prompt.empty()) {
    ET_LOG(Error, "Provide --prompt with input text.");
    return 1;
  }

  // Track timing
  ::executorch::extension::llm::Stats stats;
  stats.model_load_start_ms = ::executorch::extension::llm::time_in_ms();

  // Load model + codec + tokenizer
  voxtral_tts::VoxtralTTSRunner runner(
      FLAGS_model_path,
      FLAGS_tokenizer_path,
      FLAGS_codec_path,
      /*warmup=*/true);

  // Load voice embedding if provided
  if (!FLAGS_voice_path.empty()) {
    runner.load_voice(FLAGS_voice_path);
  }

  stats.model_load_end_ms = ::executorch::extension::llm::time_in_ms();

  ET_LOG(
      Info,
      "Model loaded in %.1f s",
      (stats.model_load_end_ms - stats.model_load_start_ms) / 1000.0);

  // Configure generation
  voxtral_tts::GenerateConfig config;
  config.max_tokens = FLAGS_max_tokens;
  config.temperature = static_cast<float>(FLAGS_temperature);
  config.seed = static_cast<uint64_t>(FLAGS_seed);

  // Progress callback (invoked per audio frame)
  int frame_count = 0;
  auto token_cb = [&frame_count](const std::string& piece) {
    frame_count++;
    if (frame_count % 100 == 0) {
      fprintf(stderr, "\r  %d audio frames generated...", frame_count);
      fflush(stderr);
    }
  };

  // Generate
  ET_LOG(Info, "Generating speech for: \"%s\"", FLAGS_prompt.c_str());
  stats.inference_start_ms = ::executorch::extension::llm::time_in_ms();

  std::vector<int64_t> audio_codes;
  auto waveform = runner.generate(FLAGS_prompt, config, token_cb, &audio_codes);

  stats.inference_end_ms = ::executorch::extension::llm::time_in_ms();

  if (frame_count > 0) {
    fprintf(stderr, "\r  %d audio frames generated.     \n", frame_count);
  }

  double inference_s =
      (stats.inference_end_ms - stats.inference_start_ms) / 1000.0;
  ET_LOG(Info, "Inference: %.1f s (%d audio frames)", inference_s, frame_count);

  // Write output
  if (!waveform.empty()) {
    voxtral_tts::write_wav(
        FLAGS_output_path,
        waveform.data(),
        static_cast<int64_t>(waveform.size()),
        static_cast<int32_t>(runner.sample_rate()));
    double duration =
        static_cast<double>(waveform.size()) / runner.sample_rate();
    ET_LOG(
        Info,
        "Audio: %.1f s at %ld Hz -> %s",
        duration,
        static_cast<long>(runner.sample_rate()),
        FLAGS_output_path.c_str());
  } else if (!audio_codes.empty()) {
    ET_LOG(
        Info,
        "Generated %zu audio codes (no codec loaded, no WAV output).",
        audio_codes.size());
  } else {
    ET_LOG(Info, "No audio generated.");
  }

  // Metal backend stats can be added here when the Metal stats API is available

  return 0;
}
