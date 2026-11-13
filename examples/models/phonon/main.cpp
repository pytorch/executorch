/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Text to speech with an exported Phonon program:
//
//   phonon_runner --model_path phonon_exports/xnnpack/phonon.pte \
//       --text "Hello from ExecuTorch." --output hello.wav

#include "runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <gflags/gflags.h>

DEFINE_string(
    model_path,
    "phonon_exports/xnnpack/phonon.pte",
    "Exported program");
DEFINE_string(data_path, "", "External CUDA data file (aoti_cuda_blob.ptd)");
DEFINE_string(
    tokenizer_path,
    "",
    "SentencePiece tokenizer.model (default: next to the program)");
DEFINE_string(
    voice_path,
    "",
    "Voice .safetensors (default: default-voice.safetensors next to the "
    "program)");
DEFINE_string(text, "", "Text to synthesize");
DEFINE_string(output, "phonon.wav", "Output WAV file (16-bit PCM, mono)");
DEFINE_double(
    temperature,
    -1,
    "Flow noise variance; 0 removes the noise (default: the model's, 0.3)");
DEFINE_uint64(seed, 0, "Noise seed");

namespace {

std::string sibling(const std::string& path, const std::string& name) {
  const auto slash = path.find_last_of("/\\");
  return slash == std::string::npos ? name : path.substr(0, slash + 1) + name;
}

void write_u32(std::ofstream& out, uint32_t v) {
  const unsigned char b[4] = {
      static_cast<unsigned char>(v),
      static_cast<unsigned char>(v >> 8),
      static_cast<unsigned char>(v >> 16),
      static_cast<unsigned char>(v >> 24)};
  out.write(reinterpret_cast<const char*>(b), 4);
}

void write_u16(std::ofstream& out, uint16_t v) {
  const unsigned char b[2] = {
      static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8)};
  out.write(reinterpret_cast<const char*>(b), 2);
}

// 16-bit mono PCM, rounded like run_phonon.py (round half to even).
void write_wav(
    const std::string& path,
    const std::vector<float>& pcm,
    int64_t sample_rate) {
  std::ofstream out(path, std::ios::binary);
  const auto data_bytes = static_cast<uint32_t>(pcm.size() * 2);
  out.write("RIFF", 4);
  write_u32(out, 36 + data_bytes);
  out.write("WAVEfmt ", 8);
  write_u32(out, 16);
  write_u16(out, 1);
  write_u16(out, 1);
  write_u32(out, static_cast<uint32_t>(sample_rate));
  write_u32(out, static_cast<uint32_t>(sample_rate * 2));
  write_u16(out, 2);
  write_u16(out, 16);
  out.write("data", 4);
  write_u32(out, data_bytes);
  for (float sample : pcm) {
    double value = std::isfinite(sample)
        ? std::nearbyint(static_cast<double>(sample) * 32767.0)
        : 0.0;
    value = std::clamp(value, -32768.0, 32767.0);
    write_u16(out, static_cast<uint16_t>(static_cast<int16_t>(value)));
  }
  if (!out) {
    throw std::runtime_error("Could not write " + path);
  }
}

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    if (FLAGS_text.empty()) {
      throw std::invalid_argument("Provide --text");
    }
#ifdef EXECUTORCH_BUILD_CUDA
    if (FLAGS_data_path.empty()) {
      throw std::invalid_argument(
          "Provide --data_path pointing to aoti_cuda_blob.ptd for CUDA");
    }
#endif
    auto start = std::chrono::steady_clock::now();
    phonon::Runner runner(
        FLAGS_model_path,
        FLAGS_tokenizer_path.empty()
            ? sibling(FLAGS_model_path, "tokenizer.model")
            : FLAGS_tokenizer_path,
        FLAGS_data_path);
    const auto voice = runner.load_voice(
        FLAGS_voice_path.empty()
            ? sibling(FLAGS_model_path, "default-voice.safetensors")
            : FLAGS_voice_path);
    const double load_s = seconds_since(start);

    start = std::chrono::steady_clock::now();
    const auto result =
        runner.synthesize(FLAGS_text, voice, FLAGS_temperature, FLAGS_seed);
    const double synth_s = seconds_since(start);
    if (result.pcm.empty()) {
      throw std::runtime_error("No audio generated");
    }
    write_wav(FLAGS_output, result.pcm, runner.sample_rate());

    const double audio_s =
        static_cast<double>(result.pcm.size()) / runner.sample_rate();
    std::printf(
        "Wrote %s: %.2f s of audio, %lld frames, load %.2f s, synthesis "
        "%.2f s (RTF %.2f)\n",
        FLAGS_output.c_str(),
        audio_s,
        static_cast<long long>(result.frames),
        load_s,
        synth_s,
        synth_s / audio_s);
  } catch (const std::exception& error) {
    std::cerr << "Phonon: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
