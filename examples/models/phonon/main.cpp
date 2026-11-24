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
//
// --stream plays the audio while it is generated:
//
//   phonon_runner --model_path ... --text "..." --stream |
//       aplay -q -f S16_LE -r 24000

#include "runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

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
DEFINE_bool(
    stream,
    false,
    "Write raw 16-bit little-endian mono PCM to stdout as each frame is "
    "decoded, instead of the WAV file");
DEFINE_double(
    temperature,
    -1,
    "Flow noise variance; 0 removes the noise (default: the model's, 0.3)");
DEFINE_uint64(seed, 0, "Noise seed");
DEFINE_bool(
    benchmark,
    false,
    "Synthesize the text 4 times and print the timing of the first (cold) "
    "run and the median of the other runs (warm)");

namespace {

constexpr int kBenchmarkRuns = 4;

struct Timing {
  double rtf, first_audio_ms, audio_s, flow_lm_ms, mimi_decode_ms;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

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

// Rounded like run_phonon.py (round half to even).
int16_t to_pcm16(float sample) {
  const double value = std::isfinite(sample)
      ? std::nearbyint(static_cast<double>(sample) * 32767.0)
      : 0.0;
  return static_cast<int16_t>(std::clamp(value, -32768.0, 32767.0));
}

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
    write_u16(out, static_cast<uint16_t>(to_pcm16(sample)));
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

    double first_audio_s = 0;
    std::vector<unsigned char> bytes;
    const auto on_audio = [&](const float* pcm, size_t size) {
      if (first_audio_s == 0) {
        first_audio_s = seconds_since(start);
      }
      if (!FLAGS_stream) {
        return;
      }
      bytes.resize(size * 2);
      for (size_t i = 0; i < size; ++i) {
        const auto v = static_cast<uint16_t>(to_pcm16(pcm[i]));
        bytes[2 * i] = static_cast<unsigned char>(v);
        bytes[2 * i + 1] = static_cast<unsigned char>(v >> 8);
      }
      std::fwrite(bytes.data(), 1, bytes.size(), stdout);
      std::fflush(stdout);
    };
    phonon::Synthesis result;
    double synth_s = 0;
    std::vector<Timing> timings;
    for (int run = 0; run < (FLAGS_benchmark ? kBenchmarkRuns : 1); ++run) {
      first_audio_s = 0;
      start = std::chrono::steady_clock::now();
      result = runner.synthesize(
          FLAGS_text, voice, FLAGS_temperature, FLAGS_seed, on_audio);
      synth_s = seconds_since(start);
      if (result.pcm.empty()) {
        throw std::runtime_error("No audio generated");
      }
      const double audio_s =
          static_cast<double>(result.pcm.size()) / runner.sample_rate();
      timings.push_back(
          {synth_s / audio_s,
           first_audio_s * 1000,
           audio_s,
           median(result.flow_lm_ms),
           median(result.mimi_decode_ms)});
    }
    if (!FLAGS_stream) {
      write_wav(FLAGS_output, result.pcm, runner.sample_rate());
    }

    const double audio_s = timings.back().audio_s;
    std::fprintf(
        FLAGS_stream ? stderr : stdout,
        "%s %s: %.2f s of audio, %lld frames, load %.2f s, first audio "
        "%.3f s, synthesis %.2f s (RTF %.2f)\n",
        FLAGS_stream ? "Streamed" : "Wrote",
        FLAGS_stream ? "to stdout" : FLAGS_output.c_str(),
        audio_s,
        static_cast<long long>(result.frames),
        load_s,
        first_audio_s,
        synth_s,
        synth_s / audio_s);
    if (FLAGS_benchmark) {
      const auto warm = [&](double Timing::*field) {
        std::vector<double> values;
        for (size_t run = 1; run < timings.size(); ++run) {
          values.push_back(timings[run].*field);
        }
        return median(values);
      };
      const Timing warm_median{
          warm(&Timing::rtf),
          warm(&Timing::first_audio_ms),
          warm(&Timing::audio_s),
          warm(&Timing::flow_lm_ms),
          warm(&Timing::mimi_decode_ms)};
      for (const auto& [label, timing] :
           {std::pair{"cold (run 1)", timings.front()},
            std::pair{"warm (median of runs 2-4)", warm_median}}) {
        std::fprintf(
            FLAGS_stream ? stderr : stdout,
            "%s: RTF %.3f, first audio %.1f ms, %.2f s of audio, flow_lm "
            "%.2f ms/step, mimi_decode %.2f ms/call\n",
            label,
            timing.rtf,
            timing.first_audio_ms,
            timing.audio_s,
            timing.flow_lm_ms,
            timing.mimi_decode_ms);
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "Phonon: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
