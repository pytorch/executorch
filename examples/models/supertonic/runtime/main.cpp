/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "supertonic_runner.h"
#include "wav_writer.h"

#include <gflags/gflags.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

DEFINE_string(pte, "", "Path to the single Supertonic FP16 MLX .pte file.");
DEFINE_string(
    asset_dir,
    "",
    "Published Supertonic asset root containing onnx/unicode_indexer.json.");
DEFINE_string(
    voice_style,
    "",
    "Exactly one published batch-1 voice-style JSON path.");
DEFINE_string(text, "", "Text to synthesize.");
DEFINE_string(language, "en", "Published Supertonic language tag.");
DEFINE_double(speed, 1.05, "Positive duration speed divisor.");
DEFINE_uint64(seed, 42, "Portable xorshift64/Box-Muller latent seed.");
DEFINE_string(output, "supertonic.wav", "Output mono PCM16 WAV path.");

namespace {

void require_file(const std::filesystem::path& path, const char* flag) {
  if (!std::filesystem::is_regular_file(path)) {
    throw std::invalid_argument(
        std::string(flag) + " does not name a readable file: " + path.string());
  }
}

} // namespace

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Synthesize Supertonic speech with a batch-1 FP16 MLX PTE.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  try {
    if (FLAGS_pte.empty() || FLAGS_asset_dir.empty() ||
        FLAGS_voice_style.empty() || FLAGS_text.empty()) {
      throw std::invalid_argument(
          "--pte, --asset_dir, --voice_style, and --text are required");
    }
    const std::filesystem::path pte(FLAGS_pte);
    const std::filesystem::path indexer =
        std::filesystem::path(FLAGS_asset_dir) / "onnx" /
        "unicode_indexer.json";
    require_file(pte, "--pte");
    require_file(indexer, "--asset_dir");
    const std::string style =
        supertonic::require_single_voice_style_path({FLAGS_voice_style});
    supertonic::validate_language(FLAGS_language);
    if (!std::isfinite(FLAGS_speed) || FLAGS_speed <= 0.0 ||
        FLAGS_speed > std::numeric_limits<float>::max()) {
      throw std::invalid_argument(
          "--speed must be finite, positive, and representable as float");
    }
    require_file(style, "--voice_style");

    supertonic::SupertonicRunner runner(pte.string(), indexer.string());

    supertonic::SynthesisOptions options;
    options.text = FLAGS_text;
    options.language = FLAGS_language;
    options.voice_style_paths = {style};
    options.speed = static_cast<float>(FLAGS_speed);
    options.seed = FLAGS_seed;
    auto result = runner.synthesize(options);
    if (!supertonic::write_pcm16_wav(
            FLAGS_output,
            result.waveform,
            static_cast<int>(runner.metadata().sample_rate))) {
      throw std::runtime_error("failed to write output WAV: " + FLAGS_output);
    }
    const double audio_seconds = static_cast<double>(result.waveform.size()) /
        runner.metadata().sample_rate;
    std::cout << "Wrote " << result.waveform.size() << " samples ("
              << audio_seconds << " s) at " << runner.metadata().sample_rate
              << " Hz to " << FLAGS_output << "\n";
    if (audio_seconds > 0.0) {
      std::cout << "Synthesis " << result.elapsed_seconds << " s, RTF "
                << result.elapsed_seconds / audio_seconds << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Supertonic synthesis failed: " << error.what() << "\n";
    return 1;
  }
}
