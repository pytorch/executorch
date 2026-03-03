/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Benchmark for the Parakeet ASR pipeline.
 *
 * Runs preprocessor -> encoder -> TDT greedy decode with real audio input
 * over multiple iterations and reports per-phase timing.
 *
 * Backend-agnostic: works with any .pte (TensorRT, CUDA, XNNPACK, etc.).
 *
 * Usage:
 *   parakeet_benchmark --model model.pte --audio speech.wav
 *   parakeet_benchmark --model model.pte --audio speech.wav -n 20 -w 3
 *   parakeet_benchmark --model model.pte --audio speech.wav --tokenizer
 * tok.model
 */

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "decode.h"
#include "tokenizer_utils.h"
#include "types.h"

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/runner/wav_loader.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;

// -----------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------

static constexpr uint32_t kDefaultIterations = 10;
static constexpr uint32_t kDefaultWarmup = 2;

struct Args {
  std::string model_path;
  std::string audio_path;
  std::string tokenizer_path = "tokenizer.model";
  std::string data_path;
  uint32_t iterations = kDefaultIterations;
  uint32_t warmup = kDefaultWarmup;
};

static void print_usage() {
  printf(
      "Usage: parakeet_benchmark [options]\n"
      "\n"
      "Options:\n"
      "  --model PATH       Path to .pte model (required)\n"
      "  --audio PATH       Path to .wav audio file (required)\n"
      "  --tokenizer PATH   Path to tokenizer.model (default: tokenizer.model)\n"
      "  --data_path PATH   Delegate data file (for CUDA backend)\n"
      "  -n, --iterations N Timed iterations (default: %u)\n"
      "  -w, --warmup N     Warmup iterations (default: %u)\n"
      "  -h, --help         Show this message\n",
      kDefaultIterations,
      kDefaultWarmup);
}

static bool parse_args(int argc, char** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage();
      return false;
    } else if (arg == "--model" && i + 1 < argc) {
      args.model_path = argv[++i];
    } else if (arg == "--audio" && i + 1 < argc) {
      args.audio_path = argv[++i];
    } else if (arg == "--tokenizer" && i + 1 < argc) {
      args.tokenizer_path = argv[++i];
    } else if (arg == "--data_path" && i + 1 < argc) {
      args.data_path = argv[++i];
    } else if ((arg == "-n" || arg == "--iterations") && i + 1 < argc) {
      args.iterations = static_cast<uint32_t>(std::stoul(argv[++i]));
    } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
      args.warmup = static_cast<uint32_t>(std::stoul(argv[++i]));
    } else {
      fprintf(stderr, "Error: unknown argument '%s'\n", arg.c_str());
      print_usage();
      return false;
    }
  }
  if (args.model_path.empty() || args.audio_path.empty()) {
    fprintf(stderr, "Error: --model and --audio are required\n");
    print_usage();
    return false;
  }
  return true;
}

// -----------------------------------------------------------------------
// Timing helpers
// -----------------------------------------------------------------------

static long now_ms() {
  return ::executorch::extension::llm::time_in_ms();
}

struct PhaseStats {
  std::vector<double> times_ms;

  void record(double ms) {
    times_ms.push_back(ms);
  }

  double avg() const {
    if (times_ms.empty())
      return 0;
    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) /
        times_ms.size();
  }

  double min_val() const {
    if (times_ms.empty())
      return 0;
    return *std::min_element(times_ms.begin(), times_ms.end());
  }

  double max_val() const {
    if (times_ms.empty())
      return 0;
    return *std::max_element(times_ms.begin(), times_ms.end());
  }
};

// -----------------------------------------------------------------------
// Benchmark
// -----------------------------------------------------------------------

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, args)) {
    return 1;
  }

  // --- Load model ---
  printf("Loading model: %s\n", args.model_path.c_str());
  long load_start = now_ms();

  std::unique_ptr<Module> model;
  if (!args.data_path.empty()) {
    model = std::make_unique<Module>(
        args.model_path, args.data_path, Module::LoadMode::Mmap);
  } else {
    model = std::make_unique<Module>(args.model_path, Module::LoadMode::Mmap);
  }
  if (model->load() != Error::Ok) {
    fprintf(stderr, "Failed to load model\n");
    return 1;
  }

  const std::vector<std::string> methods = {
      "preprocessor", "encoder", "decoder_step", "joint"};
  for (const auto& m : methods) {
    if (model->load_method(m) != Error::Ok) {
      fprintf(stderr, "Failed to load method: %s\n", m.c_str());
      return 1;
    }
  }
  long load_end = now_ms();
  printf("Model loaded in %ld ms\n", load_end - load_start);

  // --- Load audio ---
  printf("Loading audio: %s\n", args.audio_path.c_str());
  std::vector<float> audio_data =
      ::executorch::extension::llm::load_wav_audio_data(args.audio_path);
  if (audio_data.empty()) {
    fprintf(stderr, "Failed to load audio\n");
    return 1;
  }
  double audio_duration_ms =
      static_cast<double>(audio_data.size()) / 16000.0 * 1000.0;
  printf(
      "Audio: %.2fs (%zu samples)\n",
      audio_duration_ms / 1000.0,
      audio_data.size());

  auto audio_tensor = from_blob(
      audio_data.data(),
      {static_cast<executorch::aten::SizesType>(audio_data.size())},
      executorch::aten::ScalarType::Float);
  std::vector<int64_t> audio_len_data = {
      static_cast<int64_t>(audio_data.size())};
  auto audio_len_tensor =
      from_blob(audio_len_data.data(), {1}, executorch::aten::ScalarType::Long);

  // --- Preprocessor (run once) ---
  printf("Running preprocessor...\n");
  long preproc_start = now_ms();
  auto proc_result = model->execute(
      "preprocessor", std::vector<EValue>{audio_tensor, audio_len_tensor});
  long preproc_end = now_ms();
  if (!proc_result.ok()) {
    fprintf(stderr, "Preprocessor failed\n");
    return 1;
  }
  double preproc_ms = static_cast<double>(preproc_end - preproc_start);
  auto mel = proc_result.get()[0].toTensor();
  int64_t mel_len_value =
      proc_result.get()[1].toTensor().const_data_ptr<int64_t>()[0];
  std::vector<int64_t> mel_len_data = {mel_len_value};
  auto mel_len =
      from_blob(mel_len_data.data(), {1}, executorch::aten::ScalarType::Long);

  printf(
      "Mel: [%ld, %ld, %ld], len=%lld\n",
      static_cast<long>(mel.sizes()[0]),
      static_cast<long>(mel.sizes()[1]),
      static_cast<long>(mel.sizes()[2]),
      static_cast<long long>(mel_len_value));

  // --- Query model metadata ---
  std::vector<EValue> empty;
  int64_t num_rnn_layers =
      model->execute("num_rnn_layers", empty).get()[0].toInt();
  int64_t pred_hidden = model->execute("pred_hidden", empty).get()[0].toInt();
  int64_t blank_id = model->execute("blank_id", empty).get()[0].toInt();

  // --- Warmup ---
  printf("Warming up (%u iterations)...\n", args.warmup);
  std::vector<parakeet::Token> last_tokens;
  for (uint32_t i = 0; i < args.warmup; ++i) {
    auto enc_result =
        model->execute("encoder", std::vector<EValue>{mel, mel_len});
    if (!enc_result.ok()) {
      fprintf(stderr, "Encoder failed during warmup\n");
      return 1;
    }
    auto f_proj = enc_result.get()[0].toTensor();
    int64_t enc_len =
        enc_result.get()[1].toTensor().const_data_ptr<int64_t>()[0];
    last_tokens = parakeet::greedy_decode_executorch(
        *model, f_proj, enc_len, blank_id, num_rnn_layers, pred_hidden);
  }

  // --- Timed iterations ---
  printf("Benchmarking (%u iterations)...\n", args.iterations);
  PhaseStats encoder_stats, decode_stats, e2e_stats;

  for (uint32_t i = 0; i < args.iterations; ++i) {
    printf("  [%u/%u]\r", i + 1, args.iterations);
    fflush(stdout);

    long iter_start = now_ms();

    // Encoder
    long enc_start = now_ms();
    auto enc_result =
        model->execute("encoder", std::vector<EValue>{mel, mel_len});
    long enc_end = now_ms();
    if (!enc_result.ok()) {
      fprintf(stderr, "Encoder failed at iteration %u\n", i);
      return 1;
    }
    auto f_proj = enc_result.get()[0].toTensor();
    int64_t enc_len =
        enc_result.get()[1].toTensor().const_data_ptr<int64_t>()[0];

    // Decode
    long dec_start = now_ms();
    last_tokens = parakeet::greedy_decode_executorch(
        *model, f_proj, enc_len, blank_id, num_rnn_layers, pred_hidden);
    long dec_end = now_ms();

    long iter_end = now_ms();

    encoder_stats.record(static_cast<double>(enc_end - enc_start));
    decode_stats.record(static_cast<double>(dec_end - dec_start));
    e2e_stats.record(static_cast<double>(iter_end - iter_start));
  }
  printf("                    \r");

  // --- Transcription (sanity check) ---
  std::string transcription;
  auto tokenizer =
      ::executorch::extension::llm::load_tokenizer(args.tokenizer_path);
  if (tokenizer) {
    std::vector<parakeet::TokenId> token_ids;
    for (const auto& tok : last_tokens) {
      token_ids.push_back(tok.id);
    }
    transcription =
        parakeet::tokenizer_utils::decode_token_sequence(token_ids, *tokenizer);
  }

  // --- Print results ---
  printf("\n");
  printf("═══════════════════════════════════════════════════════════\n");
  printf("  Parakeet Benchmark\n");
  printf("═══════════════════════════════════════════════════════════\n");
  printf("  Model:      %s\n", args.model_path.c_str());
  printf(
      "  Audio:      %s (%.2fs, %zu samples)\n",
      args.audio_path.c_str(),
      audio_duration_ms / 1000.0,
      audio_data.size());
  printf("  Iterations: %u (warmup: %u)\n", args.iterations, args.warmup);
  printf("  Model load: %ld ms\n", load_end - load_start);
  printf("\n");
  printf(
      "  %-16s %10s %10s %10s\n", "Phase", "Avg (ms)", "Min (ms)", "Max (ms)");
  printf("  ────────────────────────────────────────────────────\n");
  printf("  %-16s %10.1f %10s %10s\n", "Preprocessor", preproc_ms, "-", "-");
  printf(
      "  %-16s %10.1f %10.1f %10.1f\n",
      "Encoder",
      encoder_stats.avg(),
      encoder_stats.min_val(),
      encoder_stats.max_val());
  printf(
      "  %-16s %10.1f %10.1f %10.1f\n",
      "Decode",
      decode_stats.avg(),
      decode_stats.min_val(),
      decode_stats.max_val());
  printf("  ────────────────────────────────────────────────────\n");
  printf(
      "  %-16s %10.1f %10.1f %10.1f\n",
      "End-to-end",
      e2e_stats.avg(),
      e2e_stats.min_val(),
      e2e_stats.max_val());
  printf("\n");

  double rtf = (preproc_ms + e2e_stats.avg()) / audio_duration_ms;
  printf(
      "  Real-time factor: %.3fx (%.2fs audio in %.1f ms)\n",
      rtf,
      audio_duration_ms / 1000.0,
      preproc_ms + e2e_stats.avg());

  if (!transcription.empty()) {
    // Truncate long transcriptions for display
    std::string display = transcription;
    if (display.size() > 100) {
      display = display.substr(0, 97) + "...";
    }
    printf("  Transcription: \"%s\"\n", display.c_str());
  }
  printf(
      "  Tokens: %zu (%lld encoder frames)\n",
      last_tokens.size(),
      static_cast<long long>(
          last_tokens.empty() ? 0 : last_tokens.back().start_offset));
  printf("═══════════════════════════════════════════════════════════\n");

  return 0;
}
