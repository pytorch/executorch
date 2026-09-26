/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Standalone DFlash CLI backed by the same MuseGlimmerEngine / LLMSession
// implementation used by muse_glimmer_worker.

#include <gflags/gflags.h>

#include <executorch/examples/models/muse-glimmer/runtime/engine/dflash_session.h>
#include <executorch/examples/models/muse-glimmer/runtime/engine/muse_glimmer_engine.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/types.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

DEFINE_string(model_path, "", "DFlash model .pte file path.");
DEFINE_string(data_path, "", "Data file (.ptd), if the export produced one.");
DEFINE_string(tokenizer_path, "", "Hugging Face tokenizer.json path.");
DEFINE_string(
    image_path,
    "",
    "Optional local JPEG/PNG image path; the prompt must contain one <img> "
    "marker at which to splice its soft tokens.");
DEFINE_string(
    pos_embed_path,
    "",
    "Optional pos_embed.bin path; defaults beside model.pte.");
DEFINE_string(prompt, "The meaning of life is", "Prompt text.");
DEFINE_string(
    prompt_file,
    "",
    "Path to file containing prompt text (overrides --prompt).");
#ifdef EXECUTORCH_BUILD_CUDA
DEFINE_string(
    prompt_tokens_file,
    "",
    "Little-endian int64 prompt IDs; bypasses tokenizer encoding.");
DEFINE_bool(
    tokens_have_bos,
    false,
    "The prompt token file already includes BOS.");
DEFINE_string(
    generated_tokens_file,
    "",
    "Optional path for generated little-endian int64 token IDs.");
#endif
DEFINE_double(temperature, 0.0, "Sampling temperature (0 = greedy).");
DEFINE_int32(top_k, 0, "Top-K sampling (0 = disabled).");
DEFINE_double(top_p, 1.0, "Top-P (nucleus) sampling (1.0 = disabled).");
DEFINE_int64(
    seed,
    -1,
    "RNG seed (-1 = fresh random stream; otherwise positive).");
DEFINE_bool(
    draft_argmax,
    true,
    "Use exact deterministic argmax proposals for stochastic DFlash drafting.");
DEFINE_int32(max_new_tokens, 128, "Maximum visible tokens to generate.");
DEFINE_bool(
    ignore_eos,
    false,
    "Ignore EOS and emit exactly max_new_tokens tokens for benchmarking.");
DEFINE_int32(bos_id, 200000, "BOS token id to prepend.");
DEFINE_int32(eos_id, 200001, "EOS token id.");
DEFINE_int32(
    n_draft,
    3,
    "Draft positions per call (0 = block_length - 1). On CUDA this must be at "
    "most 3.");
DEFINE_int32(
    block_length,
    4,
    "Draft block length per call (0 = artifact maximum; otherwise must be in "
    "[2, block_size]).");
#ifdef EXECUTORCH_BUILD_CUDA
DEFINE_bool(
    cuda_graph,
    true,
    "Enable CUDA graph capture for DFlash target and draft methods.");
#else
DEFINE_bool(cuda_graph, false, "CUDA only.");
#endif

extern "C" void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  if (level == 'D' || level == 'I') {
    return;
  }
  fprintf(stderr, "%c [%s:%zu] %s\n", (char)level, filename, line, message);
}

namespace {
namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;

bool validate_flags() {
  if (FLAGS_model_path.empty() || FLAGS_tokenizer_path.empty()) {
    ET_LOG(Error, "--model_path and --tokenizer_path are required");
    return false;
  }
  if (FLAGS_max_new_tokens <= 0) {
    ET_LOG(Error, "--max_new_tokens must be positive");
    return false;
  }
  if (!std::isfinite(FLAGS_temperature) || FLAGS_temperature < 0.0 ||
      FLAGS_temperature > 2.0) {
    ET_LOG(Error, "--temperature must be finite and in [0, 2]");
    return false;
  }
  if (!std::isfinite(FLAGS_top_p) || FLAGS_top_p <= 0.0 || FLAGS_top_p > 1.0) {
    ET_LOG(Error, "--top_p must be finite and in (0, 1]");
    return false;
  }
  if (FLAGS_top_k < 0) {
    ET_LOG(Error, "--top_k must be nonnegative");
    return false;
  }
  if (FLAGS_seed != -1 && FLAGS_seed <= 0) {
    ET_LOG(Error, "--seed must be -1 or positive");
    return false;
  }
  if (FLAGS_bos_id < 0 || FLAGS_eos_id < 0) {
    ET_LOG(Error, "--bos_id and --eos_id must be nonnegative");
    return false;
  }
  if (FLAGS_block_length < 0 || FLAGS_n_draft < 0) {
    ET_LOG(Error, "--block_length and --n_draft must be nonnegative");
    return false;
  }
  return true;
}

bool read_prompt(std::string& prompt) {
  prompt = FLAGS_prompt;
  if (FLAGS_prompt_file.empty()) {
    return true;
  }

  std::ifstream input(FLAGS_prompt_file);
  if (!input.is_open()) {
    ET_LOG(Error, "Failed to open prompt file: %s", FLAGS_prompt_file.c_str());
    return false;
  }
  prompt.assign(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  return true;
}

#ifdef EXECUTORCH_BUILD_CUDA
bool read_prompt_tokens(std::vector<uint64_t>& tokens) {
  std::ifstream input(
      FLAGS_prompt_tokens_file, std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    ET_LOG(Error, "Cannot open --prompt_tokens_file");
    return false;
  }
  const std::streamsize size = input.tellg();
  if (size <= 0 || size % sizeof(int64_t) != 0) {
    ET_LOG(Error, "Prompt token file must contain little-endian int64 values");
    return false;
  }
  input.seekg(0, std::ios::beg);
  std::vector<int64_t> values(size / sizeof(int64_t));
  if (!input.read(reinterpret_cast<char*>(values.data()), size)) {
    ET_LOG(Error, "Failed to read --prompt_tokens_file");
    return false;
  }
  for (const int64_t value : values) {
    if (value < 0) {
      ET_LOG(Error, "Prompt token IDs must be nonnegative");
      return false;
    }
    tokens.push_back(static_cast<uint64_t>(value));
  }
  if (!FLAGS_tokens_have_bos) {
    tokens.insert(tokens.begin(), static_cast<uint64_t>(FLAGS_bos_id));
  }
  return true;
}

bool write_generated_tokens(const std::vector<int64_t>& tokens) {
  if (FLAGS_generated_tokens_file.empty()) {
    return true;
  }
  std::ofstream output(FLAGS_generated_tokens_file, std::ios::binary);
  output.write(
      reinterpret_cast<const char*>(tokens.data()),
      tokens.size() * sizeof(int64_t));
  if (!output) {
    ET_LOG(Error, "Failed to write --generated_tokens_file");
    return false;
  }
  return true;
}
#endif

void print_rate(const char* label, int64_t tokens, long start_ms, long end_ms) {
  const double elapsed_ms = static_cast<double>(end_ms - start_ms);
  const double tokens_per_second =
      elapsed_ms > 0.0 ? tokens * 1000.0 / elapsed_ms : 0.0;
  printf(
      "%s: %" PRId64 " tokens in %.1f ms (%.1f tok/s)\n",
      label,
      tokens,
      elapsed_ms,
      tokens_per_second);
}

void print_int64_array(const std::vector<int64_t>& values) {
  std::putchar('[');
  for (size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      std::putchar(',');
    }
    std::printf("%" PRId64, values[index]);
  }
  std::putchar(']');
}

void print_dflash_timing(const llm::DFlashDecodeTiming& timing) {
  const double speculative_denominator =
      timing.speculative_cycles > 0 ? timing.speculative_cycles : 1;
  const double cycle_denominator = timing.cycles > 0 ? timing.cycles : 1;
  const double accounted_ms = timing.draft_execute_ms +
      timing.draft_logits_copy_ms + timing.draft_sampling_ms +
      timing.target_execute_ms + timing.target_logits_copy_ms +
      timing.target_hidden_copy_ms + timing.accept_correction_ms +
      timing.state_commit_ms;
  const double accounted_percent = timing.total_cycle_ms > 0.0
      ? accounted_ms * 100.0 / timing.total_cycle_ms
      : 0.0;
  printf(
      "DFlashDecodeTiming {\"cycles\":%" PRId64
      ",\"speculative_cycles\":%" PRId64 ",\"target_only_cycles\":%" PRId64
      ",\"total_cycle_ms\":%.3f"
      ",\"avg_cycle_ms\":%.3f"
      ",\"draft_execute_ms\":%.3f"
      ",\"avg_draft_execute_ms\":%.3f"
      ",\"draft_logits_copy_ms\":%.3f"
      ",\"avg_draft_logits_copy_ms\":%.3f"
      ",\"draft_sampling_ms\":%.3f"
      ",\"avg_draft_sampling_ms\":%.3f"
      ",\"target_execute_ms\":%.3f"
      ",\"avg_target_execute_ms\":%.3f"
      ",\"target_logits_copy_ms\":%.3f"
      ",\"avg_target_logits_copy_ms\":%.3f"
      ",\"target_hidden_copy_ms\":%.3f"
      ",\"avg_target_hidden_copy_ms\":%.3f"
      ",\"accept_correction_ms\":%.3f"
      ",\"avg_accept_correction_ms\":%.3f"
      ",\"state_commit_ms\":%.3f"
      ",\"avg_state_commit_ms\":%.3f"
      ",\"accounted_percent\":%.1f",
      timing.cycles,
      timing.speculative_cycles,
      timing.target_only_cycles,
      timing.total_cycle_ms,
      timing.total_cycle_ms / cycle_denominator,
      timing.draft_execute_ms,
      timing.draft_execute_ms / speculative_denominator,
      timing.draft_logits_copy_ms,
      timing.draft_logits_copy_ms / speculative_denominator,
      timing.draft_sampling_ms,
      timing.draft_sampling_ms / speculative_denominator,
      timing.target_execute_ms,
      timing.target_execute_ms / cycle_denominator,
      timing.target_logits_copy_ms,
      timing.target_logits_copy_ms / cycle_denominator,
      timing.target_hidden_copy_ms,
      timing.target_hidden_copy_ms / cycle_denominator,
      timing.accept_correction_ms,
      timing.accept_correction_ms / cycle_denominator,
      timing.state_commit_ms,
      timing.state_commit_ms / cycle_denominator,
      accounted_percent);
  std::printf(",\"draft_attempts_by_row\":");
  print_int64_array(timing.draft_attempts_by_row);
  std::printf(",\"draft_accepts_by_row\":");
  print_int64_array(timing.draft_accepts_by_row);
  std::printf("}\n");
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (!validate_flags()) {
    return 1;
  }
#ifndef EXECUTORCH_BUILD_CUDA
  if (FLAGS_cuda_graph) {
    ET_LOG(Info, "--cuda_graph ignored on non-CUDA build");
  }
#endif

  llm::Stats stats;
  stats.model_load_start_ms = llm::time_in_ms();

  llm::MuseGlimmerConfig config;
  config.model_path = FLAGS_model_path;
  config.data_path = FLAGS_data_path;
  config.tokenizer_path = FLAGS_tokenizer_path;
  config.pos_embed_path = FLAGS_pos_embed_path;
  config.max_sessions = 1;
  config.eos_id = FLAGS_eos_id;
  config.enable_cuda_graph = FLAGS_cuda_graph;
  config.artifact_mode = llm::MuseGlimmerArtifactMode::DFlash;
  config.dflash_block_length = FLAGS_block_length;
  config.dflash_n_draft = FLAGS_n_draft;
  config.dflash_draft_argmax = FLAGS_draft_argmax;
  config.dflash_ignore_eos = FLAGS_ignore_eos;

  llm::DFlashDecodeTiming dflash_timing;
  config.dflash_timing = &dflash_timing;

  auto engine_result = llm::MuseGlimmerEngine::create(config);
  if (engine_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to create DFlash MuseGlimmerEngine");
    return 1;
  }
  auto engine = std::move(engine_result.get());
  stats.model_load_end_ms = llm::time_in_ms();

  auto session_result = engine->create_session();
  if (session_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to create DFlash LLMSession");
    return 1;
  }
  auto session = std::move(session_result.get());

  const bool has_image = !FLAGS_image_path.empty();
  std::optional<llm::PreparedMuseGlimmerImage> prepared_image;
  llm::DFlashMultimodalSession* multimodal = nullptr;
  int64_t image_tokens = 0;
  if (has_image) {
    auto prepared = engine->prepare_image_from_file(FLAGS_image_path);
    if (prepared.error() != Error::Ok) {
      ET_LOG(Error, "Failed to prepare image for DFlash prefill");
      return 1;
    }
    multimodal = llm::as_dflash_multimodal_session(session.get());
    if (multimodal == nullptr) {
      ET_LOG(Error, "DFlash session does not support image input");
      return 1;
    }
    image_tokens = prepared->num_soft_tokens;
    prepared_image.emplace(std::move(prepared.get()));
  }

  std::vector<uint64_t> prompt_tokens;
#ifdef EXECUTORCH_BUILD_CUDA
  if (!FLAGS_prompt_tokens_file.empty()) {
    if (!read_prompt_tokens(prompt_tokens)) {
      return 1;
    }
    if (has_image) {
      const auto image_patches = std::count(
          prompt_tokens.begin(),
          prompt_tokens.end(),
          llm::kMuseGlimmerImagePatchTokenId);
      if (image_patches != image_tokens) {
        ET_LOG(
            Error,
            "Pre-tokenized image prompt has %td patch tokens; expected "
            "%" PRId64 " at the image position",
            image_patches,
            image_tokens);
        return 1;
      }
    }
    stats.inference_start_ms = llm::time_in_ms();
  } else
#endif
  {
    std::string prompt_text;
    if (!read_prompt(prompt_text)) {
      return 1;
    }
    auto tokenized = llm::tokenize_muse_glimmer_prompt(
        *engine->tokenizer(),
        prompt_text,
        static_cast<uint64_t>(FLAGS_bos_id),
        has_image ? std::optional<int64_t>(image_tokens) : std::nullopt);
    if (!tokenized.ok()) {
      ET_LOG(
          Error,
          "Failed to tokenize prompt; --image_path requires exactly one "
          "<img> marker");
      return 1;
    }
    prompt_tokens = std::move(*tokenized);
    // Tokenization is excluded from the prefill timer, matching solo.cpp.
    // Timing it here made the reported prefill rate depend on prompt text
    // length, and made DFlash look slower than solo for the same work.
    stats.inference_start_ms = llm::time_in_ms();
  }
  if (has_image) {
    multimodal->stage_image_for_next_prefill(std::move(*prepared_image));
  }
  stats.token_encode_end_ms = llm::time_in_ms();
  stats.num_prompt_tokens = static_cast<int64_t>(prompt_tokens.size());

  llm::SamplingConfig sampling;
  sampling.temperature = static_cast<float>(FLAGS_temperature);
  sampling.top_p = static_cast<float>(FLAGS_top_p);
  sampling.top_k = FLAGS_top_k;
  sampling.seed = FLAGS_seed < 0 ? 0 : static_cast<uint64_t>(FLAGS_seed);

  if (session->prefill_tokens(prompt_tokens, &sampling) != Error::Ok) {
    if (auto* multimodal = llm::as_dflash_multimodal_session(session.get())) {
      multimodal->clear_staged_image();
    }
    ET_LOG(Error, "DFlash prefill failed");
    return 1;
  }
  stats.prompt_eval_end_ms = llm::time_in_ms();
  print_rate(
      "Prefill",
      stats.num_prompt_tokens,
      stats.inference_start_ms,
      stats.prompt_eval_end_ms);

  std::vector<int64_t> generated_tokens;
  generated_tokens.reserve(FLAGS_max_new_tokens);
  for (int32_t step = 0; step < FLAGS_max_new_tokens; ++step) {
    auto decode_result = session->decode_one(sampling);
    if (decode_result.error() != Error::Ok) {
      ET_LOG(Error, "DFlash decode failed at step %d", step);
      return 1;
    }
    const auto& decoded = decode_result.get();
    if (decoded.is_terminal) {
      break;
    }
    if (generated_tokens.empty()) {
      stats.first_token_ms = llm::time_in_ms();
    }
    if (!decoded.text_piece.empty()) {
      fwrite(decoded.text_piece.data(), 1, decoded.text_piece.size(), stdout);
      fflush(stdout);
    }
    generated_tokens.push_back(static_cast<int64_t>(decoded.token_id));
  }

  stats.inference_end_ms = llm::time_in_ms();
  if (FLAGS_ignore_eos &&
      generated_tokens.size() != static_cast<size_t>(FLAGS_max_new_tokens)) {
    ET_LOG(
        Error,
        "Exact-output mode emitted %zu tokens, expected %d",
        generated_tokens.size(),
        FLAGS_max_new_tokens);
    return 1;
  }
#ifdef EXECUTORCH_BUILD_CUDA
  if (!write_generated_tokens(generated_tokens)) {
    return 1;
  }
#endif

  stats.num_generated_tokens = static_cast<int64_t>(generated_tokens.size());
  printf("\n");
  print_rate(
      "Decode",
      static_cast<int64_t>(generated_tokens.size()),
      stats.prompt_eval_end_ms,
      stats.inference_end_ms);
  printf("Prompt tokens: %" PRId64 "\n", stats.num_prompt_tokens);
  llm::print_report(stats);
  print_dflash_timing(dflash_timing);
  return 0;
}
