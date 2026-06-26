/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <executorch/examples/models/gemma4_31b/gemma4_31b_engine.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>

#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/types.h>
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

#ifdef EXECUTORCH_BUILD_CUDA
#include <cuda_runtime.h>
#endif

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(data_path, "", "Data file (.ptd) for CUDA backend.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(prompt, "Hello", "Prompt text.");
DEFINE_string(
    prompt_file,
    "",
    "Path to file containing prompt text (overrides --prompt).");
DEFINE_double(temperature, 0.8, "Sampling temperature (0 = near-greedy).");
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");
DEFINE_int32(bos_id, 2, "BOS token id to prepend (Gemma convention: 2).");
DEFINE_int32(eos_id, 1, "EOS token id (Gemma convention: 1).");
DEFINE_bool(
    raw_prompt,
    false,
    "Skip chat-template wrapping (use if the prompt is already formatted).");
DEFINE_bool(
    cuda_graph,
    false,
    "Enable CUDA graph capture for the decode method. CUDA only.");

namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;

namespace {

std::optional<std::string> read_prompt() {
  if (FLAGS_prompt_file.empty()) {
    return FLAGS_prompt;
  }
  std::ifstream f(FLAGS_prompt_file);
  if (!f.is_open()) {
    ET_LOG(Error, "Failed to open prompt file: %s", FLAGS_prompt_file.c_str());
    return std::nullopt;
  }
  return std::string(
      (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::string format_prompt(std::string prompt) {
  if (FLAGS_raw_prompt) {
    return prompt;
  }
  return "<|turn>user\n" + prompt +
      "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_path.empty() || FLAGS_tokenizer_path.empty()) {
    ET_LOG(Error, "--model_path and --tokenizer_path are required");
    return 1;
  }

  llm::Stats stats;

#ifdef EXECUTORCH_BUILD_CUDA
  size_t gpu_free_bytes = 0, gpu_total_bytes = 0;
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_total_bytes = gpu_total_bytes;
  stats.gpu_free_before_load_bytes = gpu_free_bytes;
#else
  if (FLAGS_cuda_graph) {
    ET_LOG(Info, "--cuda_graph ignored on non-CUDA build");
  }
#endif

  llm::Gemma4_31BConfig config;
  config.model_path = FLAGS_model_path;
  config.data_path = FLAGS_data_path;
  config.tokenizer_path = FLAGS_tokenizer_path;
  config.max_sessions = 1;
  config.eos_id = FLAGS_eos_id;
#ifdef EXECUTORCH_BUILD_CUDA
  config.enable_cuda_graph = FLAGS_cuda_graph;
#endif

  stats.model_load_start_ms = llm::time_in_ms();
  auto engine_result = llm::Gemma4_31BEngine::create(config);
  if (engine_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to create Gemma4_31BEngine");
    return 1;
  }
  auto engine = std::move(engine_result.get());
  stats.model_load_end_ms = llm::time_in_ms();

#ifdef EXECUTORCH_BUILD_CUDA
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_load_bytes = gpu_free_bytes;
#endif

  auto session_result = engine->create_session();
  if (session_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to create session");
    return 1;
  }
  auto session = std::move(session_result.get());

  auto prompt = read_prompt();
  if (!prompt.has_value()) {
    return 1;
  }
  std::string prompt_text = format_prompt(std::move(*prompt));

  auto encoded = engine->tokenizer()->encode(prompt_text, /*bos=*/0, /*eos=*/0);
  if (!encoded.ok()) {
    ET_LOG(Error, "Failed to encode prompt");
    return 1;
  }
  std::vector<uint64_t> prompt_tokens;
  prompt_tokens.reserve(encoded->size() + 1);
  prompt_tokens.push_back(static_cast<uint64_t>(FLAGS_bos_id));
  prompt_tokens.insert(prompt_tokens.end(), encoded->begin(), encoded->end());

  stats.num_prompt_tokens = static_cast<int64_t>(prompt_tokens.size());
  printf("Prompt tokens: %" PRId64 "\n", stats.num_prompt_tokens);

  llm::SamplingConfig sampling;
  sampling.temperature = static_cast<float>(FLAGS_temperature);
  stats.inference_start_ms = llm::time_in_ms();
  if (session->prefill_tokens(prompt_tokens, &sampling) != Error::Ok) {
    ET_LOG(Error, "Prefill failed");
    return 1;
  }
  stats.prompt_eval_end_ms = llm::time_in_ms();
  stats.first_token_ms = stats.prompt_eval_end_ms;

  int64_t generated = 0;
  for (; generated < FLAGS_max_new_tokens; ++generated) {
    auto step = session->decode_one(sampling);
    if (step.error() != Error::Ok) {
      ET_LOG(Error, "Decode failed");
      return 1;
    }
    const auto& d = step.get();
    if (d.is_terminal) {
      break;
    }
    printf("%s", d.text_piece.c_str());
    fflush(stdout);
  }
  printf("\n");

  stats.inference_end_ms = llm::time_in_ms();
  stats.num_generated_tokens = generated;

#ifdef EXECUTORCH_BUILD_CUDA
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_generate_bytes = gpu_free_bytes;
  stats.gpu_peak_usage_mb =
      (stats.gpu_total_bytes - gpu_free_bytes) / 1024.0 / 1024.0;
#endif

  llm::print_report(stats);
  return 0;
}
