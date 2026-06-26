/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Thin CLI over Qwen35MoEEngine / Qwen35MoESession: parse flags, build the
// engine + a session, encode the prompt, prefill_tokens(), then loop
// decode_one() printing pieces and timing/stats. All model execution lives in
// qwen35_moe_engine.{h,cpp}.

#include <gflags/gflags.h>

#include <executorch/examples/models/qwen3_5_moe/qwen35_moe_engine.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <string>
#include <vector>

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
DEFINE_double(temperature, 0.8, "Sampling temperature (0 = greedy).");
DEFINE_double(
    top_p,
    1.0,
    "Nucleus sampling top_p in (0, 1]; 1.0 = off. Requires a model exported "
    "with --sample (MLX on-device sampling).");
DEFINE_int64(
    seed,
    0,
    "Base RNG seed for on-device sampling; the runner increments it per token. "
    "Requires a model exported with --sample.");
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");
DEFINE_int32(
    warmup,
    0,
    "Warmup iterations to discard before timing. One model load; the session is "
    "reset between iterations. Warmup ramps GPU clocks so the timed iterations "
    "reflect steady state.");
DEFINE_int32(num_iters, 1, "Timed iterations to average (after warmup).");
DEFINE_bool(
    cuda_graph,
    false,
    "Enable CUDA graph for the decode method. CUDA only; single-session mode.");

namespace llm = ::executorch::extension::llm;
using ::executorch::runtime::Error;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_path.empty()) {
    ET_LOG(Error, "Must specify --model_path");
    return 1;
  }
  if (FLAGS_tokenizer_path.empty()) {
    ET_LOG(Error, "Must specify --tokenizer_path");
    return 1;
  }

  llm::Stats stats;

#ifdef EXECUTORCH_BUILD_CUDA
  {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
      stats.gpu_total_bytes = total;
      stats.gpu_free_before_load_bytes = free;
    }
  }
#endif

  stats.model_load_start_ms = llm::time_in_ms();

  // Build engine (reads tokenizer + metadata) and a session (loads weights and
  // the prefill/decode methods).
  llm::Qwen35MoEConfig config;
  config.model_path = FLAGS_model_path;
  config.data_path = FLAGS_data_path;
  config.tokenizer_path = FLAGS_tokenizer_path;
  config.enable_cuda_graph = FLAGS_cuda_graph;
#ifndef EXECUTORCH_BUILD_CUDA
  if (FLAGS_cuda_graph) {
    ET_LOG(Info, "--cuda_graph ignored on non-CUDA build");
  }
#endif

  printf("Loading methods...\n");
  auto engine_result = llm::Qwen35MoEEngine::create(config);
  if (engine_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to create Qwen3.5 MoE engine");
    return 1;
  }
  auto engine = std::move(engine_result.get());

  auto session_result = engine->create_session();
  if (session_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to create session");
    return 1;
  }
  auto session = std::move(session_result.get());

  stats.model_load_end_ms = llm::time_in_ms();

#ifdef EXECUTORCH_BUILD_CUDA
  {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
      stats.gpu_free_after_load_bytes = free;
    }
  }
#endif

  // Read prompt from file or flag.
  std::string prompt_text = FLAGS_prompt;
  if (!FLAGS_prompt_file.empty()) {
    std::ifstream f(FLAGS_prompt_file);
    if (!f.is_open()) {
      ET_LOG(
          Error, "Failed to open prompt file: %s", FLAGS_prompt_file.c_str());
      return 1;
    }
    prompt_text = std::string(
        (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  }

  // Encode prompt via the engine's tokenizer.
  auto encode_result = engine->tokenizer()->encode(prompt_text);
  if (!encode_result.ok()) {
    ET_LOG(Error, "Failed to encode prompt");
    return 1;
  }
  std::vector<uint64_t> prompt_tokens = std::move(*encode_result);
  const int64_t num_prompt_tokens = static_cast<int64_t>(prompt_tokens.size());
  printf("Prompt tokens: %" PRId64 "\n", num_prompt_tokens);

  stats.num_prompt_tokens = num_prompt_tokens;

  // Warmup + timed iterations on one loaded session (reset between). The first
  // FLAGS_warmup iterations are discarded; they let allocator growth and GPU
  // clock ramp settle so the timed iterations reflect steady state. Text is
  // printed only on the first iteration (coherence check).
  llm::SamplingConfig sampling;
  sampling.temperature = static_cast<float>(FLAGS_temperature);
  sampling.top_p = static_cast<float>(FLAGS_top_p);
  sampling.seed = static_cast<uint64_t>(FLAGS_seed);
  const int total_iters = FLAGS_warmup + std::max(1, FLAGS_num_iters);
  std::vector<double> prefill_tps_samples;
  std::vector<double> decode_tps_samples;
  double prefill_ms = 0.0;
  int64_t num_generated = 0;

  for (int iter = 0; iter < total_iters; ++iter) {
    if (iter > 0 && session->reset() != Error::Ok) {
      ET_LOG(Error, "Session reset failed before iteration %d", iter);
      return 1;
    }
    const bool measured = iter >= FLAGS_warmup;
    const bool print_text = (iter == 0);

    stats.inference_start_ms = llm::time_in_ms();
    if (session->prefill_tokens(prompt_tokens, &sampling) != Error::Ok) {
      ET_LOG(Error, "Prefill failed");
      return 1;
    }
    stats.prompt_eval_end_ms = llm::time_in_ms();
    stats.first_token_ms = stats.prompt_eval_end_ms;

    num_generated = 0;
    for (int32_t step = 0; step < FLAGS_max_new_tokens; step++) {
      auto step_result = session->decode_one(sampling);
      if (step_result.error() != Error::Ok) {
        ET_LOG(Error, "Decode step %d failed", step);
        return 1;
      }
      const auto& d = step_result.get();
      // A terminal step is the loop terminator, not generated output.
      if (d.is_terminal) {
        if (print_text) {
          printf("\n");
        }
        break;
      }
      num_generated++;
      if (step == 0) {
        stats.first_token_ms = llm::time_in_ms();
      }
      if (print_text && !d.text_piece.empty()) {
        fwrite(d.text_piece.data(), 1, d.text_piece.size(), stdout);
        fflush(stdout);
      }
    }
    stats.inference_end_ms = llm::time_in_ms();
    stats.num_generated_tokens = num_generated;

    prefill_ms = (double)(stats.prompt_eval_end_ms - stats.inference_start_ms);
    const double decode_ms_iter =
        (double)(stats.inference_end_ms - stats.prompt_eval_end_ms);
    const double pf_tps =
        num_prompt_tokens / prefill_ms * stats.SCALING_FACTOR_UNITS_PER_SECOND;
    const double dc_tps =
        num_generated / decode_ms_iter * stats.SCALING_FACTOR_UNITS_PER_SECOND;
    printf(
        "[iter %d%s] prefill %.1f tok/s (%" PRId64
        " tok, %.1f ms) | "
        "decode %.1f tok/s (%" PRId64 " tok, %.1f ms)\n",
        iter,
        measured ? "" : " warmup",
        pf_tps,
        num_prompt_tokens,
        prefill_ms,
        dc_tps,
        num_generated,
        decode_ms_iter);
    if (measured) {
      prefill_tps_samples.push_back(pf_tps);
      decode_tps_samples.push_back(dc_tps);
    }
  }

  printf("\n");

#ifdef EXECUTORCH_BUILD_CUDA
  {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
      stats.gpu_free_after_generate_bytes = free;
      size_t min_free = free;
      if (stats.gpu_free_before_load_bytes != static_cast<uint64_t>(-1)) {
        min_free = std::min(min_free, (size_t)stats.gpu_free_before_load_bytes);
      }
      if (stats.gpu_free_after_load_bytes != static_cast<uint64_t>(-1)) {
        min_free = std::min(min_free, (size_t)stats.gpu_free_after_load_bytes);
      }
      stats.gpu_peak_usage_mb = (double)(total - min_free) / 1024.0 / 1024.0;
    }
  }
#endif

  printf("\n");
  const double decode_ms =
      (double)(stats.inference_end_ms - stats.prompt_eval_end_ms);
  printf(
      "Prefill: %" PRId64 " tokens in %.1f ms (%.1f tok/s)\n",
      num_prompt_tokens,
      prefill_ms,
      num_prompt_tokens / prefill_ms * stats.SCALING_FACTOR_UNITS_PER_SECOND);
  printf(
      "Decode: %" PRId64 " tokens in %.1f ms (%.1f tok/s)\n",
      num_generated,
      decode_ms,
      num_generated / decode_ms * stats.SCALING_FACTOR_UNITS_PER_SECOND);
  printf("Prompt tokens: %" PRId64 "\n", num_prompt_tokens);

  printf("PyTorchObserver %s\n", llm::stats_to_json_string(stats).c_str());

  const double ms_per_s = stats.SCALING_FACTOR_UNITS_PER_SECOND;
  const double model_load_s =
      (double)(stats.model_load_end_ms - stats.model_load_start_ms) / ms_per_s;
  const double inference_time_ms =
      (double)(stats.inference_end_ms - stats.inference_start_ms);
  const double prompt_eval_ms =
      (double)(stats.prompt_eval_end_ms - stats.inference_start_ms);
  const double eval_ms =
      (double)(stats.inference_end_ms - stats.prompt_eval_end_ms);
  const double ttft_s =
      (double)(stats.first_token_ms - stats.inference_start_ms) / ms_per_s;
  const double sampling_s = (double)stats.aggregate_sampling_time_ms / ms_per_s;

  printf("\n");
  printf(
      "\tPrompt Tokens: %" PRId64 "    Generated Tokens: %" PRId64 "\n",
      stats.num_prompt_tokens,
      stats.num_generated_tokens);
  printf("\tModel Load Time:\t\t%f (seconds)\n", model_load_s);
  printf(
      "\tTotal inference time:\t\t%f (seconds)\t\t Rate: \t%f (tokens/second)\n",
      inference_time_ms / ms_per_s,
      stats.num_generated_tokens / inference_time_ms * ms_per_s);
  printf(
      "\t\tPrompt evaluation:\t%f (seconds)\t\t Rate: \t%f (tokens/second)\n",
      prompt_eval_ms / ms_per_s,
      stats.num_prompt_tokens / prompt_eval_ms * ms_per_s);
  printf(
      "\t\tGenerated %" PRId64
      " tokens:\t%f (seconds)\t\t Rate: \t%f (tokens/second)\n",
      stats.num_generated_tokens,
      eval_ms / ms_per_s,
      stats.num_generated_tokens / eval_ms * ms_per_s);
  printf("\tTime to first generated token:\t%f (seconds)\n", ttft_s);
  printf(
      "\tSampling time over %" PRId64 " tokens:\t%f (seconds)\n",
      stats.num_prompt_tokens + stats.num_generated_tokens,
      sampling_s);

  if (stats.gpu_total_bytes != static_cast<uint64_t>(-1)) {
    printf(
        "\tGPU total memory: %.2f MB\n",
        stats.gpu_total_bytes / 1024.0 / 1024.0);
    if (stats.gpu_free_before_load_bytes != static_cast<uint64_t>(-1)) {
      printf(
          "\tGPU free before load: %.2f MB\n",
          stats.gpu_free_before_load_bytes / 1024.0 / 1024.0);
    }
    if (stats.gpu_free_after_load_bytes != static_cast<uint64_t>(-1)) {
      printf(
          "\tGPU free after load: %.2f MB\n",
          stats.gpu_free_after_load_bytes / 1024.0 / 1024.0);
    }
    if (stats.gpu_free_after_generate_bytes != static_cast<uint64_t>(-1)) {
      printf(
          "\tGPU free after generate: %.2f MB\n",
          stats.gpu_free_after_generate_bytes / 1024.0 / 1024.0);
    }
    if (stats.gpu_peak_usage_mb >= 0.0) {
      printf("\tGPU peak usage: %.2f MB\n", stats.gpu_peak_usage_mb);
    }
  }

  if (!prefill_tps_samples.empty()) {
    auto mean = [](const std::vector<double>& v) {
      double s = 0.0;
      for (double x : v) {
        s += x;
      }
      return s / v.size();
    };
    printf(
        "\n=== mean over %zu timed iter(s) (warmup %d) | prompt %" PRId64
        ", gen %" PRId64 " ===\n",
        prefill_tps_samples.size(),
        FLAGS_warmup,
        num_prompt_tokens,
        num_generated);
    printf("\tPrefill: %.1f tok/s\n", mean(prefill_tps_samples));
    printf("\tDecode:  %.1f tok/s\n", mean(decode_tps_samples));
  }

  return 0;
}
