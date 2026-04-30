/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <string>
#include <vector>

#ifdef EXECUTORCH_BUILD_CUDA
#include <cuda_runtime.h>
#else
#include <executorch/extension/llm/sampler/util.h>
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
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");
DEFINE_bool(
    cuda_graph,
    false,
    "Enable CUDA graph for decode method. CUDA only.");

namespace llm = ::executorch::extension::llm;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

using SizesType = executorch::aten::SizesType;

// Convert a model output tensor to the next sampled token id.
//
// On the CUDA build, the model fuses the sampler in (see sampler.py /
// Qwen35MoE.forward) and returns a single sampled token id as a [B, 1]
// float tensor; we just copy that scalar back from device.
//
// On non-CUDA builds (Metal / MLX / CPU), the model returns raw logits
// of shape [B, T, V] in the model dtype (typically bf16). We sample on
// CPU via the shared `llm::logits_to_token` helper, which accepts a
// temperature (0 = greedy / argmax).
static uint64_t read_token(const executorch::aten::Tensor& output) {
#ifdef EXECUTORCH_BUILD_CUDA
  const void* ptr = output.const_data_ptr();

  cudaPointerAttributes attrs;
  bool on_device = cudaPointerGetAttributes(&attrs, ptr) == cudaSuccess &&
      attrs.type == cudaMemoryTypeDevice;

  float val;
  if (on_device) {
    cudaError_t err =
        cudaMemcpy(&val, ptr, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      ET_LOG(
          Error,
          "read_token: cudaMemcpy D2H failed: %s",
          cudaGetErrorString(err));
      return 0;
    }
  } else {
    memcpy(&val, ptr, sizeof(float));
  }
  return static_cast<uint64_t>(val);
#else
  // logits_to_token handles 2D / 3D logits and Float / Half / BFloat16 /
  // UInt16 dtypes. Negative temperatures are clamped to 0 (greedy).
  const float temp =
      FLAGS_temperature <= 0.0 ? 0.0f : static_cast<float>(FLAGS_temperature);
  return static_cast<uint64_t>(llm::logits_to_token(output, temp));
#endif
}

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
  // GPU memory before load
  size_t gpu_free_bytes = 0, gpu_total_bytes = 0;
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_total_bytes = gpu_total_bytes;
  stats.gpu_free_before_load_bytes = gpu_free_bytes;
#endif

  stats.model_load_start_ms = llm::time_in_ms();

  // Load tokenizer
  auto tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  auto tok_status = tokenizer->load(FLAGS_tokenizer_path);
  if (tok_status != tokenizers::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to load tokenizer from %s",
        FLAGS_tokenizer_path.c_str());
    return 1;
  }

  // GPU memory: before load
  {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
      stats.gpu_total_bytes = total;
      stats.gpu_free_before_load_bytes = free;
    }
  }

  stats.model_load_start_ms = llm::time_in_ms();

  // Create Module with share_memory_arenas=true so prefill and decode
  // share mutable buffers (KV cache, conv_state, recurrent_state).
  std::vector<std::string> data_files;
  if (!FLAGS_data_path.empty()) {
    data_files.push_back(FLAGS_data_path);
  }
  auto module = std::make_unique<Module>(
      FLAGS_model_path,
      data_files,
      Module::LoadMode::File,
      /*event_tracer=*/nullptr,
      /*memory_allocator=*/nullptr,
      /*temp_allocator=*/nullptr,
      /*share_memory_arenas=*/true);

  // Get metadata
  auto metadata_result = llm::get_llm_metadata(tokenizer.get(), module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to get metadata from model");
    return 1;
  }
  auto metadata = metadata_result.get();

#ifdef EXECUTORCH_BUILD_CUDA
  // Set CUDA graph option if requested (must be before load_method)
  if (FLAGS_cuda_graph) {
    executorch::runtime::BackendOptions<2> cuda_opts;
    cuda_opts.set_option("enable_cuda_graph_for_method", "decode");
    executorch::runtime::set_option("CudaBackend", cuda_opts.view());
    printf("CUDA graph enabled for decode method\n");
  }
#else
  if (FLAGS_cuda_graph) {
    ET_LOG(Info, "--cuda_graph ignored on non-CUDA build");
  }
#endif

  printf("Loading methods...\n");

#ifdef EXECUTORCH_BUILD_CUDA
  // Enable cross-method per-FQN weight sharing in the CUDA backend so that
  // prefill and decode (which share KV cache and other mutable buffers /
  // weights) avoid duplicate GPU allocations. This is critical for fitting
  // Qwen 3.5 MoE on a single GPU. MUST be set BEFORE load_method, since the
  // backend reads this flag during init() to decide between the per-weight
  // cache path and the legacy per-method blob load.
  {
    executorch::runtime::BackendOptions<1> backend_options;
    auto set_err =
        backend_options.set_option("weight_sharing_across_methods", true);
    if (set_err != Error::Ok) {
      ET_LOG(
          Error,
          "Failed to construct weight_sharing_across_methods option: %d",
          static_cast<int>(set_err));
      return 1;
    }
    const auto opt_err =
        executorch::runtime::set_option("CudaBackend", backend_options.view());
    if (opt_err != Error::Ok) {
      ET_LOG(
          Error,
          "Failed to enable weight_sharing_across_methods: %d",
          static_cast<int>(opt_err));
      return 1;
    }
  }
#endif

  auto err = module->load_method("prefill");
  if (err != Error::Ok) {
    ET_LOG(Error, "Failed to load prefill method");
    return 1;
  }
  err = module->load_method("decode");
  if (err != Error::Ok) {
    ET_LOG(Error, "Failed to load decode method");
    return 1;
  }

  stats.model_load_end_ms = llm::time_in_ms();

  // GPU memory: after load
  {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
      stats.gpu_free_after_load_bytes = free;
    }
  }

  // Get EOS ids
  auto eos_ids = llm::get_eos_ids(tokenizer.get(), module.get());

  // Read prompt from file or flag
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

  // Encode prompt
  auto encode_result = tokenizer->encode(prompt_text);
  if (!encode_result.ok()) {
    ET_LOG(Error, "Failed to encode prompt");
    return 1;
  }
  auto prompt_tokens = std::move(*encode_result);
  int64_t num_prompt_tokens = prompt_tokens.size();
  printf("Prompt tokens: %" PRId64 "\n", num_prompt_tokens);

  stats.num_prompt_tokens = num_prompt_tokens;
  stats.inference_start_ms = llm::time_in_ms();

  // ---------------------------------------------------------------
  // Sampling tensors (shared between prefill and decode)
  // ---------------------------------------------------------------
  auto S = [](int64_t v) -> SizesType { return static_cast<SizesType>(v); };

#ifdef EXECUTORCH_BUILD_CUDA
  // CUDA build: model fuses the sampler in. Pass a temperature tensor as
  // a third input. Use a very small temperature for greedy to avoid
  // division by zero while keeping the Gumbel noise negligible relative
  // to logit differences.
  float temp_val =
      FLAGS_temperature <= 0.0 ? 1e-6f : static_cast<float>(FLAGS_temperature);
  auto temp_tensor =
      from_blob(&temp_val, {1}, executorch::aten::ScalarType::Float);
#endif

  stats.inference_start_ms = llm::time_in_ms();
  stats.num_prompt_tokens = num_prompt_tokens;

  // ---------------------------------------------------------------
  // Prefill
  // ---------------------------------------------------------------
  uint64_t cur_token = 0;

  // Use prefill method for T>=2, decode method for T=1
  // (prefill was exported with min seq_len=2)
  std::string run_method = "prefill";
  if (num_prompt_tokens == 1) {
    run_method = "decode";
  }

  std::vector<int64_t> pos_data(num_prompt_tokens);
  for (int64_t i = 0; i < num_prompt_tokens; i++) {
    pos_data[i] = i;
  }
  std::vector<int64_t> token_data(prompt_tokens.begin(), prompt_tokens.end());
  auto tokens_tensor = from_blob(
      token_data.data(),
      {1, S(num_prompt_tokens)},
      executorch::aten::ScalarType::Long);
  auto pos_tensor = from_blob(
      pos_data.data(),
      {S(num_prompt_tokens)},
      executorch::aten::ScalarType::Long);

  std::vector<EValue> prefill_inputs;
  prefill_inputs.push_back(tokens_tensor);
  prefill_inputs.push_back(pos_tensor);
#ifdef EXECUTORCH_BUILD_CUDA
  prefill_inputs.push_back(temp_tensor);
#endif

  auto prefill_result = module->execute(run_method, prefill_inputs);
  if (prefill_result.error() != Error::Ok) {
    ET_LOG(Error, "Prefill failed");
    return 1;
  }
  auto& prefill_outputs = prefill_result.get();

  cur_token = read_token(prefill_outputs[0].toTensor());

  stats.prompt_eval_end_ms = llm::time_in_ms();
  stats.first_token_ms = stats.prompt_eval_end_ms;
  double prefill_ms =
      (double)(stats.prompt_eval_end_ms - stats.inference_start_ms);
  printf(
      "Prefill: %" PRId64 " tokens in %.1f ms (%.1f tok/s)\n",
      num_prompt_tokens,
      prefill_ms,
      num_prompt_tokens / prefill_ms * stats.SCALING_FACTOR_UNITS_PER_SECOND);

#ifdef EXECUTORCH_BUILD_CUDA
  // Synchronize CUDA device to ensure prefill's writes to shared mutable
  // buffers (KV cache, conv_state, recurrent_state) are visible to the
  // decode method, which may run on a different CUDA stream.
  cudaDeviceSynchronize();
#endif

  // ---------------------------------------------------------------
  // Decode — generate tokens one at a time
  // ---------------------------------------------------------------
  int64_t pos = num_prompt_tokens;
  uint64_t prev_token;

  std::vector<int64_t> decode_token_data = {static_cast<int64_t>(cur_token)};
  std::vector<int64_t> decode_pos_data = {pos};
  auto decode_tokens = from_blob(
      decode_token_data.data(), {1, 1}, executorch::aten::ScalarType::Long);
  auto decode_pos = from_blob(
      decode_pos_data.data(), {1}, executorch::aten::ScalarType::Long);

  for (int32_t step = 0; step < FLAGS_max_new_tokens; step++) {
    decode_token_data[0] = static_cast<int64_t>(cur_token);
    decode_pos_data[0] = pos;

    std::vector<EValue> decode_inputs;
    decode_inputs.push_back(EValue(decode_tokens));
    decode_inputs.push_back(EValue(decode_pos));
#ifdef EXECUTORCH_BUILD_CUDA
    decode_inputs.push_back(EValue(temp_tensor));
#endif

    auto decode_result = module->execute("decode", decode_inputs);
    if (decode_result.error() != Error::Ok) {
      ET_LOG(Error, "Decode step %d failed", step);
      return 1;
    }
    auto& decode_outputs = decode_result.get();

    prev_token = cur_token;
    cur_token = read_token(decode_outputs[0].toTensor());

    if (step == 0) {
      stats.first_token_ms = llm::time_in_ms();
    }

    pos++;

    auto decode_str = tokenizer->decode(prev_token, cur_token);
    if (decode_str.ok()) {
      printf("%s", decode_str->c_str());
      fflush(stdout);
    }

    if (eos_ids.find(cur_token) != eos_ids.end()) {
      printf("\n");
      break;
    }
  }

  stats.inference_end_ms = llm::time_in_ms();

  printf("\n");
  int64_t num_generated = pos - num_prompt_tokens;
  stats.num_generated_tokens = num_generated;

  // GPU memory: after generate + peak usage
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

  printf("\n");

  double decode_ms =
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

  // Structured stats report (matches stats.h print_report)
  printf("PyTorchObserver %s\n", llm::stats_to_json_string(stats).c_str());

  double ms_per_s = stats.SCALING_FACTOR_UNITS_PER_SECOND;

  double model_load_s =
      (double)(stats.model_load_end_ms - stats.model_load_start_ms) / ms_per_s;
  double inference_time_ms =
      (double)(stats.inference_end_ms - stats.inference_start_ms);
  double prompt_eval_ms =
      (double)(stats.prompt_eval_end_ms - stats.inference_start_ms);
  double eval_ms = (double)(stats.inference_end_ms - stats.prompt_eval_end_ms);
  double ttft_s =
      (double)(stats.first_token_ms - stats.inference_start_ms) / ms_per_s;
  double sampling_s = (double)stats.aggregate_sampling_time_ms / ms_per_s;

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

  // GPU memory reporting
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

  return 0;
}
