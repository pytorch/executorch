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
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

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
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");

namespace llm = ::executorch::extension::llm;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

using SizesType = executorch::aten::SizesType;

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

  // GPU memory before load
  size_t gpu_free_bytes, gpu_total_bytes;
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_total_bytes = gpu_total_bytes;
  stats.gpu_free_before_load_bytes = gpu_free_bytes;

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

  // Create Module with share_memory_arenas=true so prefill and forward
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

  printf("Loading methods...\n");

  // Try loading both methods; fall back to single "forward" method
  bool dual_method = true;
  std::string prefill_method = "prefill";
  auto err = module->load_method("prefill");
  if (err != Error::Ok) {
    // Try "forward" for single-method export
    err = module->load_method("forward");
    if (err != Error::Ok) {
      ET_LOG(Error, "Failed to load prefill/forward method");
      return 1;
    }
    prefill_method = "forward";
    dual_method = false;
    printf("Using single-method mode (forward)\n");
  }
  if (dual_method) {
    err = module->load_method("decode");
    if (err != Error::Ok) {
      ET_LOG(Error, "Failed to load decode method");
      return 1;
    }
  }

  stats.model_load_end_ms = llm::time_in_ms();

  // GPU memory after load
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_load_bytes = gpu_free_bytes;

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
  printf("Prompt tokens: %ld\n", num_prompt_tokens);

  stats.num_prompt_tokens = num_prompt_tokens;
  stats.inference_start_ms = llm::time_in_ms();

  // ---------------------------------------------------------------
  // Prefill or decode-only
  // ---------------------------------------------------------------
  auto S = [](int64_t v) -> SizesType { return static_cast<SizesType>(v); };

  uint64_t cur_token = 0;

  // Use prefill method for T>=2, decode method for T=1
  // (prefill was exported with min seq_len=2)
  std::string run_method = prefill_method;
  if (dual_method && num_prompt_tokens == 1) {
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

  auto prefill_result = module->execute(run_method, prefill_inputs);
  if (prefill_result.error() != Error::Ok) {
    ET_LOG(Error, "Prefill failed");
    return 1;
  }
  auto& prefill_outputs = prefill_result.get();

  auto logits_tensor = prefill_outputs[0].toTensor();
  auto logits_ptr =
      std::make_shared<executorch::aten::Tensor>(std::move(logits_tensor));
  cur_token = llm::logits_to_token(*logits_ptr, FLAGS_temperature);

  stats.prompt_eval_end_ms = llm::time_in_ms();

  double prefill_ms =
      (double)(stats.prompt_eval_end_ms - stats.inference_start_ms);
  printf(
      "Prefill: %ld tokens in %.1f ms (%.1f tok/s)\n",
      num_prompt_tokens,
      prefill_ms,
      num_prompt_tokens * 1000.0 / prefill_ms);

#ifdef EXECUTORCH_BUILD_CUDA
  // Synchronize CUDA device to ensure prefill's writes to shared mutable
  // buffers (KV cache, conv_state, recurrent_state) are visible to the
  // decode method, which may run on a different CUDA stream.
  cudaDeviceSynchronize();
#endif

  if (!dual_method) {
    printf("Single-method mode: skipping decode\n");
    return 0;
  }

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

    auto decode_result = module->execute("decode", decode_inputs);
    if (decode_result.error() != Error::Ok) {
      ET_LOG(Error, "Decode step %d failed", step);
      return 1;
    }
    auto& decode_outputs = decode_result.get();

    auto step_logits = decode_outputs[0].toTensor();
    auto step_logits_ptr =
        std::make_shared<executorch::aten::Tensor>(std::move(step_logits));

    prev_token = cur_token;
    stats.on_sampling_begin();
    cur_token = llm::logits_to_token(*step_logits_ptr, FLAGS_temperature);
    stats.on_sampling_end();

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

  double decode_ms =
      (double)(stats.inference_end_ms - stats.prompt_eval_end_ms);
  printf(
      "Decode: %ld tokens in %.1f ms (%.1f tok/s)\n",
      num_generated,
      decode_ms,
      num_generated * 1000.0 / decode_ms);
  printf("Prompt tokens: %ld\n", num_prompt_tokens);

  // GPU memory after generation
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_generate_bytes = gpu_free_bytes;
  stats.gpu_peak_usage_mb =
      (stats.gpu_total_bytes - gpu_free_bytes) / 1024.0 / 1024.0;

  llm::print_report(stats);

  return 0;
}
