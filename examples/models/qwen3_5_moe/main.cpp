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
DEFINE_bool(cuda_graph, false, "Enable CUDA graph for decode method.");
DEFINE_int64(
    top_k,
    -1,
    "Top-k sampling cutoff (<=0 = no-op default of vocab_size, keeps all tokens).");
DEFINE_double(
    top_p,
    1.0,
    "Top-p (nucleus) sampling threshold. 1.0 = no-op (keeps full nucleus).");

namespace llm = ::executorch::extension::llm;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

using SizesType = executorch::aten::SizesType;

// Read a sampled token from the model output tensor [B, 1].
// The model performs Gumbel-max sampling on-device and returns a single
// float token ID. This function copies it from GPU and casts to uint64.
static uint64_t read_token(const executorch::aten::Tensor& output) {
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

  // Set CUDA graph option if requested (must be before load_method)
  if (FLAGS_cuda_graph) {
    executorch::runtime::BackendOptions<2> cuda_opts;
    cuda_opts.set_option("enable_cuda_graph_for_method", "decode");
    executorch::runtime::set_option("CudaBackend", cuda_opts.view());
    printf("CUDA graph enabled for decode method\n");
  }

  printf("Loading methods...\n");

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
  // Sampling tensors (shared between prefill and decode)
  // ---------------------------------------------------------------
  auto S = [](int64_t v) -> SizesType { return static_cast<SizesType>(v); };

  // Use a very small temperature for greedy to avoid division by zero
  // while keeping the Gumbel noise negligible relative to logit differences.
  float temp_val =
      FLAGS_temperature <= 0.0 ? 1e-6f : static_cast<float>(FLAGS_temperature);
  auto temp_tensor =
      from_blob(&temp_val, {1}, executorch::aten::ScalarType::Float);

  // top_k / top_p are 0-D scalar tensors matching the export-time signature
  // (see examples/models/qwen3_5_moe/export.py). The default flag values
  // (top_k = vocab_size, top_p = 1.0) are mathematical no-ops: the sort+
  // scatter subgraph still runs (it was traced into the graph at export
  // time), but produces all-False filter masks so logits pass through
  // unchanged. Override at runtime to enable real filtering.
  int64_t vocab_size = metadata.count(llm::kVocabSize)
      ? metadata[llm::kVocabSize]
      : static_cast<int64_t>(tokenizer->vocab_size());
  int64_t top_k_val = (FLAGS_top_k <= 0) ? vocab_size : FLAGS_top_k;
  float top_p_val = static_cast<float>(FLAGS_top_p);
  auto top_k_tensor =
      from_blob(&top_k_val, {}, executorch::aten::ScalarType::Long);
  auto top_p_tensor =
      from_blob(&top_p_val, {}, executorch::aten::ScalarType::Float);

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
  prefill_inputs.push_back(temp_tensor);
  prefill_inputs.push_back(top_k_tensor);
  prefill_inputs.push_back(top_p_tensor);

  auto prefill_result = module->execute(run_method, prefill_inputs);
  if (prefill_result.error() != Error::Ok) {
    ET_LOG(Error, "Prefill failed");
    return 1;
  }
  auto& prefill_outputs = prefill_result.get();

  cur_token = read_token(prefill_outputs[0].toTensor());

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
    decode_inputs.push_back(EValue(temp_tensor));
    decode_inputs.push_back(EValue(top_k_tensor));
    decode_inputs.push_back(EValue(top_p_tensor));

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
