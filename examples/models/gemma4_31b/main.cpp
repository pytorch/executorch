/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Gemma 4 31B-IT runner for the CUDA ExecuTorch backend.
//
// Drives the prefill + decode methods produced by export.py.
// The exported model performs Gumbel-max sampling on-device and returns a
// single float token ID per call, so this runner only has to feed tokens
// in and decode them via the HuggingFace tokenizer.

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

#include <cinttypes>
#include <fstream>
#include <string>
#include <vector>

#ifdef EXECUTORCH_BUILD_CUDA
#include <cuda_runtime.h>
#endif

DEFINE_string(model_path, "", "Path to model.pte.");
DEFINE_string(data_path, "", "Path to model.ptd (CUDA tensor data).");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(prompt, "Hello", "Prompt text.");
DEFINE_string(
    prompt_file,
    "",
    "Optional path to a file with the prompt text (overrides --prompt).");
DEFINE_double(temperature, 0.8, "Sampling temperature (0 = near-greedy).");
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");
DEFINE_bool(
    cuda_graph,
    false,
    "Enable CUDA graph capture for the decode method.");

namespace llm = ::executorch::extension::llm;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

using SizesType = executorch::aten::SizesType;

// The model performs sampling on-device and returns a [B, 1] float tensor
// holding a token ID. Copy it to host and convert to uint64.
static uint64_t read_token(const executorch::aten::Tensor& output) {
  const void* ptr = output.const_data_ptr();
  float val = 0.0f;

#ifdef EXECUTORCH_BUILD_CUDA
  cudaPointerAttributes attrs{};
  bool on_device = cudaPointerGetAttributes(&attrs, ptr) == cudaSuccess &&
      attrs.type == cudaMemoryTypeDevice;
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
#else
  memcpy(&val, ptr, sizeof(float));
#endif

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

#ifdef EXECUTORCH_BUILD_CUDA
  size_t gpu_free_bytes = 0, gpu_total_bytes = 0;
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_total_bytes = gpu_total_bytes;
  stats.gpu_free_before_load_bytes = gpu_free_bytes;
#endif

  stats.model_load_start_ms = llm::time_in_ms();

  // Tokenizer
  auto tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  if (tokenizer->load(FLAGS_tokenizer_path) != tokenizers::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to load tokenizer from %s",
        FLAGS_tokenizer_path.c_str());
    return 1;
  }

  // Module: share_memory_arenas=true so prefill and decode see the same
  // KV-cache memory (we exported with share_mutable_buffers=True).
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

  auto metadata_result = llm::get_llm_metadata(tokenizer.get(), module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to read model metadata");
    return 1;
  }

  if (FLAGS_cuda_graph) {
    executorch::runtime::BackendOptions<2> cuda_opts;
    cuda_opts.set_option("enable_cuda_graph_for_method", "decode");
    executorch::runtime::set_option("CudaBackend", cuda_opts.view());
    printf("CUDA graph enabled for decode method\n");
  }

  // Cross-method per-FQN weight sharing: prefill + decode share the same
  // weight tensors and (more importantly) the same KV-cache buffers, so
  // without this flag we would allocate them twice. MUST be set before
  // load_method.
  {
    executorch::runtime::BackendOptions<1> backend_options;
    if (backend_options.set_option("weight_sharing_across_methods", true) !=
            Error::Ok ||
        executorch::runtime::set_option(
            "CudaBackend", backend_options.view()) != Error::Ok) {
      ET_LOG(Error, "Failed to enable weight_sharing_across_methods");
      return 1;
    }
  }

  printf("Loading methods...\n");
  if (module->load_method("prefill") != Error::Ok) {
    ET_LOG(Error, "Failed to load prefill method");
    return 1;
  }
  if (module->load_method("decode") != Error::Ok) {
    ET_LOG(Error, "Failed to load decode method");
    return 1;
  }
  stats.model_load_end_ms = llm::time_in_ms();

#ifdef EXECUTORCH_BUILD_CUDA
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_load_bytes = gpu_free_bytes;
#endif

  auto eos_ids = llm::get_eos_ids(tokenizer.get(), module.get());

  std::string prompt_text = FLAGS_prompt;
  if (!FLAGS_prompt_file.empty()) {
    std::ifstream f(FLAGS_prompt_file);
    if (!f.is_open()) {
      ET_LOG(
          Error, "Failed to open prompt file: %s", FLAGS_prompt_file.c_str());
      return 1;
    }
    prompt_text.assign(
        (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  }

  auto encode_result = tokenizer->encode(prompt_text);
  if (!encode_result.ok()) {
    ET_LOG(Error, "Failed to encode prompt");
    return 1;
  }
  auto prompt_tokens = std::move(*encode_result);
  int64_t num_prompt_tokens = static_cast<int64_t>(prompt_tokens.size());
  printf("Prompt tokens: %" PRId64 "\n", num_prompt_tokens);
  stats.num_prompt_tokens = num_prompt_tokens;

  stats.inference_start_ms = llm::time_in_ms();

  auto S = [](int64_t v) -> SizesType { return static_cast<SizesType>(v); };

  // Temperature: clamp 0 to a tiny epsilon so the divide in the exported
  // sampler stays well-defined. Gumbel noise then becomes negligible
  // relative to logit gaps and we get effectively-greedy sampling.
  float temp_val =
      FLAGS_temperature <= 0.0 ? 1e-6f : static_cast<float>(FLAGS_temperature);
  auto temp_tensor =
      from_blob(&temp_val, {1}, executorch::aten::ScalarType::Float);

  // ---------------------------------------------------------------
  // Prefill
  // ---------------------------------------------------------------
  std::string run_method = "prefill";
  if (num_prompt_tokens == 1) {
    // prefill was exported with min seq_len=2; decode handles T==1.
    run_method = "decode";
  }

  std::vector<int64_t> token_data(prompt_tokens.begin(), prompt_tokens.end());
  std::vector<int64_t> pos_data(num_prompt_tokens);
  for (int64_t i = 0; i < num_prompt_tokens; i++) {
    pos_data[i] = i;
  }
  auto tokens_tensor = from_blob(
      token_data.data(),
      {1, S(num_prompt_tokens)},
      executorch::aten::ScalarType::Long);
  auto pos_tensor = from_blob(
      pos_data.data(),
      {S(num_prompt_tokens)},
      executorch::aten::ScalarType::Long);

  std::vector<EValue> prefill_inputs = {
      EValue(tokens_tensor),
      EValue(pos_tensor),
      EValue(temp_tensor),
  };

  auto prefill_result = module->execute(run_method, prefill_inputs);
  if (prefill_result.error() != Error::Ok) {
    ET_LOG(Error, "%s failed", run_method.c_str());
    return 1;
  }
  uint64_t cur_token = read_token(prefill_result.get()[0].toTensor());

  stats.prompt_eval_end_ms = llm::time_in_ms();
  double prefill_ms =
      static_cast<double>(stats.prompt_eval_end_ms - stats.inference_start_ms);
  printf(
      "Prefill: %" PRId64 " tokens in %.1f ms (%.1f tok/s)\n",
      num_prompt_tokens,
      prefill_ms,
      num_prompt_tokens * 1000.0 / prefill_ms);

#ifdef EXECUTORCH_BUILD_CUDA
  // Make prefill's writes to the shared KV cache visible before decode
  // potentially runs on a different stream.
  cudaDeviceSynchronize();
#endif

  // ---------------------------------------------------------------
  // Decode loop
  // ---------------------------------------------------------------
  int64_t pos = num_prompt_tokens;
  std::vector<int64_t> decode_token_data = {static_cast<int64_t>(cur_token)};
  std::vector<int64_t> decode_pos_data = {pos};
  auto decode_tokens = from_blob(
      decode_token_data.data(), {1, 1}, executorch::aten::ScalarType::Long);
  auto decode_pos = from_blob(
      decode_pos_data.data(), {1}, executorch::aten::ScalarType::Long);

  uint64_t prev_token = cur_token;
  for (int32_t step = 0; step < FLAGS_max_new_tokens; step++) {
    decode_token_data[0] = static_cast<int64_t>(cur_token);
    decode_pos_data[0] = pos;

    std::vector<EValue> decode_inputs = {
        EValue(decode_tokens),
        EValue(decode_pos),
        EValue(temp_tensor),
    };

    auto decode_result = module->execute("decode", decode_inputs);
    if (decode_result.error() != Error::Ok) {
      ET_LOG(Error, "Decode step %d failed", step);
      return 1;
    }

    prev_token = cur_token;
    cur_token = read_token(decode_result.get()[0].toTensor());

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
      static_cast<double>(stats.inference_end_ms - stats.prompt_eval_end_ms);
  printf(
      "Decode: %" PRId64 " tokens in %.1f ms (%.1f tok/s)\n",
      num_generated,
      decode_ms,
      num_generated * 1000.0 / decode_ms);
  printf("Prompt tokens: %" PRId64 "\n", num_prompt_tokens);

#ifdef EXECUTORCH_BUILD_CUDA
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_generate_bytes = gpu_free_bytes;
  stats.gpu_peak_usage_mb =
      (stats.gpu_total_bytes - gpu_free_bytes) / 1024.0 / 1024.0;
#endif

  llm::print_report(stats);
  return 0;
}
