/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Gemma 4 31B-IT runner for ExecuTorch. Supports two backends:
//   CUDA  — exports ``prefill`` (T>=2, dynamic) + ``decode`` (T=1, static)
//           methods sharing KV-cache buffers; on-device Gumbel-max sampling
//           with temperature passed as a third input; returns a scalar
//           float token id.
//   MLX   — exports a single ``forward`` method with dynamic seq_len;
//           returns last-token logits; the runner samples on the host via
//           ``llm::logits_to_token`` with the same temperature semantics.

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
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
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

using SizesType = executorch::aten::SizesType;

// Read a sampled token ID from a scalar float output (CUDA path).
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

  return static_cast<uint64_t>(llrintf(val));
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

  // Module
  std::vector<std::string> data_files;
  if (!FLAGS_data_path.empty()) {
    data_files.push_back(FLAGS_data_path);
  }
  auto module = std::make_unique<Module>(
      FLAGS_model_path,
      data_files,
      Module::LoadMode::MmapUseMlockIgnoreErrors,
      /*event_tracer=*/nullptr,
      /*memory_allocator=*/nullptr,
      /*temp_allocator=*/nullptr,
      /*share_memory_arenas=*/true);

  // Get metadata
  auto metadata_result = llm::get_llm_metadata(tokenizer.get(), module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to read model metadata");
    return 1;
  }

  int64_t max_prefill_chunk = (*metadata_result)[llm::kMaxSeqLen] - 1;
  {
    auto get_result = module->get("get_max_prefill_chunk");
    if (get_result.ok()) {
      max_prefill_chunk = get_result->toScalar().to<int64_t>();
    }
  }

  auto S = [](int64_t v) -> SizesType { return static_cast<SizesType>(v); };

  float temp_val =
      FLAGS_temperature <= 0.0 ? 1e-6f : static_cast<float>(FLAGS_temperature);

#ifdef EXECUTORCH_BUILD_CUDA
  if (FLAGS_cuda_graph) {
    executorch::runtime::BackendOptions<2> cuda_opts;
    cuda_opts.set_option("enable_cuda_graph_for_method", "decode");
    executorch::runtime::set_option("CudaBackend", cuda_opts.view());
    printf("CUDA graph enabled for decode method\n");
  }
  {
    executorch::runtime::BackendOptions<1> backend_options;
    auto set_err =
        backend_options.set_option("weight_sharing_across_methods", true);
    if (set_err != Error::Ok) {
      ET_LOG(
          Error,
          "Failed to set weight_sharing_across_methods: %d",
          static_cast<int>(set_err));
      return 1;
    }
    auto opt_err =
        executorch::runtime::set_option("CudaBackend", backend_options.view());
    if (opt_err != Error::Ok) {
      ET_LOG(
          Error,
          "Failed to enable weight_sharing_across_methods: %d",
          static_cast<int>(opt_err));
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
  auto temp_tensor =
      from_blob(&temp_val, {1}, executorch::aten::ScalarType::Float);
#else
  if (FLAGS_cuda_graph) {
    ET_LOG(Info, "--cuda_graph ignored on non-CUDA build");
  }
  printf("Loading model...\n");
  if (module->load_method("forward") != Error::Ok) {
    ET_LOG(Error, "Failed to load forward method");
    return 1;
  }
#endif

  stats.model_load_end_ms = llm::time_in_ms();

#ifdef EXECUTORCH_BUILD_CUDA
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_load_bytes = gpu_free_bytes;
#endif

  auto eos_ids = llm::get_eos_ids(tokenizer.get(), module.get());
  eos_ids.insert(static_cast<uint64_t>(FLAGS_eos_id));
  auto turn_ids = tokenizer->encode("<turn|>", /*bos=*/0, /*eos=*/0);
  if (turn_ids.ok() && turn_ids->size() == 1) {
    eos_ids.insert(turn_ids.get()[0]);
  }

  // Read prompt
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

  // Wrap with Gemma 4 IT chat template unless --raw_prompt is set.
  // BOS is prepended separately below; this adds the turn structure and the
  // empty thought block required by the instruction-tuned model.
  if (!FLAGS_raw_prompt) {
    prompt_text = "<|turn>user\n" + prompt_text +
        "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
  }

  // Encode prompt
  auto encode_result = tokenizer->encode(prompt_text);
  if (!encode_result.ok()) {
    ET_LOG(Error, "Failed to encode prompt");
    return 1;
  }
  auto prompt_tokens = std::move(*encode_result);
  // Gemma models require BOS at the start of the sequence.
  prompt_tokens.insert(
      prompt_tokens.begin(), static_cast<uint64_t>(FLAGS_bos_id));
  int64_t num_prompt_tokens = static_cast<int64_t>(prompt_tokens.size());
  printf("Prompt tokens: %" PRId64 "\n", num_prompt_tokens);
  stats.num_prompt_tokens = num_prompt_tokens;

  stats.inference_start_ms = llm::time_in_ms();

  // ---------------------------------------------------------------
  // Prefill (chunked to respect ring-buffer KV cache limit)
  // ---------------------------------------------------------------
  uint64_t cur_token = 0;
  int64_t prefill_pos = 0;
  while (prefill_pos < num_prompt_tokens) {
    int64_t chunk_len =
        std::min(num_prompt_tokens - prefill_pos, max_prefill_chunk);

    std::vector<int64_t> token_data(
        prompt_tokens.begin() + prefill_pos,
        prompt_tokens.begin() + prefill_pos + chunk_len);
    std::vector<int64_t> pos_data(chunk_len);
    for (int64_t i = 0; i < chunk_len; i++) {
      pos_data[i] = prefill_pos + i;
    }
    auto tokens_tensor = from_blob(
        token_data.data(),
        {1, S(chunk_len)},
        executorch::aten::ScalarType::Long);
    auto pos_tensor = from_blob(
        pos_data.data(), {S(chunk_len)}, executorch::aten::ScalarType::Long);

    std::vector<EValue> inputs;
    inputs.push_back(EValue(tokens_tensor));
    inputs.push_back(EValue(pos_tensor));

#ifdef EXECUTORCH_BUILD_CUDA
    inputs.push_back(EValue(temp_tensor));
    std::string method = (chunk_len == 1) ? "decode" : "prefill";
#else
    std::string method = "forward";
#endif

    auto result = module->execute(method, inputs);
    if (result.error() != Error::Ok) {
      ET_LOG(Error, "%s failed at pos %" PRId64, method.c_str(), prefill_pos);
      return 1;
    }

#ifdef EXECUTORCH_BUILD_CUDA
    cur_token = read_token(result.get()[0].toTensor());
#else
    cur_token = static_cast<uint64_t>(
        llm::logits_to_token(result.get()[0].toTensor(), temp_val));
#endif

    prefill_pos += chunk_len;
  }

  stats.prompt_eval_end_ms = llm::time_in_ms();
  // First generated token came from the last prefill chunk; TTFT is prefill.
  stats.first_token_ms = stats.prompt_eval_end_ms;

#ifdef EXECUTORCH_BUILD_CUDA
  cudaDeviceSynchronize();
#endif

  // Print the first generated token (from the last prefill chunk).
  // Use the last prompt token as the streaming-decode prefix so any BPE
  // partial-character handling stays correct.
  {
    auto first_str = tokenizer->decode(prompt_tokens.back(), cur_token);
    if (first_str.ok()) {
      printf("%s", first_str->c_str());
      fflush(stdout);
    }
  }

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
  bool hit_eos = eos_ids.contains(cur_token);
  for (int32_t step = 0; step < FLAGS_max_new_tokens && !hit_eos; step++) {
    decode_token_data[0] = static_cast<int64_t>(cur_token);
    decode_pos_data[0] = pos;

    std::vector<EValue> inputs;
    inputs.push_back(EValue(decode_tokens));
    inputs.push_back(EValue(decode_pos));

#ifdef EXECUTORCH_BUILD_CUDA
    inputs.push_back(EValue(temp_tensor));
    auto result = module->execute("decode", inputs);
#else
    auto result = module->execute("forward", inputs);
#endif

    if (result.error() != Error::Ok) {
      ET_LOG(Error, "Decode step %d failed", step);
      return 1;
    }

    prev_token = cur_token;
#ifdef EXECUTORCH_BUILD_CUDA
    cur_token = read_token(result.get()[0].toTensor());
#else
    cur_token = static_cast<uint64_t>(
        llm::logits_to_token(result.get()[0].toTensor(), temp_val));
#endif
    pos++;

    auto decode_str = tokenizer->decode(prev_token, cur_token);
    if (decode_str.ok()) {
      printf("%s", decode_str->c_str());
      fflush(stdout);
    }

    hit_eos = eos_ids.contains(cur_token);
  }
  printf("\n");

  stats.inference_end_ms = llm::time_in_ms();
  stats.num_generated_tokens = pos - num_prompt_tokens;

#ifdef EXECUTORCH_BUILD_CUDA
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_generate_bytes = gpu_free_bytes;
  stats.gpu_peak_usage_mb =
      (stats.gpu_total_bytes - gpu_free_bytes) / 1024.0 / 1024.0;
#endif

  llm::print_report(stats);
  return 0;
}
