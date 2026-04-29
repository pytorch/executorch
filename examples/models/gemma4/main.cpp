/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Gemma4 lowered .pte runner. Mirrors examples/models/qwen3_5_moe/main.cpp
// but adapts to the Gemma4 export contract:
//   - decode/prefill take exactly (tokens[B,T], input_pos[T])
//   - model returns full logits [B, T, vocab] (bf16); the runner argmaxes
//     over the last position to produce the next greedy token.
//
// Crucially: weight_sharing_across_methods = true is set BEFORE load_method,
// so prefill and decode see the same KV cache state in the AOTI container.
// Without this, decode runs against a zero-init cache and degenerates.

#include <gflags/gflags.h>

#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/platform.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Strong definition of the logging PAL so ET_LOG messages are visible.
#include <executorch/runtime/platform/types.h>
extern "C" void et_pal_emit_log_message(
    ET_UNUSED et_timestamp_t timestamp,
    et_pal_log_level_t level,
    const char* filename,
    ET_UNUSED const char* function,
    size_t line,
    const char* message,
    ET_UNUSED size_t length) {
  fprintf(stderr, "%c [%s:%zu] %s\n", (char)level, filename, line, message);
  fflush(stderr);
}

#ifdef EXECUTORCH_BUILD_CUDA
#include <cuda_runtime.h>
#endif

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(data_path, "", "Data file (.ptd) for CUDA backend.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(prompt, "The capital of France is", "Prompt text.");
DEFINE_string(
    prompt_file,
    "",
    "Path to file containing prompt text (overrides --prompt).");
DEFINE_int32(max_new_tokens, 30, "Maximum tokens to generate.");
DEFINE_int32(bos_id, 2, "BOS token id to prepend (Gemma convention: 2).");
DEFINE_bool(prepend_bos, true, "Prepend BOS token id before encoded prompt.");

namespace llm = ::executorch::extension::llm;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;

using SizesType = executorch::aten::SizesType;

// Argmax over the last vocab dimension of a [B, T, vocab] bf16 logits tensor,
// returning the token id at position T-1, batch 0.
//
// Logits live on GPU in the lowered model. We copy the slice for batch 0,
// last position to host (vocab * 2 bytes, ~512 KiB for vocab=262144) and
// scan it. Done in bf16 directly — argmax doesn't need fp32, only ordering.
static uint64_t argmax_last_token(const executorch::aten::Tensor& output) {
  const auto sizes = output.sizes();
  if (sizes.size() != 3) {
    ET_LOG(
        Error,
        "argmax_last_token: expected rank 3 logits, got rank %zu",
        sizes.size());
    return 0;
  }
  const int64_t T = sizes[1];
  const int64_t V = sizes[2];

  if (output.scalar_type() != executorch::aten::ScalarType::BFloat16) {
    ET_LOG(
        Error,
        "argmax_last_token: expected bf16 logits, got dtype %d",
        static_cast<int>(output.scalar_type()));
    return 0;
  }

  const auto* base = static_cast<const uint16_t*>(output.const_data_ptr());
  // Slice is logits[0, T-1, :] — element offset = (T - 1) * V.
  const uint16_t* slice_dev = base + (T - 1) * V;

  // Bring the slice to host (works whether base is on device or host).
  std::vector<uint16_t> host_logits(V);
  bool on_device = false;
#ifdef EXECUTORCH_BUILD_CUDA
  cudaPointerAttributes attrs;
  on_device = cudaPointerGetAttributes(&attrs, base) == cudaSuccess &&
      attrs.type == cudaMemoryTypeDevice;
  if (on_device) {
    cudaError_t err = cudaMemcpy(
        host_logits.data(),
        slice_dev,
        V * sizeof(uint16_t),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      ET_LOG(
          Error,
          "argmax_last_token: cudaMemcpy D2H failed: %s",
          cudaGetErrorString(err));
      return 0;
    }
  }
#endif
  if (!on_device) {
    std::memcpy(host_logits.data(), slice_dev, V * sizeof(uint16_t));
  }

  // bf16 ordering matches fp32 ordering when both are non-NaN, so we can
  // compare the raw 16-bit pattern as a signed int (treat bf16 like a
  // truncated fp32). Use bit-cast to fp32 for safety.
  auto bf16_to_float = [](uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
  };

  uint64_t best_idx = 0;
  float best_val = bf16_to_float(host_logits[0]);
  for (int64_t i = 1; i < V; i++) {
    float v = bf16_to_float(host_logits[i]);
    if (v > best_val) {
      best_val = v;
      best_idx = static_cast<uint64_t>(i);
    }
  }
  return best_idx;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  et_pal_init();

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
  auto tok_status = tokenizer->load(FLAGS_tokenizer_path);
  if (tok_status != tokenizers::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to load tokenizer from %s",
        FLAGS_tokenizer_path.c_str());
    return 1;
  }

  // Module — share_memory_arenas=true lets prefill/decode share mutable
  // buffer regions at the ExecuTorch level (no-op for fully-delegated CUDA
  // graphs but harmless and matches qwen).
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

  printf("Loading methods...\n");

  // *** THE FIX ***
  // Enable cross-method per-FQN cache sharing in the CUDA backend. Without
  // this, the AOTI container for each method gets its own copy of the KV
  // cache state — decode then runs against zero-init cache and degenerates
  // (token 236743 / 13166 cycle observed via pybinding loader, which has
  // no API to set this option). Must be set BEFORE load_method since the
  // backend reads it during init().
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

#ifdef EXECUTORCH_BUILD_CUDA
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_load_bytes = gpu_free_bytes;
#endif

  // EOS ids (best-effort; fine if empty for this debug runner)
  auto eos_ids = llm::get_eos_ids(tokenizer.get(), module.get());

  // Prompt text
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
  std::vector<uint64_t> prompt_token_ids(std::move(*encode_result));

  std::vector<int64_t> token_data;
  if (FLAGS_prepend_bos) {
    token_data.push_back(static_cast<int64_t>(FLAGS_bos_id));
  }
  for (uint64_t t : prompt_token_ids) {
    token_data.push_back(static_cast<int64_t>(t));
  }
  int64_t num_prompt_tokens = static_cast<int64_t>(token_data.size());

  printf("Prompt: %s\n", prompt_text.c_str());
  printf("Prompt tokens (%" PRId64 "):", num_prompt_tokens);
  for (int64_t t : token_data) {
    printf(" %" PRId64, t);
  }
  printf("\n");

  stats.num_prompt_tokens = num_prompt_tokens;
  stats.inference_start_ms = llm::time_in_ms();

  auto S = [](int64_t v) -> SizesType { return static_cast<SizesType>(v); };

  // ---------------------------------------------------------------
  // Prefill — produces logits[1, T, vocab]; argmax over last position
  // ---------------------------------------------------------------
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

  std::vector<EValue> prefill_inputs;
  prefill_inputs.push_back(tokens_tensor);
  prefill_inputs.push_back(pos_tensor);

  // For T=1 inputs, prefill was exported with min seq_len=2, so use decode.
  std::string run_method = "prefill";
  if (num_prompt_tokens == 1) {
    run_method = "decode";
    // For decode, tokens shape must be [1,1] (already is).
  }

  auto prefill_result = module->execute(run_method, prefill_inputs);
  if (prefill_result.error() != Error::Ok) {
    ET_LOG(Error, "Prefill failed");
    return 1;
  }
  auto& prefill_outputs = prefill_result.get();
  uint64_t cur_token = argmax_last_token(prefill_outputs[0].toTensor());

  stats.prompt_eval_end_ms = llm::time_in_ms();

  double prefill_ms =
      (double)(stats.prompt_eval_end_ms - stats.inference_start_ms);
  printf(
      "Prefill: %" PRId64 " tokens in %.1f ms (first new token id = %" PRIu64
      ")\n",
      num_prompt_tokens,
      prefill_ms,
      cur_token);

#ifdef EXECUTORCH_BUILD_CUDA
  // Make sure prefill's KV cache writes are visible to decode (which may
  // run on a different CUDA stream).
  cudaDeviceSynchronize();
#endif

  // ---------------------------------------------------------------
  // Decode loop — generate tokens one at a time
  // ---------------------------------------------------------------
  std::vector<int64_t> generated;
  generated.push_back(static_cast<int64_t>(cur_token));

  int64_t pos = num_prompt_tokens;

  std::vector<int64_t> decode_token_data = {static_cast<int64_t>(cur_token)};
  std::vector<int64_t> decode_pos_data = {pos};
  auto decode_tokens = from_blob(
      decode_token_data.data(), {1, 1}, executorch::aten::ScalarType::Long);
  auto decode_pos = from_blob(
      decode_pos_data.data(), {1}, executorch::aten::ScalarType::Long);

  for (int32_t step = 1; step < FLAGS_max_new_tokens; step++) {
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

    cur_token = argmax_last_token(decode_outputs[0].toTensor());
    generated.push_back(static_cast<int64_t>(cur_token));

    if (step == 1) {
      stats.first_token_ms = llm::time_in_ms();
    }

    pos++;

    if (eos_ids.find(cur_token) != eos_ids.end()) {
      break;
    }
  }

  stats.inference_end_ms = llm::time_in_ms();

  // Decode and print all generated tokens at the end (greedy, no streaming
  // — keeps the output deterministic and trivially diffable against eager).
  std::vector<uint64_t> all_ids;
  for (int64_t t : token_data) {
    all_ids.push_back(static_cast<uint64_t>(t));
  }
  for (int64_t t : generated) {
    all_ids.push_back(static_cast<uint64_t>(t));
  }

  auto cont_decoded = tokenizer->decode(0, 0); // placeholder
  std::string full_text;
  // tokenizer->decode takes (prev_id, cur_id); easiest: decode pairs.
  for (size_t i = 1; i < all_ids.size(); i++) {
    auto piece = tokenizer->decode(all_ids[i - 1], all_ids[i]);
    if (piece.ok()) {
      full_text += *piece;
    }
  }

  printf("\n=== Generated ids ===\n");
  for (int64_t t : generated) {
    printf("%" PRId64 " ", t);
  }
  printf("\n");

  printf("\n=== Full decoded text ===\n%s\n", full_text.c_str());

  int64_t num_generated = pos - num_prompt_tokens;
  stats.num_generated_tokens = num_generated;

  double decode_ms =
      (double)(stats.inference_end_ms - stats.prompt_eval_end_ms);
  printf(
      "\nDecode: %" PRId64 " tokens in %.1f ms (%.1f tok/s)\n",
      num_generated,
      decode_ms,
      num_generated * 1000.0 / decode_ms);

#ifdef EXECUTORCH_BUILD_CUDA
  cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
  stats.gpu_free_after_generate_bytes = gpu_free_bytes;
  stats.gpu_peak_usage_mb =
      (stats.gpu_total_bytes - gpu_free_bytes) / 1024.0 / 1024.0;
#endif

  llm::print_report(stats);

  return 0;
}
