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
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <vector>

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(data_path, "", "Data file (.ptd) for CUDA backend.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(prompt, "Hello", "Prompt text.");
DEFINE_double(temperature, 0.8, "Sampling temperature (0 = greedy).");
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");

namespace llm = ::executorch::extension::llm;
using ::executorch::extension::Module;
using ::executorch::extension::from_blob;
using ::executorch::extension::TensorPtr;
using ::executorch::runtime::EValue;
using ::executorch::runtime::BackendOptions;
using ::executorch::runtime::Error;
using ::executorch::runtime::LoadBackendOptionsMap;

using SizesType = executorch::aten::SizesType;

// Helper to create a zero-initialized state tensor
static TensorPtr make_state(
    std::vector<SizesType> shape,
    executorch::aten::ScalarType dtype = executorch::aten::ScalarType::BFloat16) {
  int64_t numel = 1;
  for (auto s : shape) numel *= s;
  size_t nbytes = numel * executorch::runtime::elementSize(dtype);
  auto* data = calloc(1, nbytes);
  return from_blob(
      data,
      shape,
      dtype,
      [](void* p) { free(p); });
}

// Copy logits from GPU to a CPU buffer for sampling.
// With skip_copy_output_to_cpu_for_method, all outputs stay on GPU.
// We only need logits on CPU; state tensors stay on GPU for zero-copy reuse.
static TensorPtr copy_logits_to_cpu(
    executorch::aten::Tensor& gpu_logits,
    std::vector<char>& cpu_buffer) {
  size_t nbytes = gpu_logits.nbytes();
  cpu_buffer.resize(nbytes);
  cudaMemcpy(cpu_buffer.data(), gpu_logits.const_data_ptr(),
      nbytes, cudaMemcpyDeviceToHost);
  auto sizes = gpu_logits.sizes();
  return from_blob(
      cpu_buffer.data(),
      std::vector<SizesType>(sizes.begin(), sizes.end()),
      gpu_logits.scalar_type());
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

  // Create Module
  std::vector<std::string> data_files;
  if (!FLAGS_data_path.empty()) {
    data_files.push_back(FLAGS_data_path);
  }
  auto module = std::make_unique<Module>(FLAGS_model_path, data_files);

  // Get metadata
  auto metadata_result = llm::get_llm_metadata(tokenizer.get(), module.get());
  if (metadata_result.error() != Error::Ok) {
    ET_LOG(Error, "Failed to get metadata from model");
    return 1;
  }
  auto metadata = metadata_result.get();
  int64_t max_seq_len = metadata.at("get_max_seq_len");

  // Read custom state-shape metadata via constant methods
  auto read_meta = [&](const char* name) -> int64_t {
    auto result = module->execute(name);
    if (result.error() != Error::Ok) {
      ET_LOG(Error, "Failed to read metadata: %s", name);
      return -1;
    }
    return result.get()[0].toInt();
  };
  int64_t num_fla = read_meta("get_num_fla_layers");
  int64_t num_attn = read_meta("get_num_attn_layers");
  int64_t conv_dim = read_meta("get_conv_dim");
  int64_t conv_ks = read_meta("get_conv_kernel_size");
  int64_t num_v_heads = read_meta("get_num_v_heads");
  int64_t head_k_dim = read_meta("get_head_k_dim");
  int64_t head_v_dim = read_meta("get_head_v_dim");
  int64_t n_kv_heads = read_meta("get_n_kv_heads");
  int64_t head_dim = read_meta("get_head_dim");

  printf(
      "Model: max_seq=%ld, fla_layers=%ld, attn_layers=%ld\n",
      max_seq_len,
      num_fla,
      num_attn);

  // Allocate initial state tensors (zero-initialized)
  auto S = [](int64_t v) -> SizesType { return static_cast<SizesType>(v); };
  auto conv_states = make_state(
      {S(num_fla), 1, S(conv_dim), S(conv_ks)});
  auto recurrent_states = make_state(
      {S(num_fla), 1, S(num_v_heads), S(head_k_dim), S(head_v_dim)});
  auto k_caches = make_state(
      {S(num_attn), 1, S(n_kv_heads), S(max_seq_len), S(head_dim)});
  auto v_caches = make_state(
      {S(num_attn), 1, S(n_kv_heads), S(max_seq_len), S(head_dim)});

  // Configure CUDA backend: keep outputs on GPU, use shared CUDA stream
  BackendOptions<2> cuda_opts;
  cuda_opts.set_option("skip_copy_output_to_cpu_for_method", "prefill,forward");
  cuda_opts.set_option("use_shared_cuda_stream", true);

  LoadBackendOptionsMap backend_options;
  backend_options.set_options("CudaBackend", cuda_opts.view());

  // Load both methods with backend options
  printf("Loading prefill method...\n");
  auto err = module->load_method("prefill", nullptr, nullptr, &backend_options);
  if (err != Error::Ok) {
    ET_LOG(Error, "Failed to load prefill method");
    return 1;
  }
  printf("Loading forward (decode) method...\n");
  err = module->load_method("forward", nullptr, nullptr, &backend_options);
  if (err != Error::Ok) {
    ET_LOG(Error, "Failed to load forward method");
    return 1;
  }

  // Get EOS ids
  auto eos_ids = llm::get_eos_ids(tokenizer.get(), module.get());

  // Encode prompt
  auto encode_result = tokenizer->encode(FLAGS_prompt);
  if (!encode_result.ok()) {
    ET_LOG(Error, "Failed to encode prompt");
    return 1;
  }
  auto prompt_tokens = std::move(*encode_result);
  int64_t num_prompt_tokens = prompt_tokens.size();
  printf("Prompt tokens: %ld\n", num_prompt_tokens);

  // ---------------------------------------------------------------
  // Prefill — process all prompt tokens at once
  // ---------------------------------------------------------------
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
  prefill_inputs.push_back(conv_states);
  prefill_inputs.push_back(recurrent_states);
  prefill_inputs.push_back(k_caches);
  prefill_inputs.push_back(v_caches);

  auto prefill_result = module->execute("prefill", prefill_inputs);
  if (prefill_result.error() != Error::Ok) {
    ET_LOG(Error, "Prefill failed");
    return 1;
  }
  auto& prefill_outputs = prefill_result.get();

  // Copy logits from GPU to CPU for sampling; state stays on GPU
  auto gpu_logits = prefill_outputs[0].toTensor();
  std::vector<char> logits_cpu_buf;
  auto logits_ptr = copy_logits_to_cpu(gpu_logits, logits_cpu_buf);
  auto out_conv_states = prefill_outputs[1].toTensor();
  auto out_recurrent_states = prefill_outputs[2].toTensor();
  auto out_k_caches = prefill_outputs[3].toTensor();
  auto out_v_caches = prefill_outputs[4].toTensor();

  uint64_t cur_token = llm::logits_to_token(*logits_ptr, FLAGS_temperature);
  printf("Prefill done, first token: %lu\n", cur_token);

  // ---------------------------------------------------------------
  // Decode — generate tokens one at a time
  // ---------------------------------------------------------------
  llm::Stats stats;
  int64_t pos = num_prompt_tokens;
  uint64_t prev_token;

  std::vector<int64_t> decode_token_data = {static_cast<int64_t>(cur_token)};
  std::vector<int64_t> decode_pos_data = {pos};
  auto decode_tokens = from_blob(
      decode_token_data.data(), {1, 1}, executorch::aten::ScalarType::Long);
  auto decode_pos = from_blob(
      decode_pos_data.data(), {1}, executorch::aten::ScalarType::Long);

  // State as EValues — pass GPU output directly as next input (zero-copy)
  EValue ev_conv(out_conv_states);
  EValue ev_rec(out_recurrent_states);
  EValue ev_k(out_k_caches);
  EValue ev_v(out_v_caches);

  auto gen_start = std::chrono::steady_clock::now();

  for (int32_t step = 0; step < FLAGS_max_new_tokens; step++) {
    decode_token_data[0] = static_cast<int64_t>(cur_token);
    decode_pos_data[0] = pos;

    std::vector<EValue> decode_inputs;
    decode_inputs.push_back(EValue(decode_tokens));
    decode_inputs.push_back(EValue(decode_pos));
    decode_inputs.push_back(ev_conv);
    decode_inputs.push_back(ev_rec);
    decode_inputs.push_back(ev_k);
    decode_inputs.push_back(ev_v);

    auto decode_result = module->execute("forward", decode_inputs);
    if (decode_result.error() != Error::Ok) {
      ET_LOG(Error, "Decode step %d failed", step);
      return 1;
    }
    auto& decode_outputs = decode_result.get();

    // Copy logits from GPU to CPU; state stays on GPU
    auto gpu_step_logits = decode_outputs[0].toTensor();
    auto step_logits_ptr = copy_logits_to_cpu(gpu_step_logits, logits_cpu_buf);

    // Pass state outputs directly as next step's inputs (zero-copy on GPU)
    ev_conv = decode_outputs[1];
    ev_rec = decode_outputs[2];
    ev_k = decode_outputs[3];
    ev_v = decode_outputs[4];

    prev_token = cur_token;
    stats.on_sampling_begin();
    cur_token = llm::logits_to_token(*step_logits_ptr, FLAGS_temperature);
    stats.on_sampling_end();

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

  auto gen_end = std::chrono::steady_clock::now();

  printf("\n");
  int64_t num_generated = pos - num_prompt_tokens;
  double gen_ms = std::chrono::duration<double, std::milli>(
                      gen_end - gen_start)
                      .count();
  printf(
      "Generated %ld tokens in %.1f ms (%.1f tok/s)\n",
      num_generated,
      gen_ms,
      num_generated * 1000.0 / gen_ms);
  printf("Prompt tokens: %ld\n", num_prompt_tokens);

  return 0;
}
