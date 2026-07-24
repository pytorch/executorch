/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Standalone C++ DFlash speculative decoding driver for Gemma4-31B.
 * Ports run_dflash.py's draft/verify/accept loop directly using
 * ExecuTorch's Module API (raw tensor I/O), since the higher-level
 * LLMEngine/LLMSession framework used by main.cpp is built for
 * single-model token-at-a-time generation, not this two-model
 * draft-then-verify pattern with multi-output tensors.
 */

#include <gflags/gflags.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/platform/log.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

using executorch::extension::Module;
using executorch::extension::make_tensor_ptr;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;
using executorch::runtime::Error;

DEFINE_string(target_pte, "gemma4_31b_dflash_exports_mlx/model.pte", "Target .pte path.");
DEFINE_string(draft_pte, "gemma4_31b_dflash_draft.pte", "Draft .pte path.");
DEFINE_string(tokenizer_path, "gemma-4-31B-it-HQQ-INT4/tokenizer.json", "Tokenizer path.");
DEFINE_string(prompt, "The capital of France is", "Prompt text.");
DEFINE_int32(max_new_tokens, 64, "Maximum tokens to generate.");
DEFINE_int32(block_size, 16, "DFlash draft block size.");
DEFINE_int32(mask_id, 4, "DFlash mask token id (from draft checkpoint config).");
DEFINE_bool(raw_prompt, false, "Skip chat-template wrapping.");
DEFINE_bool(verbose, false, "Print per-round timing/acceptance debug output.");

namespace {

constexpr int64_t kBosId = 2;
const std::set<int64_t> kEosIds = {1, 50, 106};

std::string format_prompt(const std::string& prompt) {
  return "<|turn>user\n" + prompt +
      "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::vector<int64_t> argmax_last_dim(const executorch::aten::Tensor& t) {
  auto sizes = t.sizes();
  int64_t T = sizes[1];
  int64_t V = sizes[2];
  const float* data = t.const_data_ptr<float>();
  std::vector<int64_t> out(T);
  for (int64_t i = 0; i < T; ++i) {
    const float* row = data + i * V;
    out[i] = std::max_element(row, row + V) - row;
  }
  return out;
}

int64_t first_mismatch(
    const std::vector<int64_t>& draft_ids,
    const std::vector<int64_t>& target_ids) {
  for (size_t i = 0; i < draft_ids.size(); ++i) {
    if (draft_ids[i] != target_ids[i]) {
      return static_cast<int64_t>(i);
    }
  }
  return static_cast<int64_t>(draft_ids.size());
}

// The target's hidden-state output is BFloat16 (confirmed via MethodMeta:
// dtype=BFloat16), NOT Float32 -- treating its raw bytes as float32 via
// const_data_ptr<float>() silently reinterprets 2-byte bf16 values as
// 4-byte floats, producing garbage that made the draft's context
// essentially random (observed: near-zero acceptance, ctx_len advancing
// by ~1 per round instead of the expected 5-15). bf16 is simply the top
// 16 bits of a float32 (same exponent width, truncated mantissa), so
// converting is a left-shift into the upper half of a 32-bit word.
std::vector<float> bf16_tensor_to_fp32(const executorch::aten::Tensor& t) {
  int64_t numel = t.numel();
  const uint16_t* src = reinterpret_cast<const uint16_t*>(t.const_data_ptr());
  std::vector<float> out(numel);
  for (int64_t i = 0; i < numel; ++i) {
    uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
    std::memcpy(&out[i], &bits, sizeof(float));
  }
  return out;
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  if (tokenizer->load(FLAGS_tokenizer_path) != tokenizers::Error::Ok) {
    ET_LOG(Error, "Failed to load tokenizer: %s", FLAGS_tokenizer_path.c_str());
    return 1;
  }

  Module target_module(FLAGS_target_pte);
  Module draft_module(FLAGS_draft_pte);
  if (target_module.load() != Error::Ok) {
    ET_LOG(Error, "Failed to load target .pte: %s", FLAGS_target_pte.c_str());
    return 1;
  }
  if (draft_module.load() != Error::Ok) {
    ET_LOG(Error, "Failed to load draft .pte: %s", FLAGS_draft_pte.c_str());
    return 1;
  }

  std::string prompt_text =
      FLAGS_raw_prompt ? FLAGS_prompt : format_prompt(FLAGS_prompt);
  auto encoded = tokenizer->encode(prompt_text, /*bos=*/0, /*eos=*/0);
  if (!encoded.ok()) {
    ET_LOG(Error, "Failed to encode prompt");
    return 1;
  }
  std::vector<int64_t> prompt_ids;
  prompt_ids.push_back(kBosId);
  for (auto id : encoded.get()) {
    prompt_ids.push_back(static_cast<int64_t>(id));
  }
  int64_t prompt_len = static_cast<int64_t>(prompt_ids.size());
  printf("Prompt tokens: %lld\n", (long long)prompt_len);

  std::vector<int64_t> input_pos_vec(prompt_len);
  for (int64_t i = 0; i < prompt_len; ++i) input_pos_vec[i] = i;

  auto prompt_ids_tensor =
      make_tensor_ptr({1, (int)prompt_len}, prompt_ids.data(), executorch::aten::ScalarType::Long);
  auto input_pos_tensor = make_tensor_ptr(
      {(int)prompt_len}, input_pos_vec.data(), executorch::aten::ScalarType::Long);

  auto prefill_result =
      target_module.forward({EValue(prompt_ids_tensor), EValue(input_pos_tensor)});
  if (!prefill_result.ok()) {
    ET_LOG(Error, "Prefill forward failed");
    return 1;
  }
  auto prefill_outputs = prefill_result.get();
  auto logits = prefill_outputs[0].toTensor();
  auto hidden = prefill_outputs[1].toTensor();

  int64_t pos = prompt_len;
  auto logits_argmax = argmax_last_dim(logits);
  int64_t last_token = logits_argmax.back();

  std::vector<int64_t> generated = {last_token};
  int64_t rounds = 0;
  int64_t accepted_total = 0;
  int64_t emitted_total = 0;



  int64_t hidden_concat_dim = hidden.sizes()[2];
  std::vector<float> hidden_history = bf16_tensor_to_fp32(hidden);
  int64_t hidden_len = prompt_len;

  // Warm-up: MLX/Metal lazily JIT-compiles kernels on a delegate's FIRST
  // real execution, not at Module::load() time. Without this, that one-time
  // compile cost lands inside round 1's timed draft_exec (observed: ~296ms
  // vs ~40ms steady-state) and skews the reported tokens/s. Run one
  // throwaway forward with the actual round-1 inputs, discard the result,
  // so the timed loop below only measures steady-state performance --
  // matching what Python's Runtime.load_program().load_method() chain
  // appears to already do more eagerly at load time.
  {
    std::vector<int64_t> warm_input_vec(FLAGS_block_size);
    warm_input_vec[0] = last_token;
    for (int64_t i = 1; i < FLAGS_block_size; ++i) warm_input_vec[i] = FLAGS_mask_id;
    auto warm_input_tensor = make_tensor_ptr(
        {1, (int)FLAGS_block_size}, warm_input_vec.data(), executorch::aten::ScalarType::Long);
    auto warm_hidden_tensor = make_tensor_ptr(
        {1, (int)hidden_len, (int)hidden_concat_dim},
        hidden_history.data(),
        executorch::aten::ScalarType::Float);
    std::vector<int64_t> warm_pos_vec(hidden_len + FLAGS_block_size);
    for (int64_t i = 0; i < hidden_len + FLAGS_block_size; ++i) warm_pos_vec[i] = i;
    auto warm_pos_tensor = make_tensor_ptr(
        {1, (int)(hidden_len + FLAGS_block_size)},
        warm_pos_vec.data(),
        executorch::aten::ScalarType::Long);
    auto warm_result = draft_module.forward(
        {EValue(warm_input_tensor), EValue(warm_hidden_tensor), EValue(warm_pos_tensor)});
    if (!warm_result.ok()) {
      ET_LOG(Error, "Draft warm-up forward failed");
      return 1;
    }
  }

  // Timer starts AFTER prefill AND warm-up, matching run_dflash.py's t0
  // placement -- tokens/s measures only the speculative decoding round
  // loop's steady-state performance.
  double t0 = now_ms();

  while (static_cast<int64_t>(generated.size()) < FLAGS_max_new_tokens) {
    rounds++;
    int64_t bs = FLAGS_block_size;

    std::vector<int64_t> draft_input_vec(bs);
    draft_input_vec[0] = last_token;
    for (int64_t i = 1; i < bs; ++i) draft_input_vec[i] = FLAGS_mask_id;
    auto draft_input_tensor =
        make_tensor_ptr({1, (int)bs}, draft_input_vec.data(), executorch::aten::ScalarType::Long);

    auto hidden_tensor = make_tensor_ptr(
        {1, (int)hidden_len, (int)hidden_concat_dim},
        hidden_history.data(),
        executorch::aten::ScalarType::Float);

    std::vector<int64_t> draft_pos_vec(hidden_len + bs);
    for (int64_t i = 0; i < hidden_len + bs; ++i) draft_pos_vec[i] = i;
    auto draft_pos_tensor = make_tensor_ptr(
        {1, (int)(hidden_len + bs)}, draft_pos_vec.data(), executorch::aten::ScalarType::Long);

    double dt0 = now_ms();
    auto draft_result = draft_module.forward(
        {EValue(draft_input_tensor), EValue(hidden_tensor), EValue(draft_pos_tensor)});
    double draft_exec_ms = now_ms() - dt0;
    if (!draft_result.ok()) {
      ET_LOG(Error, "Draft forward failed at round %lld", (long long)rounds);
      return 1;
    }
    auto draft_logits = draft_result.get()[0].toTensor();
    auto draft_ids = argmax_last_dim(draft_logits);

    std::vector<int64_t> verify_input_vec;
    verify_input_vec.push_back(last_token);
    verify_input_vec.insert(verify_input_vec.end(), draft_ids.begin(), draft_ids.end());
    int64_t verify_len = static_cast<int64_t>(verify_input_vec.size());
    auto verify_input_tensor = make_tensor_ptr(
        {1, (int)verify_len}, verify_input_vec.data(), executorch::aten::ScalarType::Long);

    std::vector<int64_t> verify_pos_vec(verify_len);
    for (int64_t i = 0; i < verify_len; ++i) verify_pos_vec[i] = pos + i;
    auto verify_pos_tensor = make_tensor_ptr(
        {(int)verify_len}, verify_pos_vec.data(), executorch::aten::ScalarType::Long);

    double vt0 = now_ms();
    auto verify_result = target_module.forward(
        {EValue(verify_input_tensor), EValue(verify_pos_tensor)});
    double target_exec_ms = now_ms() - vt0;
    if (!verify_result.ok()) {
      ET_LOG(Error, "Verify forward failed at round %lld", (long long)rounds);
      return 1;
    }
    auto verify_outputs = verify_result.get();
    auto target_logits = verify_outputs[0].toTensor();
    auto new_hidden = verify_outputs[1].toTensor();
    auto target_ids = argmax_last_dim(target_logits);

    int64_t accepted = first_mismatch(draft_ids, target_ids);
    if (FLAGS_verbose && rounds <= 10) {
      printf(
          "  timing: draft_exec=%.1fms target_exec=%.1fms ctx_len=%lld\n",
          draft_exec_ms,
          target_exec_ms,
          (long long)hidden_len);
    }

    std::vector<int64_t> new_tokens(draft_ids.begin(), draft_ids.begin() + accepted);
    new_tokens.push_back(target_ids[accepted]);

    bool hit_eos = false;
    for (size_t i = 0; i < new_tokens.size(); ++i) {
      if (kEosIds.count(new_tokens[i])) {
        new_tokens.resize(i + 1);
        accepted = std::min(accepted, static_cast<int64_t>(new_tokens.size()) - 1);
        hit_eos = true;
        break;
      }
    }

    accepted_total += accepted;
    emitted_total += static_cast<int64_t>(new_tokens.size());
    generated.insert(generated.end(), new_tokens.begin(), new_tokens.end());

    pos += static_cast<int64_t>(new_tokens.size());
    last_token = new_tokens.back();
    int64_t append_len = static_cast<int64_t>(new_tokens.size());
    std::vector<float> new_hidden_fp32 = bf16_tensor_to_fp32(new_hidden);
    hidden_history.insert(
        hidden_history.end(),
        new_hidden_fp32.begin(),
        new_hidden_fp32.begin() + append_len * hidden_concat_dim);
    hidden_len += append_len;

    if (hit_eos) break;
  }

  double total_ms = now_ms() - t0;
  int64_t n = static_cast<int64_t>(generated.size());

  printf("\nPrompt: %s\n", FLAGS_prompt.c_str());
  printf("Generated (%lld tokens)\n", (long long)n);
  printf("\n--stats--\n");
  printf("rounds: %lld\n", (long long)rounds);
  printf(
      "avg accepted/round (draft-only): %.2f\n",
      static_cast<double>(accepted_total) / rounds);
  printf(
      "avg emitted/round (tau, incl. bonus): %.2f\n",
      static_cast<double>(emitted_total) / rounds);
  printf(
      "time: %.2fs   tokens/s: %.2f\n",
      total_ms / 1000.0,
      n / (total_ms / 1000.0));

  return 0;
}
