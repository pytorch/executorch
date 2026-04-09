/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/portable_type/device.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <vector>

DEFINE_string(model_path, "", "Model .pte file path.");
DEFINE_string(data_path, "", "Data file (.ptd) for CUDA backend.");
DEFINE_string(tokenizer_path, "", "HuggingFace tokenizer.json path.");
DEFINE_string(prompt, "Hello", "Prompt text.");
DEFINE_double(temperature, 0.8, "Sampling temperature (0 = greedy).");
DEFINE_int32(max_new_tokens, 128, "Maximum tokens to generate.");

using namespace executorch::extension;
using namespace executorch::runtime;
using etensor::DeviceType;
using executorch::aten::ScalarType;

constexpr auto kDynamicBound =
    executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND;

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_path.empty() || FLAGS_tokenizer_path.empty()) {
    std::cerr << "Must specify --model_path and --tokenizer_path" << std::endl;
    return 1;
  }

  // Load tokenizer.
  auto tokenizer = std::make_unique<tokenizers::HFTokenizer>();
  auto tok_status = tokenizer->load(FLAGS_tokenizer_path);
  if (tok_status != tokenizers::Error::Ok) {
    std::cerr << "Failed to load tokenizer from " << FLAGS_tokenizer_path
              << std::endl;
    return 1;
  }

  // Load module with optional .ptd data file for CUDA backend weights.
  std::vector<std::string> data_files;
  if (!FLAGS_data_path.empty()) {
    data_files.push_back(FLAGS_data_path);
  }
  Module module(
      FLAGS_model_path, data_files, Module::LoadMode::MmapUseMlockIgnoreErrors);

  auto forward_load = module.load_method("forward");
  if (forward_load != Error::Ok) {
    std::cerr << "Failed to load forward method" << std::endl;
    return 1;
  }
  auto sample_load = module.load_method("sample");
  if (sample_load != Error::Ok) {
    std::cerr << "Failed to load sample method" << std::endl;
    return 1;
  }

  // Encode prompt.
  auto encode_result = tokenizer->encode(FLAGS_prompt);
  if (!encode_result.ok()) {
    std::cerr << "Failed to encode prompt" << std::endl;
    return 1;
  }
  auto prompt_tokens = encode_result.get();
  int num_prompt_tokens = static_cast<int>(prompt_tokens.size());

  // ======================== PREFILL ========================

  auto prefill_start = std::chrono::high_resolution_clock::now();

  // Create CUDA tensors directly for the full prompt.
  // tokens: shape [1, num_prompt_tokens], dtype Long
  std::vector<int64_t> token_data(prompt_tokens.begin(), prompt_tokens.end());
  auto cuda_tokens = make_tensor_ptr(
      /* sizes= */ {1, static_cast<int32_t>(num_prompt_tokens)},
      /* data= */ token_data.data(),
      /* type= */ ScalarType::Long,
      /* dynamism= */ kDynamicBound,
      /* deleter= */ nullptr,
      /* device_type= */ DeviceType::CUDA);

  // positions: shape [num_prompt_tokens], dtype Long
  std::vector<int64_t> pos_data(num_prompt_tokens);
  std::iota(pos_data.begin(), pos_data.end(), 0);
  auto cuda_pos = make_tensor_ptr(
      /* sizes= */ {static_cast<int32_t>(num_prompt_tokens)},
      /* data= */ pos_data.data(),
      /* type= */ ScalarType::Long,
      /* dynamism= */ kDynamicBound,
      /* deleter= */ nullptr,
      /* device_type= */ DeviceType::CUDA);

  // Temperature tensor: shape [1], dtype Float
  float temp_val = static_cast<float>(FLAGS_temperature);
  auto cuda_temp = make_tensor_ptr(
      /* sizes= */ {1},
      /* data= */ &temp_val,
      /* type= */ ScalarType::Float,
      /* dynamism= */ kDynamicBound,
      /* deleter= */ nullptr,
      /* device_type= */ DeviceType::CUDA);

  // Forward pass — logits stay on CUDA.
  auto forward_result = module.execute(
      "forward", {/* tokens= */ *cuda_tokens, /* input_pos= */ *cuda_pos});
  if (!forward_result.ok()) {
    std::cerr << "Forward (prefill) failed" << std::endl;
    return 1;
  }
  auto& forward_outputs = forward_result.get();

  // Sample — input and output both on CUDA.
  auto sample_result = module.execute(
      "sample",
      {/* logits= */ forward_outputs[0], /* temperature= */ *cuda_temp});
  if (!sample_result.ok()) {
    std::cerr << "Sample (prefill) failed" << std::endl;
    return 1;
  }
  auto& sample_outputs = sample_result.get();

  // D2H: copy the single sampled token back to CPU.
  auto& prefill_sample_tensor = sample_outputs[0].toTensor();
  auto cpu_first_token = clone_tensor_ptr_to_cpu(TensorPtr(
      &prefill_sample_tensor, /* deleter= */ [](executorch::aten::Tensor*) {}));
  int64_t cur_token = cpu_first_token->const_data_ptr<int64_t>()[0];

  auto prefill_end = std::chrono::high_resolution_clock::now();
  double prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start)
          .count();
  double prefill_tps = num_prompt_tokens / (prefill_ms / 1000.0);

  // Print the first generated token.
  auto first_decode = tokenizer->decode(
      /* prev= */ static_cast<uint64_t>(prompt_tokens.back()),
      /* cur= */ static_cast<uint64_t>(cur_token));
  if (first_decode.ok()) {
    std::cout << *first_decode << std::flush;
  }

  // ======================== DECODE LOOP ========================

  int pos = num_prompt_tokens;
  int generated = 1;
  int64_t prev_token = static_cast<int64_t>(prompt_tokens.back());
  auto decode_start = std::chrono::high_resolution_clock::now();

  // Carry the CUDA token tensor from the previous sample call so we can
  // feed it directly to forward without an extra H2D copy.
  EValue cuda_token_ev(prefill_sample_tensor);

  // Qwen EOS token IDs.
  const std::set<int64_t> eos_tokens = {151643, 151645};

  while (generated < FLAGS_max_new_tokens) {
    if (eos_tokens.count(cur_token))
      break;

    // Position H2D (single int64 per step).
    int64_t pos_val = static_cast<int64_t>(pos);
    auto cuda_next_pos = make_tensor_ptr(
        /* sizes= */ {1},
        /* data= */ &pos_val,
        /* type= */ ScalarType::Long,
        /* dynamism= */ kDynamicBound,
        /* deleter= */ nullptr,
        /* device_type= */ DeviceType::CUDA);

    // Forward — reuse the CUDA token tensor from the previous sample output.
    auto fwd = module.execute(
        "forward",
        {/* tokens= */ cuda_token_ev, /* input_pos= */ *cuda_next_pos});
    if (!fwd.ok()) {
      std::cerr << "Forward (decode step " << generated << ") failed"
                << std::endl;
      return 1;
    }

    // Sample — stays on CUDA.
    auto smp = module.execute(
        "sample", {/* logits= */ fwd.get()[0], /* temperature= */ *cuda_temp});
    if (!smp.ok()) {
      std::cerr << "Sample (decode step " << generated << ") failed"
                << std::endl;
      return 1;
    }

    // D2H: extract next token for EOS check.
    auto& next_sample_tensor = smp.get()[0].toTensor();
    auto cpu_next_token = clone_tensor_ptr_to_cpu(TensorPtr(
        &next_sample_tensor, /* deleter= */ [](executorch::aten::Tensor*) {}));

    prev_token = cur_token;
    cur_token = cpu_next_token->const_data_ptr<int64_t>()[0];

    // Keep the CUDA tensor for the next iteration's forward call.
    cuda_token_ev = EValue(next_sample_tensor);

    // Decode and stream output.
    auto dec = tokenizer->decode(
        /* prev= */ static_cast<uint64_t>(prev_token),
        /* cur= */ static_cast<uint64_t>(cur_token));
    if (dec.ok()) {
      std::cout << *dec << std::flush;
    }

    pos++;
    generated++;
  }

  auto decode_end = std::chrono::high_resolution_clock::now();
  double decode_ms =
      std::chrono::duration<double, std::milli>(decode_end - decode_start)
          .count();
  double decode_tps =
      (generated > 1) ? (generated - 1) / (decode_ms / 1000.0) : 0;

  std::cout << std::endl;
  std::cout << "Prefill: " << num_prompt_tokens << " tokens, " << prefill_tps
            << " tok/s" << std::endl;
  std::cout << "Decode: " << generated << " tokens, " << decode_tps << " tok/s"
            << std::endl;

  return 0;
}
