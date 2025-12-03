/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model_manager.hpp"

#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#include <gflags/gflags.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

DEFINE_string(
    model_path,
    "",
    "Path to directory containing piecewise model files (input_block.pte, transformer_block_*.pte, output_block.pte)");

DEFINE_string(
    tokenizer_path,
    "",
    "Path to tokenizer file (tokenizer.model or tokenizer.bin)");

DEFINE_string(
    prompt,
    "Once upon a time,",
    "Text prompt for generation");

DEFINE_double(
    temperature,
    0.6,
    "Temperature for sampling. 0 = greedy argmax sampling (deterministic). Lower temperature = more deterministic");

DEFINE_int32(
    max_new_tokens,
    100,
    "Maximum number of new tokens to generate");

using executorch::aten::Tensor;
using executorch::extension::ModelManager;
using executorch::runtime::Error;
using tokenizers::Tokenizer;

/**
 * Samples the next token from logits using temperature sampling.
 */
static int64_t sample_token(const float* logits, int64_t vocab_size, double temperature) {
  if (temperature == 0.0) {
    // Greedy sampling - find argmax
    int64_t max_idx = 0;
    float max_val = logits[0];
    for (int64_t i = 1; i < vocab_size; ++i) {
      if (logits[i] > max_val) {
        max_val = logits[i];
        max_idx = i;
      }
    }
    return max_idx;
  }

  // Temperature sampling
  std::vector<float> probs(vocab_size);
  float max_logit = logits[0];
  for (int64_t i = 1; i < vocab_size; ++i) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }

  // Apply temperature and softmax
  float sum = 0.0f;
  for (int64_t i = 0; i < vocab_size; ++i) {
    probs[i] = std::exp((logits[i] - max_logit) / temperature);
    sum += probs[i];
  }
  for (int64_t i = 0; i < vocab_size; ++i) {
    probs[i] /= sum;
  }

  // Sample from the distribution
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
  return dist(gen);
}

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
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
  ET_LOG(Info, "Loading tokenizer from %s", FLAGS_tokenizer_path.c_str());
  auto tiktoken = example::get_tiktoken_for_llama();
  if (!tiktoken) {
    ET_LOG(Error, "Failed to create tiktoken");
    return 1;
  }

  auto load_error = tiktoken->load(FLAGS_tokenizer_path);
  if (load_error != tokenizers::Error::Ok) {
    ET_LOG(Error, "Failed to load tokenizer: %d", static_cast<int>(load_error));
    return 1;
  }

  // Load model
  ET_LOG(Info, "Loading model from %s", FLAGS_model_path.c_str());
  std::unique_ptr<ModelManager> model_manager;
  try {
    model_manager = std::make_unique<ModelManager>(FLAGS_model_path);
  } catch (const std::exception& e) {
    ET_LOG(Error, "Failed to load model: %s", e.what());
    return 1;
  }

  // Tokenize prompt
  auto encode_res = tiktoken->encode(FLAGS_prompt, /*bos=*/1, /*eos=*/0);
  if (!encode_res.ok()) {
    ET_LOG(Error, "Failed to encode prompt");
    return 1;
  }
  std::vector<uint64_t> prompt_tokens_u64 = std::move(*encode_res);

  // Convert to int64_t
  std::vector<int64_t> tokens;
  for (uint64_t token : prompt_tokens_u64) {
    tokens.push_back(static_cast<int64_t>(token));
  }

  ET_LOG(Info, "Prompt has %zu tokens", tokens.size());
  std::cout << FLAGS_prompt << std::flush;

  // Get EOS token for stop check
  uint64_t eos_token = tiktoken->eos_tok();

  // Generation loop
  int64_t max_seq_length = model_manager->get_max_seq_length();
  int64_t input_pos = 0;  // Track position externally

  // Pre-allocate buffer for input_pos tensor
  std::vector<int64_t> input_pos_buffer = {0};

  // Timing statistics
  bool is_prefill = true;
  int64_t prefill_tokens = 0;
  double prefill_time_ms = 0.0;
  int64_t decode_tokens = 0;
  double decode_time_ms = 0.0;

  for (int32_t i = 0; i < FLAGS_max_new_tokens && input_pos < max_seq_length; ++i) {
    // Create tokens tensor [1, num_tokens] - ModelManager will handle chunking if needed
    size_t num_tokens = tokens.size();
    auto tokens_tensor = executorch::extension::from_blob(
        tokens.data(),
        {1, static_cast<int>(num_tokens)},
        executorch::aten::ScalarType::Long);

    // Update input_pos buffer and create tensor [1]
    input_pos_buffer[0] = input_pos;
    auto input_pos_tensor = executorch::extension::from_blob(
        input_pos_buffer.data(),
        {1},
        executorch::aten::ScalarType::Long);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run forward pass with tensors
    auto logits_result = model_manager->forward(*tokens_tensor, *input_pos_tensor);
    if (!logits_result.ok()) {
      ET_LOG(Error, "Forward pass failed: 0x%" PRIx32,
             static_cast<uint32_t>(logits_result.error()));
      return 1;
    }

    Tensor logits = std::move(*logits_result);

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double time_ms = duration.count() / 1000.0;

    // Track prefill vs decode timing
    if (is_prefill) {
      prefill_tokens = num_tokens;
      prefill_time_ms = time_ms;
      is_prefill = false;  // Switch to decode after first pass
    } else {
      decode_tokens++;
      decode_time_ms += time_ms;
    }

    // Sample next token from logits
    // The output block returns logits shaped [batch_size, vocab_size] for the last position
    int64_t vocab_size = logits.size(logits.dim() - 1);

    // Convert fp16 logits to fp32 for sampling
    const exec_aten::Half* logits_fp16 = logits.const_data_ptr<exec_aten::Half>();

    // Calculate total elements
    int64_t total_elements = 1;
    for (int32_t d = 0; d < logits.dim(); ++d) {
      total_elements *= logits.size(d);
    }

    std::vector<float> logits_fp32(total_elements);
    for (int64_t j = 0; j < total_elements; ++j) {
      logits_fp32[j] = static_cast<float>(logits_fp16[j]);
    }

    // Calculate the total number of positions in the logits tensor
    int64_t total_positions = 1;
    for (int32_t d = 0; d < logits.dim() - 1; ++d) {
      total_positions *= logits.size(d);
    }

    // Get pointer to last position's logits
    size_t last_valid_pos_offset = (total_positions - 1) * vocab_size;
    const float* last_logits = logits_fp32.data() + last_valid_pos_offset;

    int64_t next_token = sample_token(last_logits, vocab_size, FLAGS_temperature);

    // Check for stop tokens
    if (static_cast<uint64_t>(next_token) == eos_token) {
      ET_LOG(Info, "Reached end-of-text token");
      break;
    }

    // Decode and print
    auto decode_res = tiktoken->decode(next_token, next_token);
    if (decode_res.ok()) {
      std::cout << *decode_res << std::flush;
    }

    // Update position and prepare for next iteration
    input_pos += num_tokens;  // Advance position by number of tokens processed
    tokens = {next_token};  // Next iteration processes just the new token
  }

  std::cout << std::endl;

  // Print timing statistics
  std::cout << "\n=== Generation Statistics ===" << std::endl;

  if (prefill_tokens > 0) {
    double prefill_tok_per_sec = (prefill_time_ms > 0) ? (prefill_tokens * 1000.0 / prefill_time_ms) : 0.0;
    std::cout << "Prefill: " << prefill_tokens << " tokens in "
              << prefill_time_ms << " ms ("
              << prefill_tok_per_sec << " tok/s)" << std::endl;
  }

  if (decode_tokens > 0) {
    double avg_decode_time_ms = decode_time_ms / decode_tokens;
    double decode_tok_per_sec = (decode_time_ms > 0) ? (decode_tokens * 1000.0 / decode_time_ms) : 0.0;
    std::cout << "Decode: " << decode_tokens << " tokens in "
              << decode_time_ms << " ms ("
              << decode_tok_per_sec << " tok/s, "
              << avg_decode_time_ms << " ms/tok)" << std::endl;
  }

  double total_time_ms = prefill_time_ms + decode_time_ms;
  int64_t total_tokens = prefill_tokens + decode_tokens;
  if (total_tokens > 0) {
    double total_tok_per_sec = (total_time_ms > 0) ? (total_tokens * 1000.0 / total_time_ms) : 0.0;
    std::cout << "Total: " << total_tokens << " tokens in "
              << total_time_ms << " ms ("
              << total_tok_per_sec << " tok/s)" << std::endl;
  }

  ET_LOG(Info, "Generation complete");

  return 0;
}
