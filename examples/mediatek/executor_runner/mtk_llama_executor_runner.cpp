/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2024 MediaTek Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Copyright Statement:
 *
 * This software/firmware and related documentation ("MediaTek Software") are
 * protected under relevant copyright laws. The information contained herein
 * is confidential and proprietary to MediaTek Inc. and/or its licensors.
 * Without the prior written permission of MediaTek inc. and/or its licensors,
 * any reproduction, modification, use or disclosure of MediaTek Software,
 * and information contained herein, in whole or in part, shall be strictly
 * prohibited.
 */
/* MediaTek Inc. (C) 2024. All rights reserved.
 *
 * BY OPENING THIS FILE, RECEIVER HEREBY UNEQUIVOCALLY ACKNOWLEDGES AND AGREES
 * THAT THE SOFTWARE/FIRMWARE AND ITS DOCUMENTATIONS ("MEDIATEK SOFTWARE")
 * RECEIVED FROM MEDIATEK AND/OR ITS REPRESENTATIVES ARE PROVIDED TO RECEIVER ON
 * AN "AS-IS" BASIS ONLY. MEDIATEK EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE OR NONINFRINGEMENT.
 * NEITHER DOES MEDIATEK PROVIDE ANY WARRANTY WHATSOEVER WITH RESPECT TO THE
 * SOFTWARE OF ANY THIRD PARTY WHICH MAY BE USED BY, INCORPORATED IN, OR
 * SUPPLIED WITH THE MEDIATEK SOFTWARE, AND RECEIVER AGREES TO LOOK ONLY TO SUCH
 * THIRD PARTY FOR ANY WARRANTY CLAIM RELATING THERETO. RECEIVER EXPRESSLY
 * ACKNOWLEDGES THAT IT IS RECEIVER'S SOLE RESPONSIBILITY TO OBTAIN FROM ANY
 * THIRD PARTY ALL PROPER LICENSES CONTAINED IN MEDIATEK SOFTWARE. MEDIATEK
 * SHALL ALSO NOT BE RESPONSIBLE FOR ANY MEDIATEK SOFTWARE RELEASES MADE TO
 * RECEIVER'S SPECIFICATION OR TO CONFORM TO A PARTICULAR STANDARD OR OPEN
 * FORUM. RECEIVER'S SOLE AND EXCLUSIVE REMEDY AND MEDIATEK'S ENTIRE AND
 * CUMULATIVE LIABILITY WITH RESPECT TO THE MEDIATEK SOFTWARE RELEASED HEREUNDER
 * WILL BE, AT MEDIATEK'S OPTION, TO REVISE OR REPLACE THE MEDIATEK SOFTWARE AT
 * ISSUE, OR REFUND ANY SOFTWARE LICENSE FEES OR SERVICE CHARGE PAID BY RECEIVER
 * TO MEDIATEK FOR SUCH MEDIATEK SOFTWARE AT ISSUE.
 *
 * The following software/firmware and/or related documentation ("MediaTek
 * Software") have been modified by MediaTek Inc. All revisions are subject to
 * any receiver's applicable license agreements with MediaTek Inc.
 */

#include "executorch/backends/mediatek/runtime/include/NeuronBufferAllocator.h"

#include <ctime>
#include <iostream>
#include <memory>
#include <random>

#include <gflags/gflags.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>

#include "llama_runner/LlamaConfig.h"
#include "llama_runner/LlamaRuntime.h"
#include "llama_runner/ModelChunk.h"
#include "llama_runner/Utils.h"
#include "llama_runner/llm_helper/include/llm_types.h"

#include <executorch/examples/models/llama2/tokenizer/llama_tiktoken.h>
#include <executorch/extension/llm/tokenizer/bpe_tokenizer.h>
#include <executorch/extension/llm/tokenizer/tiktoken.h>

// Llama model options
DEFINE_uint64(
    prompt_token_batch_size,
    128,
    "Token batch size for prompt model.");
DEFINE_uint64(cache_size, 1024, "Model cache size.");
DEFINE_uint64(hidden_size, 4096, "Model hidden size.");
DEFINE_uint64(num_head, 32, "Number of attention heads in each layer.");
DEFINE_uint64(num_layer, 32, "Number of layers in the model.");
DEFINE_uint64(
    max_token_length,
    2048,
    "Maximum token length that the model supports.");
DEFINE_double(
    rot_emb_base,
    10000,
    "Rotary embedding base value, aka 'rope_theta'.");

// Model IO Types
DEFINE_string(input_type, "int16", "Model input type. Default to 'int16'");
DEFINE_string(output_type, "int16", "Model output type. Default to 'int16'");
DEFINE_string(cache_type, "int16", "Model cache type. Default to 'int16'");
DEFINE_string(mask_type, "int16", "Model mask type. Default to 'int16'");
DEFINE_string(
    rot_emb_type,
    "int16",
    "Model rotary embedding type. Default to 'int16'");

// Model Paths
DEFINE_string(
    token_embedding_path,
    "embedding.bin",
    "Input token embedding lookup table path.");
DEFINE_string(
    prompt_model_paths,
    "model_128t.pte",
    "Comma-separated prompt model paths.");
DEFINE_string(
    gen_model_paths,
    "model_1t.pte",
    "Comma-separated generative model paths.");

// Tokenizer
DEFINE_string(tokenizer_path, "tokenizer.model", "tokenizer.model vocab path.");
DEFINE_string(
    tokenizer_type,
    "tiktoken",
    "Tokenizer type. One of ['bpe', 'tiktoken'].");
DEFINE_uint64(vocab_size, 128000, "Tokenizer vocab size.");
DEFINE_uint64(bos_token, 128000, "BOS token id.");
DEFINE_uint64(eos_token, 128001, "EOS token id.");

// Inference
DEFINE_uint64(max_response, 50, "Maximum number of tokens to generate.");
DEFINE_string(prompt_file, "", "File containing the prompt text.");

// Global BOS and EOS option for tokenization (encoding)
static constexpr int8_t kAddBos = 1;
static constexpr int8_t kAddEos = 0;

using namespace example::llm_helper;
using example::LlamaModelOptions;
using example::LlamaModelPaths;
using example::LlamaRuntime;
using example::utils::argmax;
using example::utils::read_file;
using example::utils::split;
using example::utils::Timer;
using example::utils::to_string;
using executorch::extension::llm::BPETokenizer;
using executorch::extension::llm::Tokenizer;
using executorch::runtime::Error;
using executorch::runtime::Result;

LlamaModelOptions get_model_options() {
  LlamaModelOptions options = {
      // Sizes
      .prompt_token_batch_size = FLAGS_prompt_token_batch_size,
      .cache_size = FLAGS_cache_size,
      .hidden_size = FLAGS_hidden_size,
      .num_head = FLAGS_num_head,
      .num_layer = FLAGS_num_layer,
      .max_token_length = FLAGS_max_token_length,
      .rot_emb_base = FLAGS_rot_emb_base,

      // Types
      .model_input_type = getLLMTypeFromName(FLAGS_input_type.c_str()),
      .model_output_type = getLLMTypeFromName(FLAGS_output_type.c_str()),
      .cache_type = getLLMTypeFromName(FLAGS_cache_type.c_str()),
      .mask_type = getLLMTypeFromName(FLAGS_mask_type.c_str()),
      .rot_emb_type = getLLMTypeFromName(FLAGS_rot_emb_type.c_str())};
  return options;
}

LlamaModelPaths get_model_paths() {
  LlamaModelPaths model_paths = {
      .tokenizer_path = FLAGS_tokenizer_path,
      .token_embedding_path = FLAGS_token_embedding_path,
      .prompt_model_paths = split(FLAGS_prompt_model_paths, ','),
      .gen_model_paths = split(FLAGS_gen_model_paths, ',')};
  return model_paths;
}

Result<uint64_t> digest_prompt(
    LlamaRuntime& llama_runtime,
    const std::unique_ptr<Tokenizer>& tokenizer,
    const std::vector<uint64_t> input_tokens) {
  const auto input_token_count = input_tokens.size();
  const auto prompt_token_batch_size = llama_runtime.GetTokenBatchSize();
  size_t cur_token_index = 0;

  Timer timer_digest_prompt([=](const auto elapsed_sec) {
    // Ideal prompt size is a multiple of prompt batch size
    const size_t ideal_prompt_size =
        std::ceil(float(input_token_count) / prompt_token_batch_size) *
        prompt_token_batch_size;
    ET_LOG(
        Info,
        "Done analyzing prompt in %f sec (%f tok/s)",
        elapsed_sec,
        (float)ideal_prompt_size / elapsed_sec);
  });

  auto getNextTokens = [&]() {
    const size_t num_tok_remain = input_token_count - cur_token_index;
    const size_t remainder = num_tok_remain % prompt_token_batch_size;
    const size_t num_new_tokens =
        remainder ? remainder : prompt_token_batch_size;
    const auto start = cur_token_index;
    const auto end = start + num_new_tokens;
    return std::vector(
        input_tokens.begin() + start, input_tokens.begin() + end);
  };

  void* logits;
  timer_digest_prompt.Start();
  while (cur_token_index < input_token_count) {
    const auto next_tokens = getNextTokens();
    ET_LOG(
        Debug,
        "Digest next tokens (size=%zu), 1st tok=%lu",
        next_tokens.size(),
        next_tokens[0]);
    logits = llama_runtime.Run(next_tokens);
    cur_token_index += next_tokens.size();
  }
  timer_digest_prompt.End();

  const auto vocab_size = tokenizer->vocab_size();
  const auto logits_type = llama_runtime.GetModelOptions().model_output_type;
  const auto first_output_token = argmax(logits_type, logits, vocab_size);
  return first_output_token;
}

Error gen_response(
    LlamaRuntime& llama_runtime,
    const std::unique_ptr<Tokenizer>& tokenizer,
    const uint64_t input_token) {
  Timer timer_model_swap(
      [](const auto elapsed_sec) { ET_LOG(Info, "Model swapped."); });

  // Swap to gen mode
  timer_model_swap.Start();
  llama_runtime.SwapModel(1);
  timer_model_swap.End();

  size_t gen_tok_count = 0;
  uint64_t prev_token = input_token;
  uint64_t output_token = input_token;

  auto decode_res = tokenizer->decode(prev_token, output_token);
  ET_CHECK_OR_RETURN_ERROR(
      decode_res.ok(),
      InvalidState,
      "Tokenizer failed to decode first generated token: %lu",
      output_token);
  std::string full_response = std::move(decode_res.get());
  std::vector<uint64_t> full_response_tokens = {input_token};

  const auto vocab_size = tokenizer->vocab_size();
  const auto logits_type = llama_runtime.GetModelOptions().model_output_type;

  double gen_total_time_sec = 0;
  Timer timer_gen_token(
      [&](const auto elapsed_sec) { gen_total_time_sec += elapsed_sec; });

  // Print first output token
  std::cout << "\n[Real-time Response]" << std::endl;
  std::cout << full_response << std::flush;

  while (gen_tok_count++ < FLAGS_max_response &&
         llama_runtime.GetTokenIndex() < FLAGS_max_token_length) {
    timer_gen_token.Start();
    void* logits = llama_runtime.Run({output_token});
    timer_gen_token.End();

    prev_token = output_token;
    output_token = argmax(logits_type, logits, vocab_size);
    full_response_tokens.push_back(output_token);

    // Stop when output is EOS
    if (output_token == tokenizer->eos_tok()) {
      std::cout << "</eos>" << std::flush;
      break;
    }
    auto decode_res = tokenizer->decode(prev_token, output_token);
    ET_CHECK_OR_RETURN_ERROR(
        decode_res.ok(),
        InvalidState,
        "Tokenizer failed to decode generated token %lu",
        output_token);
    const std::string tok_str = std::move(decode_res.get());
    full_response += tok_str;
    std::cout << tok_str << std::flush;
  }

  std::cout << "\n\n[Generated Tokens]\n"
            << to_string(full_response_tokens) << std::endl;

  ET_LOG(
      Info,
      "Token generation speed: %f tok/s",
      gen_tok_count / gen_total_time_sec);

  return Error::Ok;
}

Error inference(
    LlamaRuntime& llama_runtime,
    const std::unique_ptr<Tokenizer>& tokenizer,
    const std::string& prompt) {
  // Tokenize input prompt
  auto encode_res = tokenizer->encode(prompt, kAddBos, kAddEos);
  ET_CHECK_OR_RETURN_ERROR(
      encode_res.ok(), InvalidState, "Tokenizer failed to encode prompt");
  const auto input_tokens = std::move(encode_res.get());

  std::cout << "\n[Input Prompt]\n" << prompt << std::endl;

  // Run prompt mode (pre-fill)
  auto prefill_res = digest_prompt(llama_runtime, tokenizer, input_tokens);
  ET_CHECK_OR_RETURN_ERROR(
      prefill_res.ok(), InvalidState, "Failed to digest prompt");
  const auto first_output_token = prefill_res.get();

  // run generation mode (decoding)
  return gen_response(llama_runtime, tokenizer, first_output_token);
}

std::unique_ptr<Tokenizer> load_tokenizer() {
  std::unique_ptr<Tokenizer> tokenizer;
  if (FLAGS_tokenizer_type == "bpe") {
    tokenizer = std::make_unique<BPETokenizer>();
  } else if (FLAGS_tokenizer_type == "tiktoken") {
    tokenizer = example::get_tiktoken_for_llama();
  }
  ET_CHECK_MSG(
      tokenizer, "Invalid tokenizer type: %s", FLAGS_tokenizer_type.c_str());
  tokenizer->load(FLAGS_tokenizer_path);
  return tokenizer;
}

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::string msg = "Extra commandline args:";
    for (int i = 1 /* skip argv[0] (program name) */; i < argc; i++) {
      msg += std::string(" ") + argv[i];
    }
    ET_LOG(Error, "%s", msg.c_str());
    return 1;
  }

  LlamaModelOptions model_options = get_model_options();
  LlamaModelPaths model_paths = get_model_paths();

  if (model_paths.prompt_model_paths.empty()) {
    model_options.prompt_token_batch_size = 1;
    ET_LOG(
        Info,
        "No prompt model paths provided, overriding prompt_token_batch_size to 1");
  }

  // Prepare timers
  Timer timer_init(
      [](const auto elapsed_sec) { ET_LOG(Info, "Model initialized."); });
  Timer timer_release(
      [](const auto elapsed_sec) { ET_LOG(Info, "Model released."); });

  LlamaRuntime llama_runtime;

  // Initialize model
  ET_LOG(Info, "Begin model loading.");
  timer_init.Start();
  const auto tokenizer = load_tokenizer();
  llama_runtime.Initialize(model_options, model_paths);
  timer_init.End();

  // Run model
  ET_CHECK_MSG(!FLAGS_prompt_file.empty(), "No prompt file provided.");
  std::string prompt = read_file(FLAGS_prompt_file);
  inference(llama_runtime, tokenizer, prompt);

  // Release model
  timer_release.Start();
  llama_runtime.Release();
  timer_release.End();

  return 0;
}
