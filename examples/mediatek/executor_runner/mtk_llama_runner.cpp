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

#include <executorch/examples/mediatek/executor_runner/mtk_llama_runner.h>
#include "executorch/backends/mediatek/runtime/include/NeuronBufferAllocator.h"

#include <ctime>
#include <iostream>
#include <memory>
#include <random>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
// #include <executorch/util/util.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/core/result.h>

#include "llama_runner/ModelChunk.h"
#include "llama_runner/Utils.h"
#include "llama_runner/llm_helper/include/llama_runner_values.h"
#include "llama_runner/llm_helper/include/llm_types.h"

static uint64_t MAX_RESPONSE = 50; // Maximum number of tokens to generate.
// Global BOS and EOS option for tokenization (encoding)
static constexpr int8_t kAddBos = 1;
static constexpr int8_t kAddEos = 0;

using namespace example::llm_helper;
using example::utils::argmax;
using example::utils::split;
using example::utils::Timer;
using example::utils::to_string;
using namespace mtk::vars;

namespace llm = ::executorch::extension::llm;

MTKLlamaRunner::MTKLlamaRunner(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const float temperature)
    : modeloptions_(get_model_options()), modelpaths_(get_model_paths()) {
  executorch::runtime::runtime_init();
  ET_LOG(
      Info,
      "Creating MTK Llama runner. Current it will self-load .pte, .bin, and .so files. Initiated runtime_init().");
}

Error MTKLlamaRunner::load() {
  if (is_loaded()) {
    return Error::Ok;
  }

  // Load tokenizer
  ET_LOG(Info, "Loading tokenizer.");
  tokenizer_ = load_tokenizer();
  ET_LOG(Info, "Complete loading tokenizer.");

  // Load prompt model
  runtime_ = std::make_unique<LlamaRuntime>();
  ET_LOG(Info, "Loading prompt model.");
  runtime_->Initialize(modeloptions_, modelpaths_);
  ET_LOG(Info, "Complete loading prompt model.");

  return Error::Ok;
}

bool MTKLlamaRunner::is_loaded() const {
  return tokenizer_ && runtime_;
}

Error MTKLlamaRunner::generate(
    const std::string& prompt,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

  // Wrap the token_callback with print function
  std::function<void(const std::string&)> wrapped_callback =
      [token_callback](const std::string& piece) {
        llm::safe_printf(piece.c_str());
        fflush(stdout);
        if (token_callback) {
          token_callback(piece);
        }
      };

  ET_LOG(Info, "Starting inference from MTKLlamaRunner");
  inference(*runtime_.get(), tokenizer_, prompt, wrapped_callback);
  ET_LOG(Info, "Completed inference from MTKLlamaRunner");

  return Error::Ok;
}

void MTKLlamaRunner::stop() {
  if (is_loaded()) {
    runtime_->Release();
  } else {
    ET_LOG(Error, "Llama Runtime is not loaded, cannot stop");
  }
}

LlamaModelOptions MTKLlamaRunner::get_model_options() {
  LlamaModelOptions options = {
      // Sizes
      .prompt_token_batch_size = PROMPT_TOKEN_BATCH_SIZE,
      .cache_size = CACHE_SIZE,
      .hidden_size = HIDDEN_SIZE,
      .num_head = NUM_HEAD,
      .num_layer = NUM_LAYER,
      .max_token_length = MAX_TOKEN_LENGTH,
      .rot_emb_base = ROT_EMB_BASE,

      // Types
      .model_input_type = MODEL_INPUT_TYPE,
      .model_output_type = MODEL_OUTPUT_TYPE,
      .cache_type = CACHE_TYPE,
      .mask_type = MASK_TYPE,
      .rot_emb_type = ROT_EMB_TYPE};
  ET_LOG(Info, "Completed get_model_options");
  return options;
}

LlamaModelPaths MTKLlamaRunner::get_model_paths() {
  LlamaModelPaths model_paths = {
      .tokenizer_path = TOKENIZER_PATH,
      .token_embedding_path = TOKEN_EMBEDDING_PATH,
      .prompt_model_paths = split(PROMPT_MODEL_PATHS, ','),
      .gen_model_paths = split(GEN_MODEL_PATHS, ',')};
  ET_LOG(Info, "Completed get_model_paths");
  return model_paths;
}

Result<uint64_t> MTKLlamaRunner::digest_prompt(
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

Error MTKLlamaRunner::gen_response(
    LlamaRuntime& llama_runtime,
    const std::unique_ptr<Tokenizer>& tokenizer,
    const uint64_t input_token,
    std::function<void(const std::string&)> token_callback) {
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
  token_callback(full_response);

  while (gen_tok_count++ < MAX_RESPONSE &&
         llama_runtime.GetTokenIndex() < modeloptions_.max_token_length) {
    timer_gen_token.Start();
    void* logits = llama_runtime.Run({output_token});
    timer_gen_token.End();

    prev_token = output_token;
    output_token = argmax(logits_type, logits, vocab_size);
    full_response_tokens.push_back(output_token);

    // Stop when output is EOS
    if (output_token == tokenizer->eos_tok()) {
      token_callback("</eos>");
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
    token_callback(tok_str);
  }

  std::cout << "\n\n[Generated Tokens]\n"
            << to_string(full_response_tokens) << std::endl;

  ET_LOG(
      Info,
      "Token generation speed: %f tok/s",
      gen_tok_count / gen_total_time_sec);

  return Error::Ok;
}

Error MTKLlamaRunner::inference(
    LlamaRuntime& llama_runtime,
    const std::unique_ptr<Tokenizer>& tokenizer,
    const std::string& prompt,
    std::function<void(const std::string&)> token_callback) {
  // Tokenize input prompt
  auto encode_res = tokenizer->encode(prompt, kAddBos, kAddEos);
  ET_CHECK_OR_RETURN_ERROR(
      encode_res.ok(), InvalidState, "Tokenizer failed to encode prompt");
  const auto input_tokens = std::move(encode_res.get());

  // Run prompt mode (pre-fill)
  auto prefill_res = digest_prompt(llama_runtime, tokenizer, input_tokens);
  ET_CHECK_OR_RETURN_ERROR(
      prefill_res.ok(), InvalidState, "Failed to digest prompt");
  const auto first_output_token = prefill_res.get();

  // run generation mode (decoding)
  return gen_response(
      llama_runtime, tokenizer, first_output_token, token_callback);
}

std::unique_ptr<Tokenizer> MTKLlamaRunner::load_tokenizer() {
  std::unique_ptr<Tokenizer> tokenizer;
  // Assumes that tokenizer type is Tiktoken
  tokenizer = example::get_tiktoken_for_llama();
  tokenizer->load(modelpaths_.tokenizer_path);
  return tokenizer;
}
