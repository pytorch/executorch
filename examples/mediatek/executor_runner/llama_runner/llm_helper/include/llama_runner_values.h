/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Contains values that are used by the mtk_llama_runner.cpp

#pragma once

namespace mtk::vars {
  using example::llm_helper::LLMType;

  // Sizes
  const size_t PROMPT_TOKEN_BATCH_SIZE = 128;
  const size_t CACHE_SIZE = 512;
  const size_t HIDDEN_SIZE = 4096;
  const size_t NUM_HEAD = 32;
  const size_t NUM_LAYER = 32;
  const size_t MAX_TOKEN_LENGTH = 8192;
  const double ROT_EMB_BASE = 500000;

  // Types
  const LLMType MODEL_INPUT_TYPE = LLMType::FP32;
  const LLMType MODEL_OUTPUT_TYPE = LLMType::FP32;
  const LLMType CACHE_TYPE = LLMType::FP32;
  const LLMType MASK_TYPE = LLMType::FP32;
  const LLMType ROT_EMB_TYPE = LLMType::FP32;

  // Paths
  const std::string TOKENIZER_PATH="/data/local/tmp/et-mtk/llama3/tokenizer.model";
  const std::string TOKEN_EMBEDDING_PATH="/data/local/tmp/et-mtk/llama3/embedding_llama3-8B-instruct_fp32.bin";

  // Comma-Separated Paths
  const std::string PROMPT_MODEL_PATHS="/data/local/tmp/et-mtk/llama3/llama3-8B-instruct_A16W4_4_chunks_128t512c_0.pte,/data/local/tmp/et-mtk/llama3/llama3-8B-instruct_A16W4_4_chunks_128t512c_1.pte,/data/local/tmp/et-mtk/llama3/llama3-8B-instruct_A16W4_4_chunks_128t512c_2.pte,/data/local/tmp/et-mtk/llama3/llama3-8B-instruct_A16W4_4_chunks_128t512c_3.pte,";

  // Comma-Separated Paths
  const std::string GEN_MODEL_PATHS="/data/local/tmp/et-mtk/llama3/llama3-8B-instruct_A16W4_4_chunks_1t512c_0.pte,/data/local/tmp/et-mtk/llama3/llama3-8B-instruct_A16W4_4_chunks_1t512c_1.pte,/data/local/tmp/et-mtk/llama3/llama3-8B-instruct_A16W4_4_chunks_1t512c_2.pte,/data/local/tmp/et-mtk/llama3/llama3-8B-instruct_A16W4_4_chunks_1t512c_3.pte,";

} // namespace mtk:vars
