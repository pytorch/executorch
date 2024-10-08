/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#pragma once

#include <string>
#include <vector>

#include "llm_helper/include/llm_types.h"

namespace example {

using llm_helper::LLMType;

struct LlamaModelOptions {
  // Sizes
  size_t prompt_token_batch_size = 1;
  size_t cache_size = 1024;
  size_t hidden_size = 4096;
  size_t num_head = 32;
  size_t num_layer = 32;
  size_t max_token_length = 2048;
  double rot_emb_base = 10000.0f;

  // Types
  LLMType model_input_type = LLMType::INT16;
  LLMType model_output_type = LLMType::INT16;
  LLMType cache_type = LLMType::INT16;
  LLMType mask_type = LLMType::INT16;
  LLMType rot_emb_type = LLMType::INT16;
};

struct LlamaModelPaths {
  std::string tokenizer_path;
  std::string token_embedding_path;
  std::vector<std::string> prompt_model_paths;
  std::vector<std::string> gen_model_paths;
};

} // namespace example
