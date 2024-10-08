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

#include <executorch/runtime/platform/log.h>

#include "LlamaConfig.h"
#include "LlamaModelChunk.h"
#include "llm_helper/include/llm_types.h"

#include "llm_helper/include/rotary_embedding.h"
#include "llm_helper/include/token_embedding.h"

namespace example {

class LlamaRuntime {
 public:
  explicit LlamaRuntime() {}
  ~LlamaRuntime() {}

  void Initialize(
      const LlamaModelOptions& modelOptions,
      const LlamaModelPaths& modelPaths);

  void Release();

  void SwapModel(const size_t batchSize);

  void* Run(
      const std::vector<uint64_t>& inputTokens,
      const bool lastLogits = true);

  void Reset();

  size_t GetTokenBatchSize() const;

  size_t GetTokenIndex() const;

  const LlamaModelOptions& GetModelOptions() const;

 private:
  LlamaModelOptions mModelOptions;
  std::vector<std::unique_ptr<ModelChunk>> mLlamaModelChunks;
  std::unique_ptr<llm_helper::TokenEmbeddingLut> mTokenEmbLut;
  std::unique_ptr<llm_helper::RotaryEmbeddingMasterLut> mRotEmbMasterLut;
  size_t mTokenBatchSize = 1;
  size_t mTokenIndex = 0;
};

} // namespace example
