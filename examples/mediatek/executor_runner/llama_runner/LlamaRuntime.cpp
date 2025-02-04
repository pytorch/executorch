/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include <string>
#include <thread>
#include <vector>

#include <executorch/runtime/platform/log.h>

#include "LlamaRuntime.h"
#include "Utils.h"

#include "llm_helper/include/rotary_embedding.h"
#include "llm_helper/include/token_embedding.h"

namespace example {

void LlamaRuntime::Initialize(
    const LlamaModelOptions& modelOptions,
    const LlamaModelPaths& modelPaths) {
  mModelOptions = modelOptions;
  const size_t numChunk = modelPaths.gen_model_paths.size();
  const size_t numCache = 2 * modelOptions.num_layer / numChunk;
  ET_CHECK_MSG(numChunk > 0, "No model to initialize");

  // Initialize rotary embedding master lookup table
  const size_t rotEmbDim = modelOptions.hidden_size / modelOptions.num_head;
  mRotEmbMasterLut = std::make_unique<llm_helper::RotaryEmbeddingMasterLut>(
      modelOptions.rot_emb_type,
      modelOptions.max_token_length,
      rotEmbDim,
      modelOptions.rot_emb_base);
  mRotEmbMasterLut->generate();

  constexpr size_t numRotEmbInputs = 1;
  const bool usePromptModel = !modelPaths.prompt_model_paths.empty();
  const size_t initBatchSize =
      usePromptModel ? modelOptions.prompt_token_batch_size : 1;
  mTokenBatchSize = initBatchSize;

  for (size_t chunkIdx = 0; chunkIdx < numChunk; chunkIdx++) {
    ModelPathMap modelPathMap;
    auto addModelPath = [&](const auto& modelPaths, const size_t batchSize) {
      if (modelPaths.empty())
        return;
      modelPathMap[batchSize] = modelPaths[chunkIdx];
    };
    addModelPath(
        modelPaths.prompt_model_paths, modelOptions.prompt_token_batch_size);
    addModelPath(modelPaths.gen_model_paths, 1);
    auto llamaChunk = std::make_unique<LlamaModelChunk>(
        modelPathMap,
        modelOptions,
        initBatchSize,
        numCache,
        numRotEmbInputs,
        mRotEmbMasterLut.get());
    mLlamaModelChunks.push_back(std::move(llamaChunk));
  }

  for (size_t i = 0; i < numChunk; i++) {
    auto& modelChunk = mLlamaModelChunks[i];
    if (i > 0) {
      const auto& prevModelChunk = mLlamaModelChunks[i - 1];
      modelChunk->SetInputBuffer(prevModelChunk->GetOutputBuffer());
    }
    modelChunk->Initialize();
    // modelChunk->LogIoSummary();
  }

  // NOTE: Token embedding type here is assumed to follow the model input
  // embedding type.
  mTokenEmbLut = std::make_unique<llm_helper::TokenEmbeddingLut>(
      modelPaths.token_embedding_path,
      modelOptions.model_input_type,
      modelOptions.hidden_size);

  // Link first chunk emb input to token emb lut output
  const auto& tokenEmbInput = mLlamaModelChunks.front()->GetInputBuffer();
  mTokenEmbLut->setOutput(tokenEmbInput.data, tokenEmbInput.nbytes);
}

void LlamaRuntime::Release() {
  for (auto& llamaChunk : mLlamaModelChunks) {
    llamaChunk->Release();
  }
  mLlamaModelChunks.clear();
  mRotEmbMasterLut.reset();
  mTokenEmbLut.reset();
}

void LlamaRuntime::SwapModel(const size_t batchSize) {
  auto hotSwapChunk = [&](const auto chunkIdx) {
    const auto status = mLlamaModelChunks[chunkIdx]->HotSwapModel(batchSize);
    if (!status)
      ET_LOG(Error, "Hot swapping failed on chunk %zu", chunkIdx);
  };

  // Use multi-threading to speedup model swapping
  std::vector<std::thread> threads;
  for (size_t i = 0; i < mLlamaModelChunks.size(); i++)
    threads.emplace_back(hotSwapChunk, i);
  for (size_t i = 0; i < mLlamaModelChunks.size(); i++)
    threads[i].join();

  mTokenBatchSize = batchSize;
}

void LlamaRuntime::Reset() {
  for (auto& modelChunk : mLlamaModelChunks) {
    static_cast<LlamaModelChunk*>(modelChunk.get())->Reset();
  }
  mTokenIndex = 0;
}

void* LlamaRuntime::Run(
    const std::vector<uint64_t>& inputTokens,
    const bool lastLogits) {
  const auto& firstLlamaChunk = mLlamaModelChunks.front();
  const auto tokenIndex =
      static_cast<LlamaModelChunk*>(firstLlamaChunk.get())->GetTokenIndex();
  const auto numNewInputToken = inputTokens.size();

  ET_CHECK_MSG(
      numNewInputToken <= mTokenBatchSize,
      "Input token length (%zu) > model token batch size (%zu)",
      numNewInputToken,
      mTokenBatchSize);

  // Handle padding
  auto curInputTokens = inputTokens; // Make a copy
  const size_t padSize = mTokenBatchSize - numNewInputToken;
  constexpr uint64_t padToken = 0;

  // Use left-padding if possible as it has lower overhead than right-padding.
  // Right-padding involves cache shifting which incurs additional overhead.
  const bool isLeftPadAllowed = (tokenIndex == 0);
  if (padSize > 0) {
    if (isLeftPadAllowed) {
      // Pad left since the cache is fresh new.
      curInputTokens.insert(curInputTokens.begin(), padSize, padToken);
    } else {
      // Pad right since left side of cache is occupied either by loaded cache
      // or previous inference pass.
      curInputTokens.insert(curInputTokens.end(), padSize, padToken);
    }
    ET_LOG(Debug, "Padding size = %zu", padSize);
  }

  // Begin inference flow

  // Lookup token embedding
  mTokenEmbLut->lookupEmbedding(curInputTokens);

  // Decoder chunks
  for (auto& modelChunk : mLlamaModelChunks) {
    auto llamaChunk = static_cast<LlamaModelChunk*>(modelChunk.get());

    // Set padding if needed.
    if (isLeftPadAllowed)
      llamaChunk->SetLeftPadding(padSize);
    else
      llamaChunk->SetRightPadding(padSize);

    // Run model chunk
    llamaChunk->Run();
  }

  // Only consider valid tokens by ignoring padding
  mTokenIndex += inputTokens.size();

  // Return logits
  const auto& finalChunk = mLlamaModelChunks.back();
  const auto logitsBuffer = finalChunk->GetOutputBuffer();
  const auto logitsData = reinterpret_cast<char*>(logitsBuffer.data);
  const auto logitsSize = logitsBuffer.nbytesUsed;
  size_t offset = 0;
  const size_t rightPadSize = !isLeftPadAllowed * padSize;
  if (lastLogits && mTokenBatchSize > 1) {
    offset =
        (logitsSize / mTokenBatchSize) * (mTokenBatchSize - 1 - rightPadSize);
    ET_DCHECK(offset <= logitsSize);
  }
  return logitsData + offset;
}

size_t LlamaRuntime::GetTokenBatchSize() const {
  return mTokenBatchSize;
}

size_t LlamaRuntime::GetTokenIndex() const {
  return mTokenIndex;
}

const LlamaModelOptions& LlamaRuntime::GetModelOptions() const {
  return mModelOptions;
}

} // namespace example
