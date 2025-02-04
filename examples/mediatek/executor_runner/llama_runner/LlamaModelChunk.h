/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>

#include "LlamaConfig.h"
#include "ModelChunk.h"
#include "llm_helper/include/llm_types.h"

#include "llm_helper/include/mask_builder.h"
#include "llm_helper/include/rotary_embedding.h"

namespace example {

using llm_helper::MaskBuilder;
using llm_helper::RotaryEmbeddingMasterLut;

using TensorShape = executorch::runtime::Span<const int32_t>;
using ModelIndexMap = std::unordered_map<size_t, size_t>;

// Llama decoder chunk
class LlamaModelChunk : public ModelChunk {
 private:
  static constexpr size_t kCacheLengthDim = 2;

 public:
  explicit LlamaModelChunk(
      const ModelPathMap& modelPathMap,
      const LlamaModelOptions& modelOptions,
      const size_t initBatchSize,
      const size_t numCache,
      const size_t numRotEmbInputs,
      const RotaryEmbeddingMasterLut* rotEmbMasterLut);

  ~LlamaModelChunk();

  virtual void Initialize() override;

  virtual void Run() override;

  virtual bool HotSwapModel(const size_t tokenBatchSize) override;

  void Reset();

  void SetLeftPadding(const size_t leftPadSize);

  void SetRightPadding(const size_t rightPadSize);

  void UpdatePosEmbAndMask(const size_t numInputToken);

  void AdvanceTokenIndex();

  size_t GetTokenIndex() const;

 private:
  void SetPosEmbed(const size_t tokenIndex);

  void InitMaskBuilder();

  void InitCache();

  void PrepareCacheIOs();

  size_t GetCacheStrideSize() const;

  size_t GetCacheNumRows() const;

  size_t GetLeftPadding() const;

  size_t GetRightPadding() const;

  void PaddingPostprocess();

  virtual void LeftPaddingCachePostprocess();

  virtual void RightPaddingCachePostprocess();

  virtual void RollbackCache(
      const size_t rollbackTokCount,
      const size_t numSeenToken);

 private:
  void CheckIoCount();

  size_t GetExpectedInputCount() const;

  size_t GetExpectedOutputCount() const;

 private:
  // Input/Output Indexes
  const size_t kMaskInputIndex;
  const std::vector<size_t> kRotEmbInputIndexes;
  const std::vector<size_t> kCacheInputIndexes;
  const std::vector<size_t> kCacheOutputIndexes;

  // Cache
  TensorShape mCacheShape;
  const LLMType kCacheType;
  const size_t kMaxTokenLength;
  const size_t kCacheLength;
  const size_t kCacheTypeSize;

  // Mask
  const LLMType kMaskType;

  // Padding
  size_t mCurrentPadSize = 0;
  enum class PaddingMode { LEFT, RIGHT };
  PaddingMode mPaddingMode = PaddingMode::RIGHT;

  // Lookup table for rotary embedding
  const RotaryEmbeddingMasterLut* kRotEmbMasterLut;

  // Mask builder
  std::unique_ptr<MaskBuilder> mMaskBuilder;

  // Keep track of token index. Its value can also be viewed as numSeenToken.
  size_t mCurrentTokenIndex = 0;
};

} // namespace example
