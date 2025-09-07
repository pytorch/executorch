/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include <numeric>
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
#include "LlamaModelChunk.h"
#include "Utils.h"
#include "llm_helper/include/llm_types.h"

#include "llm_helper/include/mask_builder.h"
#include "llm_helper/include/rotary_embedding.h"

namespace example {

inline std::vector<size_t> getIndexRange(
    const size_t startIndex,
    const size_t count) {
  std::vector<size_t> indexes(count);
  size_t counter = startIndex;
  for (auto& idx : indexes) {
    idx = counter++;
  }
  return indexes;
}

LlamaModelChunk::LlamaModelChunk(
    const ModelPathMap& modelPathMap,
    const LlamaModelOptions& modelOptions,
    const bool useSharedWeights,
    const size_t initBatchSize,
    const size_t numCache,
    const size_t numRotEmbInputs,
    const bool enableSWA,
    const RotaryEmbeddingMasterLut* rotEmbMasterLut)
    : ModelChunk(modelPathMap, initBatchSize),
      kIsSharedWeightsUsed(useSharedWeights),
      kMaxTokenLength(modelOptions.max_token_length),
      kCacheLength(modelOptions.cache_size),
      kMaskType(modelOptions.mask_type),
      kRotEmbMasterLut(rotEmbMasterLut),
      kCacheType(modelOptions.cache_type),
      kWindowSize(modelOptions.window_size),

      kRotEmbInputCount(numRotEmbInputs),
      kCacheCount(numCache),
      enableSWA(enableSWA),
      kCacheTypeSize(llm_helper::getLLMTypeSize(kCacheType)) {}

std::string LlamaModelChunk::SelectMethod(
    const std::vector<std::string>& methodNames) const {
  const size_t curTokenSize = GetModelId();
  for (const auto& methodName : methodNames) {
    const auto matches = utils::extract_substr(methodName, "([0-9]+)t[0-9]+c");
    ET_CHECK_MSG(
        matches.size() == 2, "Invalid method name: %s", methodName.c_str());
    // Extract the first match group as token size
    const size_t methodTokenSize =
        static_cast<size_t>(std::atol(matches[1].c_str()));
    if (curTokenSize == methodTokenSize) {
      ET_LOG(
          Debug,
          "Selected method \"%s\" for token size %zu",
          methodName.c_str(),
          curTokenSize);
      return methodName;
    }
  }
  ET_LOG(
      Error,
      "Unable to find suitable method, fallback to use the first method.");
  return {};
}

LlamaModelChunk::~LlamaModelChunk() {}

void LlamaModelChunk::Initialize() {
  LoadModels();
  GetModelIoInfo();
  defineIOs();
  CheckIoCount();
  PrepareCacheIOs();
  AllocateIoBuffers();
  InitMaskBuilder();
  InitCache();

  SetBackendInputs();
  SetBackendOutputs();
  mIsInitialized = true;
}

void LlamaModelChunk::defineIOs() {
  // Inputs
  defineInput(IOKind::Embedding);
  defineInput(IOKind::Mask);
  defineInput(IOKind::SWAMask, enableSWA);
  defineInput(IOKind::RotEmb, kRotEmbInputCount);
  defineInput(IOKind::KVCache, kCacheCount);
  // Outputs
  defineOutput(IOKind::Logits);
  defineOutput(IOKind::KVCache, kCacheCount);
}

void LlamaModelChunk::defineInput(const IOKind kind, const size_t count) {
  ET_CHECK_MSG(
      !hasInput(kind),
      "Input kind has already been defined: %d",
      static_cast<int>(kind));
  const auto startIdx = mExpectedNumInputs;
  std::vector<size_t> indexes(count);
  std::iota(indexes.begin(), indexes.end(), startIdx);
  mInputIndexes[kind] = std::move(indexes);
  mExpectedNumInputs += count;
}

void LlamaModelChunk::defineOutput(const IOKind kind, const size_t count) {
  ET_CHECK_MSG(
      !hasOutput(kind),
      "Output kind has already been defined: %d",
      static_cast<int>(kind));
  const auto startIdx = mExpectedNumOutputs;
  std::vector<size_t> indexes(count);
  std::iota(indexes.begin(), indexes.end(), startIdx);
  mOutputIndexes[kind] = std::move(indexes);
  mExpectedNumOutputs += count;
}

bool LlamaModelChunk::hasInput(const IOKind kind) const {
  return (mInputIndexes.find(kind) != mInputIndexes.end()) &&
      !mInputIndexes.at(kind).empty();
}

bool LlamaModelChunk::hasOutput(const IOKind kind) const {
  return (mOutputIndexes.find(kind) != mOutputIndexes.end()) &&
      !mOutputIndexes.at(kind).empty();
}

const std::vector<size_t>& LlamaModelChunk::getInputIndexes(
    const IOKind kind) const {
  ET_CHECK_MSG(
      hasInput(kind), "Check failed for input kind %d", static_cast<int>(kind));
  return mInputIndexes.at(kind);
}

const std::vector<size_t>& LlamaModelChunk::getOutputIndexes(
    const IOKind kind) const {
  ET_CHECK_MSG(
      hasOutput(kind),
      "Check failed for output kind %d",
      static_cast<int>(kind));
  return mOutputIndexes.at(kind);
}

size_t LlamaModelChunk::getInputIndex(const IOKind kind, const size_t pos)
    const {
  const auto& inputIndexes = getInputIndexes(kind);
  ET_CHECK_MSG(
      pos < inputIndexes.size(), "getInputIndex(): Index out of range");
  return inputIndexes[pos];
}

size_t LlamaModelChunk::getOutputIndex(const IOKind kind, const size_t pos)
    const {
  const auto& outputIndexes = getOutputIndexes(kind);
  ET_CHECK_MSG(
      pos < outputIndexes.size(), "getOutputIndex(): Index out of range");
  return outputIndexes[pos];
}

size_t LlamaModelChunk::getNumInputsFor(const IOKind kind) const {
  if (!hasInput(kind)) {
    return 0;
  }
  return mInputIndexes.at(kind).size();
}

size_t LlamaModelChunk::getNumOutputsFor(const IOKind kind) const {
  if (!hasOutput(kind)) {
    return 0;
  }
  return mOutputIndexes.at(kind).size();
}

void LlamaModelChunk::CheckIoCount() {
  const auto& method = GetModelMethod();
  const size_t modelInputCount = method.inputs_size();
  const size_t modelOutputCount = method.outputs_size();
  ET_CHECK_MSG(
      modelInputCount == mExpectedNumInputs,
      "Number of inputs does not match (expected %zu but got %zu).",
      mExpectedNumInputs,
      modelInputCount);
  ET_CHECK_MSG(
      modelOutputCount == mExpectedNumOutputs,
      "Number of outputs does not match (expected %zu but got %zu).",
      mExpectedNumOutputs,
      modelOutputCount);
}

bool LlamaModelChunk::HotSwapModel(const size_t tokenBatchSize) {
  const auto status = ModelChunk::HotSwapModel(tokenBatchSize);

  // Force rebuild mask because different batch size values will produce
  // different mask shapes.
  mMaskBuilder->markMaskDirty();

  // Update mask size
  const auto newMaskSizeBytes =
      mInputBufferInfos[getInputIndex(IOKind::Mask)].nbytesUsed;
  mMaskBuilder->updateMaskSize(newMaskSizeBytes);

  return status;
}

void LlamaModelChunk::Reset() {
  mCurrentPadSize = 0;
  mCurrentTokenIndex = 0;
  InitCache(); // Reset cache to zeros
}

void LlamaModelChunk::SetLeftPadding(const size_t leftPadSize) {
  mCurrentPadSize = leftPadSize;
  mPaddingMode = PaddingMode::LEFT;

  // Notify mask builder about padding
  mMaskBuilder->notifyLeftPadding(leftPadSize);
}

void LlamaModelChunk::SetRightPadding(const size_t rightPadSize) {
  mCurrentPadSize = rightPadSize;
  mPaddingMode = PaddingMode::RIGHT;

  // Notify mask builder about padding
  mMaskBuilder->notifyRightPadding(rightPadSize);
}

size_t LlamaModelChunk::GetLeftPadding() const {
  return (mPaddingMode == PaddingMode::LEFT) ? mCurrentPadSize : 0;
}

size_t LlamaModelChunk::GetRightPadding() const {
  return (mPaddingMode == PaddingMode::RIGHT) ? mCurrentPadSize : 0;
}

void LlamaModelChunk::PaddingPostprocess() {
  if (mCurrentPadSize == 0) {
    return;
  }

  if (mPaddingMode == PaddingMode::RIGHT) {
    RightPaddingCachePostprocess();
  } else if (mPaddingMode == PaddingMode::LEFT) {
    LeftPaddingCachePostprocess();
  }
}

void LlamaModelChunk::LeftPaddingCachePostprocess() {
  // NOTE: This part might not actually be needed

  // Stride size is same across caches
  const size_t strideSizeBytes = GetCacheStrideSize();
  const size_t rowSize = kCacheLength * strideSizeBytes;

  const size_t numRows = GetCacheNumRows();

  const size_t offset = (kCacheLength - mTokenBatchSize) * strideSizeBytes;
  const size_t zeroCount = mCurrentPadSize * strideSizeBytes;

  // Fill padded sections with zeros
  for (const auto cacheInputIdx : getInputIndexes(IOKind::KVCache)) {
    auto cacheBuffer =
        reinterpret_cast<char*>(mInputBufferInfos[cacheInputIdx].data);
    for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
      // cacheBufRow points at the start of row
      auto cacheBufRow = cacheBuffer + rowIdx * rowSize;
      std::memset(cacheBufRow + offset, 0, zeroCount);
    }
  }
}

void LlamaModelChunk::RightPaddingCachePostprocess() {
  // NOTE: AdvanceTokenIndex() haven't been called for this inference step yet.
  const size_t numSeenToken = mCurrentTokenIndex + mTokenBatchSize;
  RollbackCache(mCurrentPadSize, numSeenToken);
}

void LlamaModelChunk::RollbackCache(
    const size_t rollbackTokCount,
    const size_t numSeenToken) {
  if (rollbackTokCount == 0) {
    return; // do nothing
  }

  const size_t numSeenTokenAlive = std::min(numSeenToken, kCacheLength);
  const size_t firstNonEmptyIdx = kCacheLength - numSeenTokenAlive;
  const size_t preserveTokCount = (numSeenTokenAlive > rollbackTokCount)
      ? numSeenTokenAlive - rollbackTokCount
      : 0;

  if (!preserveTokCount) {
    // Clear cache to zeros
    InitCache();
    return;
  }

  const size_t strideSizeBytes = GetCacheStrideSize();
  const size_t rowSize = kCacheLength * strideSizeBytes;
  const size_t numRows = GetCacheNumRows();

  // Shift right and truncate rollbackTokCount, then fill left with zeros
  for (const auto cacheInputIdx : getInputIndexes(IOKind::KVCache)) {
    auto cacheBuffer =
        reinterpret_cast<char*>(mInputBufferInfos[cacheInputIdx].data);

    for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
      // Get the addr pointing to the start of row
      auto cacheBufRow = cacheBuffer + rowIdx * rowSize;

      // Move right for the section to be preserved
      const size_t dstOffset =
          strideSizeBytes * (firstNonEmptyIdx + rollbackTokCount);
      const size_t srcOffset = strideSizeBytes * firstNonEmptyIdx;
      const size_t preserveSize = strideSizeBytes * preserveTokCount;
      ET_DCHECK(dstOffset + preserveSize <= rowSize);
      std::memmove(
          cacheBufRow + dstOffset, cacheBufRow + srcOffset, preserveSize);

      // Then fill zeros to the section being moved out
      const size_t offset = firstNonEmptyIdx * strideSizeBytes;
      const size_t zeroCount = rollbackTokCount * strideSizeBytes;
      std::memset(cacheBufRow + offset, 0, zeroCount);
    }
  }
}

void LlamaModelChunk::UpdatePosEmbAndMask(const size_t numInputToken) {
  if (mCurrentTokenIndex + numInputToken > kMaxTokenLength) {
    ET_LOG(
        Fatal,
        "Attempting to generate tokens exceeding the supported max token length (%zu)",
        kMaxTokenLength);
  }
  if (mCurrentTokenIndex > 0 && GetLeftPadding() > 0) {
    ET_LOG(Fatal, "Left-padding is only allowed in the first prompt pass.");
  }
  auto isMaskUpdatable = mMaskBuilder->getMaskUpdateStatus();
  if (enableSWA) {
    const auto& swaMaskBufferInfo =
        mInputBufferInfos[getInputIndex(IOKind::SWAMask)];
    const auto swaMaskBuffer = swaMaskBufferInfo.data;
    const auto swaMaskSizeBytes = swaMaskBufferInfo.nbytesUsed;
    mMaskBuilder->setMaskBuffer(swaMaskBuffer, swaMaskSizeBytes);
    mMaskBuilder->enableSlidingWindow(kWindowSize);
    mMaskBuilder->updateMask(
        mTokenBatchSize, mCurrentTokenIndex, numInputToken);
  }
  // Pass same isMaskUpdatable to both mask
  mMaskBuilder->setIsMaskUpdatable(isMaskUpdatable);
  const auto& maskBufferInfo = mInputBufferInfos[getInputIndex(IOKind::Mask)];
  const auto maskBuffer = maskBufferInfo.data;
  const auto maskSizeBytes = maskBufferInfo.nbytesUsed;
  mMaskBuilder->setMaskBuffer(maskBuffer, maskSizeBytes);
  mMaskBuilder->disableSlidingWindow();
  mMaskBuilder->updateMask(mTokenBatchSize, mCurrentTokenIndex, numInputToken);

  mMaskBuilder->resetPadLength();
  SetPosEmbed(mCurrentTokenIndex);
}

void LlamaModelChunk::AdvanceTokenIndex() {
  // Exclude padded tokens
  const auto numValidInputToken = mTokenBatchSize - mCurrentPadSize;
  mCurrentTokenIndex += numValidInputToken;

  // Reset padding size
  mCurrentPadSize = 0;
}

size_t LlamaModelChunk::GetTokenIndex() const {
  return mCurrentTokenIndex;
}

void LlamaModelChunk::Run() {
  UpdatePosEmbAndMask(mTokenBatchSize);
  ModelChunk::Run();
  PaddingPostprocess();
  AdvanceTokenIndex();
}

void LlamaModelChunk::SetPosEmbed(const size_t tokenIndex) {
  if (tokenIndex >= kMaxTokenLength) {
    ET_LOG(
        Fatal,
        "Attempting to set rotaty embedding using index exceeding the supported max token length "
        "(%zu)",
        kMaxTokenLength);
  }

  auto getRotEmbInputs = [&]() {
    std::vector<void*> rotEmbInputs;
    const auto& RotEmbInputIndexes = getInputIndexes(IOKind::RotEmb);
    rotEmbInputs.reserve(RotEmbInputIndexes.size());
    for (const auto inputIdx : RotEmbInputIndexes)
      rotEmbInputs.push_back(mInputBufferInfos[inputIdx].data);
    return rotEmbInputs;
  };
  kRotEmbMasterLut->setEmbed(
      getRotEmbInputs(),
      tokenIndex,
      mTokenBatchSize,
      GetLeftPadding(),
      GetRightPadding());
}

void LlamaModelChunk::PrepareCacheIOs() {
  // Get cache shape
  const auto method_meta = GetModelMethod().method_meta();
  const auto firstInCacheIdx = getInputIndex(IOKind::KVCache);
  mCacheShape = method_meta.input_tensor_meta(firstInCacheIdx)->sizes();

  // Link cache IOs
  const size_t numCaches = getNumInputsFor(IOKind::KVCache);
  for (size_t i = 0; i < numCaches; i++) {
    this->LinkModelIO(
        getInputIndex(IOKind::KVCache, i), getOutputIndex(IOKind::KVCache, i));
  }
}

size_t LlamaModelChunk::GetCacheNumRows() const {
  return std::reduce(
      mCacheShape.begin(),
      mCacheShape.begin() + kCacheLengthDim,
      1,
      std::multiplies<>());
}

size_t LlamaModelChunk::GetCacheStrideSize() const {
  return std::reduce(
      mCacheShape.begin() + kCacheLengthDim + 1,
      mCacheShape.end(),
      kCacheTypeSize,
      std::multiplies<>());
}

void LlamaModelChunk::InitMaskBuilder() {
  mMaskBuilder = std::make_unique<MaskBuilder>(kMaskType, kCacheLength);
  // SWA Mask
  if (enableSWA) {
    const auto& swaMaskBufferInfo =
        mInputBufferInfos[getInputIndex(IOKind::SWAMask)];
    const auto swaMaskBuffer = swaMaskBufferInfo.data;
    const auto swaMaskSizeBytes = swaMaskBufferInfo.nbytesUsed;
    mMaskBuilder->setMaskBuffer(swaMaskBuffer, swaMaskSizeBytes);
    mMaskBuilder->enableSlidingWindow(kWindowSize);
    mMaskBuilder->buildMask(mTokenBatchSize, mCurrentTokenIndex);
  }
  // Global Mask
  const auto& maskBufferInfo = mInputBufferInfos[getInputIndex(IOKind::Mask)];
  const auto maskBuffer = maskBufferInfo.data;
  const auto maskSizeBytes = maskBufferInfo.nbytesUsed;
  mMaskBuilder->setMaskBuffer(maskBuffer, maskSizeBytes);
  mMaskBuilder->disableSlidingWindow();
  mMaskBuilder->buildMask(mTokenBatchSize, mCurrentTokenIndex);

  mMaskBuilder->resetPadLength();
}

void LlamaModelChunk::InitCache() {
  // Zero initialization
  for (const auto cacheIdx : getInputIndexes(IOKind::KVCache)) {
    const auto& inputCacheInfo = mInputBufferInfos[cacheIdx];
    char* cacheBuffer = reinterpret_cast<char*>(inputCacheInfo.data);
    const size_t cacheSizeBytes = inputCacheInfo.nbytes;
    std::memset(cacheBuffer, 0, cacheSizeBytes);
  }
}

} // namespace example
