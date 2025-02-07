/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include "llm_helper/include/mask_builder.h"

#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

namespace example {
namespace llm_helper {

// Define mask values for different types
template <typename T>
struct MaskVal;

#define __DECL_MASK__(TYPE, TRUE_VAL, FALSE_VAL) \
  template <>                                    \
  struct MaskVal<TYPE> {                         \
    static constexpr TYPE kTrue = TRUE_VAL;      \
    static constexpr TYPE kFalse = FALSE_VAL;    \
  };

__DECL_MASK__(bool, true, false)
__DECL_MASK__(int16_t, 0, -32768)
__DECL_MASK__(__fp16, 0, -100)
__DECL_MASK__(float, 0, -100)
#undef __DECL_MASK__

MaskBuilder::MaskBuilder(
    void* maskBuffer,
    const size_t maskSizeBytes,
    const LLMType maskType,
    const size_t cacheLength)
    : mMaskBuffer(maskBuffer),
      mMaskSizeBytes(maskSizeBytes),
      kMaskType(maskType),
      kMaskTypeSize(getLLMTypeSize(maskType)),
      kCacheLength(cacheLength) {}

MaskBuilder::~MaskBuilder() {}

void MaskBuilder::updateMaskSize(const size_t sizeBytes) {
  mMaskSizeBytes = sizeBytes;
}

void MaskBuilder::markMaskDirty() {
  mIsMaskUpdatable = false;
}

template <typename MaskType>
void MaskBuilder::buildMask(
    const size_t tokenBatchSize,
    const size_t numSeenToken) {
  constexpr auto maskTrue = MaskVal<MaskType>::kTrue;
  constexpr auto maskFalse = MaskVal<MaskType>::kFalse;
  const size_t maskLength = kCacheLength + tokenBatchSize;

  // The mask is a combination (concat) of input cache mask and attention mask
  const size_t startTrueIdx =
      kCacheLength - std::min(kCacheLength, numSeenToken);

  const size_t rowSize = mMaskSizeBytes / tokenBatchSize / kMaskTypeSize;

  const size_t expectedMaskSizeBytes =
      tokenBatchSize * maskLength * kMaskTypeSize;
  // Use '<' instead of '!=' because mMaskSizeBytes may be padded by compiler to
  // fit HW
  if (mMaskSizeBytes < expectedMaskSizeBytes) {
    ET_LOG(
        Info,
        "Warn: Model input mask size (%zu) < mask size to be built (%zu). "
        "Please ensure your model options are set correctly.",
        mMaskSizeBytes,
        expectedMaskSizeBytes);
  }

  // There are tokenBatchSize number of rows
  for (size_t inTokIdx = 0; inTokIdx < tokenBatchSize; inTokIdx++) {
    const auto& rowIdx = inTokIdx; // For clarity
    auto curMaskBuffer =
        reinterpret_cast<MaskType*>(mMaskBuffer) + rowIdx * rowSize;
    size_t i = 0; // Buffer write index

    // Set the (rectangle) input cache mask
    while (i < startTrueIdx)
      curMaskBuffer[i++] = maskFalse;
    while (i < kCacheLength)
      curMaskBuffer[i++] = maskTrue;

    // Set the (triangle) attention mask
    const size_t attnTrueCount = inTokIdx + 1;
    for (size_t counter = 0; counter < attnTrueCount; counter++) {
      curMaskBuffer[i++] = maskTrue;
    }
    // Fill the remaining with False
    while (i < maskLength)
      curMaskBuffer[i++] = maskFalse;
  }

  // Modify mask for padding if needed. Mask is not updatable if modified for
  // padding.
  mIsMaskUpdatable = !adjustMaskForPadding<MaskType>(tokenBatchSize);
}

template <typename MaskType>
void MaskBuilder::updateMask(
    const size_t tokenBatchSize,
    const size_t numSeenToken,
    const size_t length) {
  constexpr auto maskTrue = MaskVal<MaskType>::kTrue;

  if (!mIsMaskUpdatable) {
    buildMask<MaskType>(tokenBatchSize, numSeenToken);
    return;
  }

  // Only set True for seen token
  const size_t trueCount = std::min(length, numSeenToken);
  if (!trueCount) {
    // Modify mask for padding if needed. Mask is not updatable if modified for
    // padding.
    mIsMaskUpdatable = !adjustMaskForPadding<MaskType>(tokenBatchSize);
    return;
  }

  // The mask is a combination (concat) of input cache mask and attention mask
  auto maskBuffer = reinterpret_cast<MaskType*>(mMaskBuffer);

  const size_t rowSize = mMaskSizeBytes / tokenBatchSize / kMaskTypeSize;

  // Only modify the left rectangle part
  const size_t startTrueOffset =
      kCacheLength - std::min(kCacheLength, numSeenToken);

  for (size_t inTokIdx = 0; inTokIdx < tokenBatchSize; inTokIdx++) {
    const auto& rowIdx = inTokIdx; // For clarity
    auto curMaskBuffer = maskBuffer + rowIdx * rowSize + startTrueOffset;
    std::fill(curMaskBuffer, curMaskBuffer + trueCount, maskTrue);
  }
  // Modify mask for padding if needed. Mask is not updatable if modified for
  // padding.
  mIsMaskUpdatable = !adjustMaskForPadding<MaskType>(tokenBatchSize);
}

void MaskBuilder::buildMask(
    const size_t tokenBatchSize,
    const size_t numSeenToken) {
  switch (kMaskType) {
    case LLMType::INT16:
      buildMask<int16_t>(tokenBatchSize, numSeenToken);
      return;
    case LLMType::FP16:
      buildMask<__fp16>(tokenBatchSize, numSeenToken);
      return;
    case LLMType::FP32:
      buildMask<float>(tokenBatchSize, numSeenToken);
      return;
    default:
      break;
  }
  ET_LOG(
      Fatal,
      "Attempting to build mask with type %s. Supported types are INT16, FP16, FP32.",
      getLLMTypeName(kMaskType));
}

void MaskBuilder::updateMask(
    const size_t tokenBatchSize,
    const size_t numSeenToken,
    const size_t length) {
  switch (kMaskType) {
    case LLMType::INT16:
      updateMask<int16_t>(tokenBatchSize, numSeenToken, length);
      return;
    case LLMType::FP16:
      updateMask<__fp16>(tokenBatchSize, numSeenToken, length);
      return;
    case LLMType::FP32:
      updateMask<float>(tokenBatchSize, numSeenToken, length);
      return;
    default:
      break;
  }
  ET_LOG(
      Fatal,
      "Attempting to update with an unsupported mask type. "
      "Supported types are INT16, FP16, FP32.");
}

void MaskBuilder::notifyLeftPadding(const size_t padLength) {
  ET_CHECK_MSG(
      mRightPadLength == 0,
      "Attempting to set left pad after right pad has been set.");
  if (mLeftPadLength > 0) {
    ET_LOG(
        Info,
        "Warn: Calling notifyLeftPadding() multiple times before building/updating mask.");
  }
  mLeftPadLength = padLength;
}

void MaskBuilder::notifyRightPadding(const size_t padLength) {
  ET_CHECK_MSG(
      mLeftPadLength == 0,
      "Attempting to set right pad after left pad has been set.");
  if (mRightPadLength > 0) {
    ET_LOG(
        Info,
        "Warn: Calling notifyLeftPadding() multiple times before building/updating mask.");
  }
  mRightPadLength = padLength;
}

template <typename MaskType>
bool MaskBuilder::adjustMaskForPadding(const size_t tokenBatchSize) {
  if (mLeftPadLength + mRightPadLength == 0) {
    return false; // No need to modify mask since no padding
  }
  ET_DCHECK_MSG(
      mLeftPadLength == 0 || mRightPadLength == 0,
      "Only allow setting either left or right pad");
  constexpr auto maskFalse = MaskVal<MaskType>::kFalse;
  const size_t maskLength = kCacheLength + tokenBatchSize;

  // The mask is a combination (concat) of input cache mask and attention mask
  auto maskBuffer = reinterpret_cast<MaskType*>(mMaskBuffer);

  const size_t rowSize = mMaskSizeBytes / tokenBatchSize / kMaskTypeSize;

  if (mLeftPadLength > 0) {
    // Mask the padded rows
    for (size_t inTokIdx = 0; inTokIdx < mLeftPadLength; inTokIdx++) {
      auto curMaskBuffer = maskBuffer + inTokIdx * rowSize;
      std::fill(curMaskBuffer, curMaskBuffer + maskLength, maskFalse);
    }
    // Mask the padded attention region
    for (size_t inTokIdx = mLeftPadLength; inTokIdx < tokenBatchSize;
         inTokIdx++) {
      auto curMaskBuffer = maskBuffer + inTokIdx * rowSize + kCacheLength;
      // Anything from inTokIdx + 1 onwards is already False, so can skip them.
      const size_t maskPadCount = std::min(mLeftPadLength, inTokIdx + 1);
      std::fill(curMaskBuffer, curMaskBuffer + maskPadCount, maskFalse);
    }
    mLeftPadLength = 0; // Reset pad length
  } else if (mRightPadLength > 0) {
    // Mask the padded rows
    const auto startIdx = tokenBatchSize - mRightPadLength;
    for (size_t inTokIdx = startIdx; inTokIdx < tokenBatchSize; inTokIdx++) {
      auto curMaskBuffer = maskBuffer + inTokIdx * rowSize;
      std::fill(curMaskBuffer, curMaskBuffer + maskLength, maskFalse);
    }
    mRightPadLength = 0; // Reset pad length
  }
  return true; // Mask is modified for padding
}

} // namespace llm_helper
} // namespace example
