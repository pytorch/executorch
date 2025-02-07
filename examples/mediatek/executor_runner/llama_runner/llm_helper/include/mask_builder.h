/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#pragma once

#include "llm_types.h"

#include <string>

namespace example {
namespace llm_helper {

class MaskBuilder {
 public:
  explicit MaskBuilder(
      void* maskBuffer,
      const size_t maskSizeBytes,
      const LLMType maskType,
      const size_t cacheLength);

  ~MaskBuilder();

  // Build mask from scratch.
  void buildMask(const size_t tokenBatchSize, const size_t numSeenToken);

  // Only set mask to true for seen tokens.
  // Will fallback to buildMask if mask is not updatable.
  void updateMask(
      const size_t tokenBatchSize,
      const size_t numSeenToken,
      const size_t length);

  void notifyLeftPadding(const size_t padLength);

  void notifyRightPadding(const size_t padLength);

  // Mark mask as non-updatable which forces updateMask to call buildMask.
  void markMaskDirty();

  // Update the model input mask size. Use raw byte size to account for any HW
  // alignment.
  void updateMaskSize(const size_t sizeBytes);

 private:
  template <typename MaskType>
  void buildMask(const size_t tokenBatchSize, const size_t numSeenToken);

  template <typename MaskType>
  void updateMask(
      const size_t tokenBatchSize,
      const size_t numSeenToken,
      const size_t length);

  // Adjust mask for padded input, and returns whether mask is modified for
  // padding. Used by buildMask/updateMask.
  template <typename MaskType>
  bool adjustMaskForPadding(const size_t tokenBatchSize);

 private:
  void* mMaskBuffer;
  size_t mMaskSizeBytes;
  const LLMType kMaskType;
  const size_t kMaskTypeSize;
  const size_t kCacheLength;

  // Set by notifyLeftPadding/notifyRightPadding. Reset by adjustMaskForPadding.
  size_t mLeftPadLength = 0;
  size_t mRightPadLength = 0;

  bool mIsMaskUpdatable = false;
};

} // namespace llm_helper
} // namespace example
