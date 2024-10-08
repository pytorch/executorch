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
#include <vector>

namespace example {
namespace llm_helper {

class RotaryEmbeddingMasterLut {
 public:
  RotaryEmbeddingMasterLut(
      const LLMType rotEmbType,
      const size_t length,
      const size_t headDim,
      const float rotBase = 10000.0,
      const float ntkScale = 1.0);

  virtual ~RotaryEmbeddingMasterLut() {}

  void load(const std::string& sinMasterPath, const std::string& cosMasterPath);

  void generate();

  template <typename RotEmbType>
  void generate();

  virtual void setEmbed(
      std::vector<void*> rotEmbedBuffers,
      const size_t tokenIndex,
      const size_t tokenBatchSize = 1,
      const size_t leftPadLength = 0,
      const size_t rightPadLength = 0) const;

  // Single rot emb input with combined cos & sin
  void setEmbed(
      void* rotEmbedBuffer,
      const size_t tokenIndex,
      const size_t tokenBatchSize = 1,
      const size_t leftPadLength = 0,
      const size_t rightPadLength = 0) const;

  // Two rot emb inputs for separated cos & sin
  void setEmbed(
      void* rotEmbedCosBuffer,
      void* rotEmbedSinBuffer,
      const size_t tokenIndex,
      const size_t tokenBatchSize = 1,
      const size_t leftPadLength = 0,
      const size_t rightPadLength = 0) const;

  size_t getRotEmbedSizeBytes(const size_t tokenBatchSize = 1) const;

  // The rotary embedding length is and determines the largest token size the
  // model can handle
  size_t getRotEmbedLength() const;

 private:
  std::unique_ptr<char[]> mMasterLut; // byte flatten array
  bool mIsReady = false;

  const LLMType kType;
  const size_t kTypeSize; // in bytes
  const size_t kLength;
  const size_t kHeadDim;
  const float kRotBase = 10000.0;
  const float kNtkScale;
};

} // namespace llm_helper
} // namespace example
