/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include "llm_helper/include/token_embedding.h"
#include "FileMemMapper.h"
#include "llm_helper/include/llm_types.h"

#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace example {
namespace llm_helper {

TokenEmbeddingLut::TokenEmbeddingLut(
    const std::string& tokenEmbLutPath,
    const LLMType tokenEmbLutType,
    const size_t hiddenSize)
    : kTokenEmbLutType(tokenEmbLutType),
      kTokenEmbLutTypeSize(getLLMTypeSize(tokenEmbLutType)),
      kHiddenSize(hiddenSize),
      kLutRowSizeBytes(kHiddenSize * kTokenEmbLutTypeSize) {
  ET_CHECK_MSG(
      fs::exists(tokenEmbLutPath),
      "Token embedding lookup table file not found: %s",
      tokenEmbLutPath.c_str());

  ET_LOG(
      Debug,
      "Loading token embedding lookup table: %s",
      tokenEmbLutPath.c_str());

  mMemMappedEmbFile = std::make_unique<FileMemMapper>(tokenEmbLutPath);
  mLutBuffer = mMemMappedEmbFile->getAddr<uint8_t*>();
  const size_t lutFileSize = mMemMappedEmbFile->getSize();

  mVocabSize = lutFileSize / hiddenSize / kTokenEmbLutTypeSize;
  ET_LOG(Debug, "TokenEmbeddingLut: Vocab size = %zu", mVocabSize);
}

TokenEmbeddingLut::~TokenEmbeddingLut() {}

void TokenEmbeddingLut::setOutput(void* buffer, const size_t size) {
  mOutputBuffer = reinterpret_cast<uint8_t*>(buffer);
  mOutputBufferSize = size;
}

void TokenEmbeddingLut::lookupEmbedding(const std::vector<uint64_t>& tokens) {
  const auto numTokens = tokens.size();
  const size_t requiredOutputSize =
      numTokens * kHiddenSize * kTokenEmbLutTypeSize;
  if (mOutputBufferSize < requiredOutputSize) {
    ET_LOG(
        Error,
        "Token embedding buffer size (%zu) is insufficient to hold embedding for %zu tokens "
        "(requires %zu).",
        mOutputBufferSize,
        numTokens,
        requiredOutputSize);
    return;
  }
  if (mOutputBuffer == nullptr) {
    ET_LOG(
        Error,
        "TokenEmbeddingLut: Output is not yet set for embedding lookup.");
    return;
  }
  size_t outputOffset = 0;
  for (const auto token : tokens) {
    // Copy one row from lookup table per token
    ET_CHECK_MSG(
        token < mVocabSize, "Token id exceeds embedding lookup table range.");
    const auto& rowIdx = token;
    const size_t lutOffset = rowIdx * kLutRowSizeBytes;
    std::memcpy(
        mOutputBuffer + outputOffset, mLutBuffer + lutOffset, kLutRowSizeBytes);
    outputOffset += kLutRowSizeBytes;
  }
  return;
}

} // namespace llm_helper
} // namespace example
