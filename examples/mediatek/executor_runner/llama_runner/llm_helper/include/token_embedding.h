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

class FileMemMapper;

namespace llm_helper {

class TokenEmbeddingLut {
 public:
  TokenEmbeddingLut(
      const std::string& tokenEmbLutPath,
      const LLMType tokenEmbLutType,
      const size_t hiddenSize);

  ~TokenEmbeddingLut();

  void setOutput(void* buffer, const size_t size);

  void lookupEmbedding(const std::vector<uint64_t>& tokens);

 private:
  // Source lookup table
  uint8_t* mLutBuffer = nullptr;
  const LLMType kTokenEmbLutType;
  const size_t kTokenEmbLutTypeSize;
  const size_t kHiddenSize;
  const size_t kLutRowSizeBytes;
  size_t mVocabSize;

  // Output write buffer
  uint8_t* mOutputBuffer = nullptr;
  size_t mOutputBufferSize = 0;

  std::unique_ptr<FileMemMapper> mMemMappedEmbFile;
};

} // namespace llm_helper
} // namespace example
