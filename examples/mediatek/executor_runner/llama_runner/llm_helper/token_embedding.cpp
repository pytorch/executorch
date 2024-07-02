/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include "llm_types.h"
#include "llm_helper/include/token_embedding.h"

#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/assert.h>

#include <string>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace torch::executor {
namespace llm_helper {

TokenEmbeddingLut::TokenEmbeddingLut(const std::string& tokenEmbLutPath,
                                     const LLMType tokenEmbLutType, const size_t hiddenSize)
    : kTokenEmbLutType(tokenEmbLutType),
      kTokenEmbLutTypeSize(getLLMTypeSize(tokenEmbLutType)),
      kHiddenSize(hiddenSize),
      kLutRowSizeBytes(kHiddenSize * kTokenEmbLutTypeSize) {
    std::ifstream file(tokenEmbLutPath, std::ios::binary);
    if (!file) {
        ET_LOG(Fatal, "Token embedding lookup table file not found: %s", tokenEmbLutPath.c_str());
    }
    ET_LOG(Debug, "Loading token embedding lookup table: %s", tokenEmbLutPath.c_str());

    const size_t lutFileSize = fs::file_size(tokenEmbLutPath);

    mVocabSize = lutFileSize / hiddenSize / kTokenEmbLutTypeSize;
    ET_LOG(Debug, "TokenEmbeddingLut: Vocab size = %zu", mVocabSize);

    mLutBuffer = new uint8_t [lutFileSize];

    file.read(reinterpret_cast<char*>(mLutBuffer), lutFileSize);
    ET_CHECK(file.gcount() == lutFileSize);
}

TokenEmbeddingLut::~TokenEmbeddingLut() {
    if (mLutBuffer != nullptr) {
        delete mLutBuffer;
    }
}

void TokenEmbeddingLut::setOutput(void* buffer, const size_t size) {
    setOutput(buffer, size, kTokenEmbLutType);
}

void TokenEmbeddingLut::setOutput(void* buffer, const size_t size, const LLMType type,
                                  const float qscale) {
    mOutputBuffer = reinterpret_cast<uint8_t*>(buffer);
    mOutputBufferSize = size;
    mTokenEmbOutputType = type;
    mTokenEmbOutputTypeSize = getLLMTypeSize(type);
    mTokenEmbQuantScale = qscale;
}

void TokenEmbeddingLut::lookupEmbedding(const std::vector<uint64_t>& tokens) {
    const auto numTokens = tokens.size();
    const size_t requiredOutputSize = numTokens * kHiddenSize * mTokenEmbOutputTypeSize;
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
        ET_LOG(Error, "TokenEmbeddingLut: Output is not yet set for embedding lookup.");
        return;
    }

    // Same type, so we can simply memcpy.
    if (kTokenEmbLutType == mTokenEmbOutputType) {
        size_t outputOffset = 0;
        for (const auto token : tokens) {
            // Copy one row from lookup table per token
            ET_CHECK_MSG(token < mVocabSize, "Token id exceeds embedding lookup table range.");
            const auto& rowIdx = token;
            const size_t lutOffset = rowIdx * kLutRowSizeBytes;
            std::memcpy(mOutputBuffer + outputOffset, mLutBuffer + lutOffset, kLutRowSizeBytes);
            outputOffset += kLutRowSizeBytes;
        }
        return;
    }

    ET_LOG(
        Fatal,
        "Unimplemented: Mismatch between token embedding lookup table type (%s) "
        "and model input embedding type (%s)",
        getLLMTypeName(kTokenEmbLutType),
        getLLMTypeName(mTokenEmbOutputType));
}

} // namespace llm_helper
} // namespace torch::executor