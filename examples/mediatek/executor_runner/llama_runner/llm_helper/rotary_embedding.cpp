/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include "llm_helper/include/rotary_embedding.h"
#include "llm_helper/include/llm_types.h"

#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

#include <cmath>
#include <fstream>
#include <type_traits>

namespace example {
namespace llm_helper {

RotaryEmbeddingMasterLut::RotaryEmbeddingMasterLut(
    const LLMType rotEmbType,
    const size_t length,
    const size_t headDim,
    const float rotBase,
    const float ntkScale)
    : kType(rotEmbType),
      kTypeSize(getLLMTypeSize(kType)),
      kLength(length),
      kHeadDim(headDim),
      kRotBase(rotBase),
      kNtkScale(ntkScale) {
  // Shape: (length, 2*headDim), where 2 is sin & cos
  mMasterLut = std::make_unique<char[]>(kLength * 2 * kHeadDim * kTypeSize);
}

void RotaryEmbeddingMasterLut::load(
    const std::string& sinMasterPath,
    const std::string& cosMasterPath) {
  if (sinMasterPath.size() == 0 && cosMasterPath.size() == 0) {
    generate();
    return;
  }

  ET_LOG(
      Debug,
      "Begin loading rotary embedding lookup table from provided paths.");

  std::ifstream fileCos(cosMasterPath, std::ios::binary);
  std::ifstream fileSin(sinMasterPath, std::ios::binary);

  // File paths checking
  if (!fileCos) {
    ET_LOG(
        Info,
        "Warn: Rotary embedding lookup table file not found: %s. "
        "Will generate rotary embedding lookup table instead.",
        cosMasterPath.c_str());
    generate();
    return;
  }
  if (!fileSin) {
    ET_LOG(
        Info,
        "Warn: Rotary embedding lookup table file not found: %s. "
        "Will generate rotary embedding lookup table instead.",
        sinMasterPath.c_str());
    generate();
    return;
  }

  const auto rows = kLength;
  const auto rowSize = 2 * kHeadDim * kTypeSize; // x2 for sin & cos
  const size_t cosOffset = 0;
  const size_t sinOffset =
      rowSize / 2; // Halfway in row because each row is [<cos><sin>]
  const auto readSize = kHeadDim * kTypeSize;

  // Read lookup table files
  for (size_t i = 0; i < rows; ++i) {
    // Read cos then sin
    fileCos.read(mMasterLut.get() + i * rowSize + cosOffset, readSize);
    fileSin.read(mMasterLut.get() + i * rowSize + sinOffset, readSize);
  }
  mIsReady = true;
}

// For float and __fp16
template <typename RotEmbType>
void RotaryEmbeddingMasterLut::generate() {
  static_assert(
      std::is_same<RotEmbType, float>() || std::is_same<RotEmbType, __fp16>(),
      "Only int16/fp16/fp32 are supported for RotEmbType");
  ET_LOG(Debug, "Generating floating rotary embedding lookup table");

  const auto rowSize = kHeadDim * 2; // x2 for sin & cos
  const size_t rotDim = kHeadDim;
  const size_t rotDimHalf = rotDim / 2;

  const float rotDimFp = static_cast<float>(kHeadDim);
  const float base = (kNtkScale == 1.0f)
      ? kRotBase
      : std::powf(kRotBase * kNtkScale, rotDimFp / (rotDimFp - 2.0f));

  for (int pos = 0; pos < kLength; pos++) { // row in lut
    for (int dim = 0; dim < rotDimHalf; dim++) {
      const float freq =
          float(pos) / std::powf(base, float(dim * 2) / rotDimFp);
      const RotEmbType embCos = static_cast<RotEmbType>(std::cos(freq));
      const RotEmbType embSin = static_cast<RotEmbType>(std::sin(freq));

      const auto& row = pos;
      const auto& col = dim; // At most kHeadDim / 2
      auto masterLutCurPtr =
          reinterpret_cast<RotEmbType*>(mMasterLut.get()) + row * rowSize + col;

      // Concat Cos then Sin, and duplicate each
      // Each row looks like this:
      //   [<--cos--><--cos--><--sin--><--sin-->]
      //    |        |        |        |
      //    0    rotDimHalf   |        |
      //                    rotDim     |
      //                        rotDim + rotDimHalf
      masterLutCurPtr[0] = embCos;
      masterLutCurPtr[rotDimHalf] = embCos;
      masterLutCurPtr[rotDim] = embSin;
      masterLutCurPtr[rotDim + rotDimHalf] = embSin;
    }
  }
  mIsReady = true;
}

// NOTE: The difference between this and the Python script generated rotary
// embedding master lut is the rounding mechanism during quantization to INT16.
// Python's Numpy library uses round-to-even (banker's rounding) whereas the
// below C++ code uses round-to-nearest.
template <>
void RotaryEmbeddingMasterLut::generate<int16_t>() {
  ET_LOG(Debug, "Generating int16 rotary embedding lookup table");

  const auto rowSize = kHeadDim * 2; // x2 for sin & cos
  const size_t rotDim = kHeadDim;
  const size_t rotDimHalf = rotDim / 2;

  const float rotDimFp = static_cast<float>(kHeadDim);
  const float base = (kNtkScale == 1.0f)
      ? kRotBase
      : std::powf(kRotBase * kNtkScale, rotDimFp / (rotDimFp - 2.0f));

  // Minmax=(-1,1), so qscale = 1/32767
  const float qscale = 0.000030518509447574615;

  auto quantFP32ToINT16 = [&](const float fpval) -> int16_t {
    const int qmin = -32768; // -2^(outBitwidth-1)
    const int qmax = +32767; // 2^(outBitwidth-1)-1
    const int quantized = std::round(fpval / qscale);
    const int clamped = std::max(qmin, std::min(quantized, qmax));
    return clamped;
  };

  for (int pos = 0; pos < kLength; pos++) { // row in lut
    for (int dim = 0; dim < rotDimHalf; dim++) {
      const float freq =
          float(pos) / std::powf(base, float(dim * 2) / rotDimFp);
      const int16_t embCos = quantFP32ToINT16(std::cos(freq));
      const int16_t embSin = quantFP32ToINT16(std::sin(freq));

      const auto& row = pos;
      const auto& col = dim; // At most kHeadDim / 2
      auto masterLutCurPtr =
          reinterpret_cast<int16_t*>(mMasterLut.get()) + row * rowSize + col;

      // Concat Cos then Sin, and duplicate each
      // Each row looks like this:
      //   [<--cos--><--cos--><--sin--><--sin-->]
      //    |        |        |        |
      //    0    rotDimHalf   |        |
      //                    rotDim     |
      //                        rotDim + rotDimHalf
      masterLutCurPtr[0] = embCos;
      masterLutCurPtr[rotDimHalf] = embCos;
      masterLutCurPtr[rotDim] = embSin;
      masterLutCurPtr[rotDim + rotDimHalf] = embSin;
    }
  }
  mIsReady = true;
}

void RotaryEmbeddingMasterLut::generate() {
  switch (kType) {
    case LLMType::INT16:
      generate<int16_t>();
      return;
    case LLMType::FP16:
      generate<__fp16>();
      return;
    case LLMType::FP32:
      generate<float>();
      return;
    default:
      break;
  }
  ET_LOG(
      Fatal,
      "Rotary embedding generator not implemented for %s",
      getLLMTypeName(kType));
}

// RotaryEmbeddingMasterLut supports 1 or 2 rotary embedding inputs
void RotaryEmbeddingMasterLut::setEmbed(
    std::vector<void*> rotEmbedBuffers,
    const size_t tokenIndex,
    const size_t tokenBatchSize,
    const size_t leftPadLength,
    const size_t rightPadLength) const {
  const auto numRotEmbInputs = rotEmbedBuffers.size();
  switch (numRotEmbInputs) {
    case 1: {
      const auto rotEmbInput = rotEmbedBuffers[0];
      setEmbed(
          rotEmbInput,
          tokenIndex,
          tokenBatchSize,
          leftPadLength,
          rightPadLength);
      break;
    }
    case 2: {
      const auto rotEmbCosInput = rotEmbedBuffers[0];
      const auto rotEmbSinInput = rotEmbedBuffers[1];
      setEmbed(
          rotEmbCosInput,
          rotEmbSinInput,
          tokenIndex,
          tokenBatchSize,
          leftPadLength,
          rightPadLength);
      break;
    }
    default:
      ET_LOG(
          Fatal,
          "RotaryEmbeddingMasterLut: Unsupported number of rotary embedding inputs (%zu).",
          numRotEmbInputs);
  }
}

void RotaryEmbeddingMasterLut::setEmbed(
    void* rotEmbedBuffer,
    const size_t tokenIndex,
    const size_t tokenBatchSize,
    const size_t leftPadLength,
    const size_t rightPadLength) const {
  // Generate Master Lut if not yet done
  if (!mIsReady) {
    ET_LOG(
        Error,
        "Attempting to use the rotary embedding lookup table before being initialized.");
    return;
  }
  const auto requestedMaxIndex = tokenIndex + tokenBatchSize - 1;
  const auto availableLength = getRotEmbedLength();
  if (requestedMaxIndex >= availableLength) {
    ET_LOG(
        Fatal,
        "Requested rotary embeddings (%zu) exceeds the max available (%zu) "
        "in the master lookup table. Please ensure that your maxTokenLength option "
        "is set correctly",
        requestedMaxIndex,
        availableLength);
  }
  // The model takes in the rot emb as [2, tokenBatchSize, kHeadDim],
  // but the master lut stores in [tokenIdx, 2, kHeadDim].
  const auto rowSizeBytes = 2 * kHeadDim * kTypeSize; // cos and sin
  const auto rowSizeBytesHalf = rowSizeBytes / 2; // one of cos or sin only
  const auto cosOffset = 0;
  const auto sinOffset = rowSizeBytesHalf;
  const auto copySize = rowSizeBytesHalf;

  auto curRotEmbedBuffer = reinterpret_cast<char*>(rotEmbedBuffer);
  const auto masterLutStart = mMasterLut.get() + tokenIndex * rowSizeBytes;

  ET_DCHECK(tokenBatchSize >= leftPadLength + rightPadLength);
  const size_t numValidInputToken =
      tokenBatchSize - leftPadLength - rightPadLength;

  const auto leftPadSize = copySize * leftPadLength;
  const auto rightPadSize = copySize * rightPadLength;

  // Skip left-padding
  curRotEmbedBuffer += leftPadSize;

  // cos
  for (size_t i = 0; i < numValidInputToken; i++) {
    std::memcpy(
        curRotEmbedBuffer,
        masterLutStart + i * rowSizeBytes + cosOffset,
        copySize);
    curRotEmbedBuffer += copySize;
  }

  // Right pad for 'cos', and left pad for 'sin'.
  std::memset(curRotEmbedBuffer, 0, rightPadSize);
  curRotEmbedBuffer += leftPadSize + rightPadSize;

  // sin
  for (size_t i = 0; i < numValidInputToken; i++) {
    std::memcpy(
        curRotEmbedBuffer,
        masterLutStart + i * rowSizeBytes + sinOffset,
        copySize);
    curRotEmbedBuffer += copySize;
  }

  // Right pad for 'sin'
  std::memset(curRotEmbedBuffer, 0, rightPadSize);
}

void RotaryEmbeddingMasterLut::setEmbed(
    void* rotEmbedCosBuffer,
    void* rotEmbedSinBuffer,
    const size_t tokenIndex,
    const size_t tokenBatchSize,
    const size_t leftPadLength,
    const size_t rightPadLength) const {
  // Generate Master Lut if not yet done
  if (!mIsReady) {
    ET_LOG(
        Error,
        "Attempting to use the rotary embedding lookup table before being initialized.");
    return;
  }
  const auto requestedMaxIndex = tokenIndex + tokenBatchSize - 1;
  const auto availableLength = getRotEmbedLength();
  if (requestedMaxIndex >= availableLength) {
    ET_LOG(
        Fatal,
        "Requested rotary embeddings (%zu) exceeds the max available (%zu) "
        "in the master lookup table. Please ensure that your maxTokenLength option "
        "is set correctly",
        requestedMaxIndex,
        availableLength);
  }
  // The model takes in the rot emb as [2, tokenBatchSize, kHeadDim],
  // but the master lut stores in [tokenIdx, 2, kHeadDim].
  const auto rowSizeBytes = 2 * kHeadDim * kTypeSize; // cos and sin
  const auto rowSizeBytesHalf = rowSizeBytes / 2; // one of cos or sin only
  const auto cosOffset = 0;
  const auto sinOffset = rowSizeBytesHalf;
  const auto copySize = rowSizeBytesHalf;

  const auto masterLutStart = mMasterLut.get() + tokenIndex * rowSizeBytes;

  auto curRotEmbedCosBuffer = reinterpret_cast<char*>(rotEmbedCosBuffer);
  auto curRotEmbedSinBuffer = reinterpret_cast<char*>(rotEmbedSinBuffer);

  ET_DCHECK(tokenBatchSize >= leftPadLength + rightPadLength);
  const size_t numValidInputToken =
      tokenBatchSize - leftPadLength - rightPadLength;

  const auto leftPadSize = copySize * leftPadLength;
  const auto rightPadSize = copySize * rightPadLength;

  // Skip left-padding
  curRotEmbedCosBuffer += leftPadSize;
  curRotEmbedSinBuffer += leftPadSize;

  for (size_t i = 0; i < numValidInputToken; i++) {
    std::memcpy(
        curRotEmbedCosBuffer,
        masterLutStart + i * rowSizeBytes + cosOffset,
        copySize);
    std::memcpy(
        curRotEmbedSinBuffer,
        masterLutStart + i * rowSizeBytes + sinOffset,
        copySize);
    curRotEmbedCosBuffer += copySize;
    curRotEmbedSinBuffer += copySize;
  }
  std::memset(curRotEmbedCosBuffer, 0, rightPadSize);
  std::memset(curRotEmbedSinBuffer, 0, rightPadSize);
}

size_t RotaryEmbeddingMasterLut::getRotEmbedSizeBytes(
    const size_t tokenBatchSize) const {
  return 2 * tokenBatchSize * kHeadDim * kTypeSize;
}

// The rotary embedding length is and determines the largest token size the
// model can handle
size_t RotaryEmbeddingMasterLut::getRotEmbedLength() const {
  return kLength;
}

} // namespace llm_helper
} // namespace example
