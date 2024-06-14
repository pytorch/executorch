/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <sstream>

#include <sentencepiece_processor.h>

// TODO(lunwenh): Add unit tests
class SentencePieceTokenizer {
 public:
  SentencePieceTokenizer(const std::string& filePath) {
    const auto status = processor_.Load(filePath);
    if (!status.ok()) {
      std::ostringstream errorMessageStream;
      errorMessageStream << "Failed to load SentencePiece model from "
                         << filePath << " with error " << status.ToString();
      throw std::runtime_error(errorMessageStream.str());
    }
    processor_.SetEncodeExtraOptions("bos");
  }

  std::vector<int64_t> encode(const std::string& piece) {
    std::vector<int> ids;
    processor_.Encode(piece, &ids);
    std::vector<int64_t> idsLong(ids.begin(), ids.end());
    return idsLong;
  }

  std::string decode(const std::vector<int64_t>& ids) {
    std::vector<int> idsInt(ids.begin(), ids.end());
    std::string piece;
    processor_.Decode(idsInt, &piece);
    return piece;
  }

 private:
  sentencepiece::SentencePieceProcessor processor_;
};
