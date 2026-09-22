/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace supertonic {

struct TextBatch {
  std::vector<int64_t> ids;
  std::vector<float> mask;
  std::vector<int64_t> shape;
};

void validate_language(const std::string& language);
std::string preprocess_text(
    const std::string& text,
    const std::string& language);
std::vector<std::string> chunk_text(const std::string& text, size_t max_length);
std::vector<std::string> chunk_text_for_language(
    const std::string& text,
    const std::string& language);

class UnicodeProcessor {
 public:
  explicit UnicodeProcessor(const std::string& indexer_path);

  void configure_vocabulary(int64_t vocabulary_size);
  TextBatch process(
      const std::vector<std::string>& texts,
      const std::vector<std::string>& languages) const;

 private:
  std::vector<int64_t> indexer_;
  int64_t vocabulary_size_ = 0;
};

} // namespace supertonic
