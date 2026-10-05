/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/batching/test/fake_executor.h>
#include <executorch/extension/llm/serving/serving_runtime.h>

#include <numeric>

namespace executorch::examples::llm_server::testing {

// A synthetic image encoder, not an image codec or a model adapter. Its private
// rows exercise preparation, slicing and token feedback through the real
// worker.
class ImageExecutor : public extension::llm::batching::testing::FakeExecutor {
  using Token = extension::llm::batching::Token;
  using PreparedInput = extension::llm::batching::PreparedInput;
  using PreparedInputPtr = extension::llm::batching::PreparedInputPtr;
  using TokenPreparedInput = extension::llm::batching::TokenPreparedInput;
  using PreparationInput = extension::llm::batching::PreparationInput;
  using PreparationConfig = extension::llm::batching::PreparationConfig;
  using BatchInput = extension::llm::batching::BatchInput;
  using BatchOutput = extension::llm::batching::BatchOutput;

  struct Row {
    Token encoded;
    bool image;
  };

  struct Rows final : PreparedInput {
    std::vector<Row> rows;
    std::size_t position_count() const override {
      return rows.size();
    }
    std::size_t retained_bytes() const override {
      return sizeof(*this) + rows.capacity() * sizeof(Row);
    }
    static const void* tag() {
      static const char identity = 0;
      return &identity;
    }
    const void* compatibility_tag() const override {
      return tag();
    }
  };

 public:
  explicit ImageExecutor(bool images = false)
      : FakeExecutor(configuration(images)), images_(images) {}

  bool accepts(const PreparedInput& input) const override {
    return Executor::accepts(input) ||
        (images_ && input.compatibility_tag() == Rows::tag());
  }

  bool prepare(const PreparationInput& input, PreparedInputPtr& out) override {
    if (!images_) {
      return Executor::prepare(input, out);
    }
    out.reset();
    std::size_t count = 0;
    std::size_t images = 0;
    for (const auto& segment : input.segments) {
      const auto positions = segment.is_tokens() ? segment.get_tokens().size()
          : segment.is_image()                   ? 4
                                                 : 0;
      if (positions > preparation_config().max_positions - count ||
          (!segment.is_tokens() && !segment.is_image()) ||
          (segment.is_image() && ++images > preparation_config().max_images)) {
        return false;
      }
      count += positions;
    }
    if (!bounded(count)) {
      return false;
    }
    auto prepared = std::make_shared<Rows>();
    prepared->rows.reserve(count);
    for (const auto& segment : input.segments) {
      if (segment.is_tokens()) {
        for (Token token : segment.get_tokens()) {
          prepared->rows.push_back({token ^ mask_, false});
        }
      } else {
        const auto& image = segment.get_image();
        if (!image.is_uint8() || image.get_uint8_data().empty()) {
          return false;
        }
        const auto& bytes = image.get_uint8_data();
        const auto hash = std::accumulate(
            bytes.begin(), bytes.end(), Token{0}, [](Token value, auto byte) {
              return value * 131 + byte;
            });
        for (Token row = 0; row < 4; ++row) {
          prepared->rows.push_back({(hash + row) ^ mask_, true});
        }
      }
    }
    out = std::move(prepared);
    return true;
  }

  bool wrap_tokens(
      std::shared_ptr<const std::vector<Token>> tokens,
      PreparedInputPtr& out) override {
    if (!images_) {
      return Executor::wrap_tokens(std::move(tokens), out);
    }
    out.reset();
    if (!tokens || !bounded(tokens->size())) {
      return false;
    }
    auto prepared = std::make_shared<Rows>();
    prepared->rows.reserve(tokens->size());
    for (Token token : *tokens) {
      prepared->rows.push_back({token ^ mask_, false});
    }
    out = std::move(prepared);
    return true;
  }

  bool execute(const BatchInput& batch, BatchOutput& out) override {
    if (!validate_batch(batch)) {
      return false;
    }
    BatchInput decoded;
    for (const auto& input : batch.inputs) {
      auto copy = input;
      if (input.prepared->compatibility_tag() == Rows::tag()) {
        auto tokens = std::make_shared<std::vector<Token>>();
        tokens->reserve(input.prepared->position_count());
        for (std::size_t i = 0; i < input.prepared->position_count(); ++i) {
          tokens->push_back(input_token(*input.prepared, i));
        }
        copy.prepared = std::make_shared<TokenPreparedInput>(std::move(tokens));
      }
      decoded.inputs.push_back(std::move(copy));
    }
    return FakeExecutor::execute(decoded, out);
  }

  static void enable_images(
      extension::llm::serving::ServingRuntimeConfig& config) {
    config.max_images = 1;
    config.max_image_preprocessed_bytes = 1;
    // Synthetic bounded CPU preparation hashes the validated encoded source.
    // Real integrations supply a codec/preprocessor with its own allocation
    // bounds.
    config.image_preprocessor = [bound = config.max_image_preprocessed_bytes](
                                    const extension::llm::EncodedImage& image)
        -> runtime::Result<extension::llm::Image> {
      if (image.data.empty() || bound < 1) {
        return runtime::Error::InvalidArgument;
      }
      const auto hash = std::accumulate(
          image.data.begin(),
          image.data.end(),
          std::uint8_t{0},
          [](std::uint8_t value, auto byte) {
            return static_cast<std::uint8_t>(value * 31 + byte);
          });
      return extension::llm::Image(std::vector<std::uint8_t>{hash}, 1, 1, 1);
    };
  }

 protected:
  Token input_token(const PreparedInput& input, std::size_t offset) const {
    if (input.compatibility_tag() == Rows::tag()) {
      return static_cast<const Rows&>(input).rows.at(offset).encoded ^ mask_;
    }
    return static_cast<const TokenPreparedInput&>(input).tokens().at(offset);
  }

 private:
  static PreparationConfig configuration(bool images) {
    PreparationConfig config;
    config.max_positions = 4096;
    config.max_retained_bytes = 1024 * 1024;
    config.max_workspace_bytes = 1024 * 1024;
    config.max_total_retained_bytes = 16 * 1024 * 1024;
    config.max_images = images ? 1 : 0;
    return config;
  }

  bool bounded(std::size_t count) const {
    const auto& config = preparation_config();
    return count > 0 && count <= config.max_positions &&
        config.max_retained_bytes >= sizeof(Rows) &&
        count <= (config.max_retained_bytes - sizeof(Rows)) / sizeof(Row) &&
        count <= config.max_workspace_bytes / sizeof(Row);
  }

  static constexpr Token mask_ = 0xd35ac97e45128b60ULL;
  const bool images_;
};

} // namespace executorch::examples::llm_server::testing
