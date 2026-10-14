/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/embedding_materializer.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <cstring>

namespace executorch::extension::llm {
using runtime::Error;

bool MuseGlimmerMaterializer::accepts(
    const batching::PreparedInput& input) const {
  return spec_ && input.kind() == MuseGlimmerPreparedInput::kind_tag() &&
      static_cast<const MuseGlimmerPreparedInput&>(input).compatible(*spec_);
}

runtime::Result<TensorPtr> MuseGlimmerMaterializer::materialize(
    const std::vector<batching::Input>& inputs,
    const EncodeImage& encode_image,
    const EmbedText& embed_text) const {
  if (!spec_ || spec_->hidden_dim <= 0 || spec_->max_forward_tokens <= 0 ||
      (spec_->activation_dtype != aten::ScalarType::Half &&
       spec_->activation_dtype != aten::ScalarType::BFloat16)) {
    return Error::InvalidArgument;
  }
  size_t rows = 0;
  for (const auto& input : inputs) {
    size_t total = 0;
    if (const auto* raw =
            std::get_if<batching::TokenInputPtr>(&input.payload)) {
      if (!*raw)
        return Error::InvalidArgument;
      total = (*raw)->size();
    } else {
      const auto& prepared =
          std::get<batching::PreparedInputPtr>(input.payload);
      if (!prepared || !accepts(*prepared))
        return Error::InvalidArgument;
      total = prepared->size();
    }
    if (input.offset > total || input.size == 0 ||
        input.size > total - input.offset ||
        input.size > static_cast<size_t>(spec_->max_forward_tokens) - rows) {
      return Error::InvalidArgument;
    }
    if (const auto* raw =
            std::get_if<batching::TokenInputPtr>(&input.payload)) {
      for (size_t i = input.offset; i < input.offset + input.size; ++i) {
        if ((**raw)[i] >= static_cast<uint64_t>(spec_->vocab_size)) {
          return Error::InvalidArgument;
        }
      }
    }
    rows += input.size;
  }
  if (rows == 0)
    return Error::InvalidArgument;
  auto output = zeros(
      {1, static_cast<aten::SizesType>(rows), spec_->hidden_dim},
      spec_->activation_dtype);
  auto* destination = static_cast<uint16_t*>(output->mutable_data_ptr());
  const size_t hidden = spec_->hidden_dim;
  std::vector<int64_t> text_tokens;
  std::vector<size_t> text_rows;
  size_t row = 0;
  for (const auto& input : inputs) {
    const auto* raw = std::get_if<batching::TokenInputPtr>(&input.payload);
    const auto* prepared = raw
        ? nullptr
        : static_cast<const MuseGlimmerPreparedInput*>(
              std::get<batching::PreparedInputPtr>(input.payload).get());
    const auto* backing = prepared ? prepared->backing_.get() : nullptr;
    for (size_t offset = input.offset; offset < input.offset + input.size;
         ++offset, ++row) {
      const size_t index = prepared ? prepared->start_ + offset : offset;
      const bool image_row = backing && index >= backing->image_span_.offset &&
          index - backing->image_span_.offset < backing->image_span_.size;
      if (image_row) {
        if (backing->image_embeddings_.empty()) {
          if (!backing->image_)
            return Error::InvalidProgram;
          ET_ASSIGN_OR_RETURN(image, encode_image(*backing->image_));
          if (image.hidden_dim != spec_->hidden_dim ||
              image.num_soft_tokens != backing->grid_.soft_tokens ||
              image.embeddings.size() != backing->image_span_.size * hidden) {
            return Error::InvalidProgram;
          }
          backing->image_embeddings_ = std::move(image.embeddings);
          backing->image_.reset();
        }
        const size_t source_row = index - backing->image_span_.offset;
        std::memcpy(
            destination + row * hidden,
            backing->image_embeddings_.data() + source_row * hidden,
            hidden * sizeof(uint16_t));
      } else {
        const auto token = raw ? (**raw)[index] : (*backing->tokens_)[index];
        if (token >= static_cast<uint64_t>(spec_->vocab_size))
          return Error::InvalidArgument;
        text_tokens.push_back(static_cast<int64_t>(token));
        text_rows.push_back(row);
      }
    }
  }
  // All vision outputs are copied to host before embed_text can reuse arenas.
  if (!text_tokens.empty()) {
    ET_ASSIGN_OR_RETURN(text, embed_text(text_tokens));
    if (!text || text->dim() != 3 || text->size(0) != 1 ||
        text->size(1) != static_cast<int64_t>(text_tokens.size()) ||
        text->size(2) != spec_->hidden_dim ||
        text->scalar_type() != spec_->activation_dtype ||
        !text->device().is_cpu() || text->strides()[2] != 1 ||
        text->strides()[1] != spec_->hidden_dim ||
        text->const_data_ptr() == nullptr) {
      return Error::InvalidProgram;
    }
    const auto* source = static_cast<const uint16_t*>(text->const_data_ptr());
    for (size_t i = 0; i < text_rows.size(); ++i) {
      std::memcpy(
          destination + text_rows[i] * hidden,
          source + i * hidden,
          hidden * sizeof(uint16_t));
    }
  }
  return output;
}

} // namespace executorch::extension::llm
