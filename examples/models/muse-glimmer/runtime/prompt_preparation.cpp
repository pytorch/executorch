/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/prompt_preparation.h>

#include <CommonCrypto/CommonDigest.h>
#include <executorch/extension/llm/serving/detail/prompt_preparer.h>
#include <nlohmann/json.hpp>
#include <pytorch/tokenizers/tokenizer.h>
#include <algorithm>
#include <limits>
#include <utility>

namespace executorch::extension::llm {
namespace {
char prepared_kind;

serving::ServingError invalid(const char* message) {
  return {serving::ErrorCode::InvalidArgument, message};
}

bool valid_tokens(
    const std::vector<batching::Token>& tokens,
    const MuseGlimmerPreparationSpec& spec) {
  return std::all_of(tokens.begin(), tokens.end(), [&](auto token) {
    return token < static_cast<uint64_t>(spec.vocab_size);
  });
}

bool valid_image(
    int32_t width,
    int32_t height,
    size_t bytes,
    const MuseGlimmerPreparationSpec& spec) {
  const int64_t pixels = static_cast<int64_t>(width) * height;
  return width > 0 && height > 0 &&
      width <= spec.image_limits.max_image_dimension &&
      height <= spec.image_limits.max_image_dimension &&
      pixels <= spec.image_limits.max_image_pixels &&
      static_cast<uint64_t>(pixels) <= std::numeric_limits<size_t>::max() / 3 &&
      bytes == static_cast<size_t>(pixels) * 3;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
batching::ContentKey image_key(
    const MuseGlimmerRGBImage& image,
    const MuseGlimmerImageGrid& grid) {
  CC_SHA256_CTX hash;
  CC_SHA256_Init(&hash);
  constexpr char version[] = "muse-glimmer-image-v1";
  CC_SHA256_Update(&hash, version, sizeof(version) - 1);
  // Five unsigned 64-bit big-endian fields, followed by canonical HWC bytes.
  for (uint64_t value :
       {uint64_t(image.width),
        uint64_t(image.height),
        uint64_t(grid.width),
        uint64_t(grid.height),
        uint64_t(grid.soft_tokens)}) {
    uint8_t bytes[8];
    for (size_t i = 0; i < 8; ++i)
      bytes[7 - i] = static_cast<uint8_t>(value >> (i * 8));
    CC_SHA256_Update(&hash, bytes, sizeof(bytes));
  }
  for (size_t offset = 0; offset < image.rgb.size();) {
    const auto count = static_cast<CC_LONG>(std::min(
        image.rgb.size() - offset,
        static_cast<size_t>(std::numeric_limits<CC_LONG>::max())));
    CC_SHA256_Update(&hash, image.rgb.data() + offset, count);
    offset += count;
  }
  batching::ContentKey key;
  CC_SHA256_Final(key.data(), &hash);
  return key;
}
#pragma clang diagnostic pop
} // namespace

MuseGlimmerPreparedInput::MuseGlimmerPreparedInput(
    std::shared_ptr<const MuseGlimmerPreparationSpec> spec,
    std::vector<batching::Token> tokens,
    MuseGlimmerRGBImage image,
    MuseGlimmerImageGrid grid,
    MuseGlimmerImageSpan image_span)
    : backing_(std::make_shared<Backing>(
          std::move(spec),
          std::move(tokens),
          std::move(image),
          grid,
          image_span)) {}

MuseGlimmerPreparedInput::Backing::Backing(
    std::shared_ptr<const MuseGlimmerPreparationSpec> spec,
    std::vector<batching::Token> tokens,
    MuseGlimmerRGBImage image,
    MuseGlimmerImageGrid grid,
    MuseGlimmerImageSpan image_span)
    : spec_(std::move(spec)),
      tokens_(std::make_shared<const std::vector<batching::Token>>(
          std::move(tokens))),
      image_(std::move(image)),
      grid_(grid),
      image_span_(image_span),
      valid_(spec_ && validate_structure()),
      image_key_(valid_ ? image_key(*image_, grid_) : batching::ContentKey{}) {}

const void* MuseGlimmerPreparedInput::kind_tag() {
  return &prepared_kind;
}
const void* MuseGlimmerPreparedInput::kind() const {
  return kind_tag();
}

bool MuseGlimmerPreparedInput::compatible(
    const MuseGlimmerPreparationSpec& spec) const {
  return backing_->spec_.get() == &spec && backing_->valid_;
}

batching::PreparedInputPtr MuseGlimmerPreparedInput::suffix(
    size_t start) const {
  return backing_->valid_ && start < size()
      ? batching::PreparedInputPtr(
            new MuseGlimmerPreparedInput(backing_, start_ + start))
      : nullptr;
}

batching::PrefixIdentityPtr MuseGlimmerPreparedInput::prefix_identity() const {
  if (!backing_->valid_)
    return nullptr;
  auto identity = std::make_shared<batching::PrefixIdentity>();
  const auto& b = *backing_;
  const auto append = [&](auto span, size_t begin) {
    const size_t skip =
        start_ > begin ? std::min(start_ - begin, span.size) : 0;
    span.offset += skip;
    span.size -= skip;
    if (span.size)
      identity->spans.emplace_back(std::move(span));
  };
  const size_t end = b.image_span_.offset + b.image_span_.size;
  append(batching::TokenSpan{b.tokens_, 0, b.image_span_.offset}, 0);
  append(
      batching::OpaqueSpan{b.image_key_, 0, b.image_span_.size},
      b.image_span_.offset);
  append(batching::TokenSpan{b.tokens_, end, b.tokens_->size() - end}, end);
  return identity;
}

bool MuseGlimmerPreparedInput::Backing::validate_structure() const {
  const auto& spec = *spec_;
  const auto& image = *image_;
  const auto& tokens = *tokens_;
  if (!spec.has_vision || spec.hidden_dim <= 0 || spec.vocab_size <= 0 ||
      spec.max_context_length <= 1 ||
      (spec.activation_dtype != aten::ScalarType::Half &&
       spec.activation_dtype != aten::ScalarType::BFloat16) ||
      tokens.empty() ||
      tokens.size() >= static_cast<size_t>(spec.max_context_length) ||
      image_span_.offset > tokens.size() || image_span_.size == 0 ||
      image_span_.size > tokens.size() - image_span_.offset ||
      grid_.soft_tokens != static_cast<int64_t>(image_span_.size) ||
      grid_.soft_tokens > spec.max_soft_tokens ||
      !valid_image(image.width, image.height, image.rgb.size(), spec)) {
    return false;
  }
  auto grid =
      muse_glimmer_image_grid(image.width, image.height, spec.max_soft_tokens);
  if (!grid.ok() || grid->height != grid_.height ||
      grid->width != grid_.width || grid->soft_tokens != grid_.soft_tokens ||
      !valid_tokens(tokens, spec)) {
    return false;
  }
  for (size_t i = 0; i < tokens.size(); ++i) {
    const bool in_image =
        i >= image_span_.offset && i - image_span_.offset < image_span_.size;
    if ((tokens[i] == spec.patch_id) != in_image) {
      return false;
    }
  }
  return true;
}

serving::PromptPreparationResult prepare_muse_glimmer_prompt(
    const nlohmann::json& request,
    const serving::PromptPreparationContext& context,
    std::shared_ptr<const MuseGlimmerPreparationSpec> spec) {
  if (!spec || spec->max_context_length <= 1 || !request.is_object() ||
      request.contains("prompt") == request.contains("prompt_segments")) {
    return invalid(
        "supply exactly one of prompt or prompt_segments with a valid specification");
  }
  const auto cancelled = [&] {
    return context.cancelled && context.cancelled();
  };
  const size_t limit = std::min(
      context.max_prompt_positions,
      static_cast<size_t>(spec->max_context_length - 1));
  if (cancelled())
    return invalid("prompt preparation cancelled");
  serving::PromptInput prompt;
  size_t known_positions = 0;
  bool has_image = false;
  const char* failure = "invalid prompt segment";
  const auto append_text = [&](const std::string& text) {
    prompt.segments.emplace_back(text);
    return !cancelled();
  };
  const auto append_ids = [&](const nlohmann::json& ids) {
    if (!ids.is_array() || ids.size() > limit - known_positions) {
      failure = "prompt IDs must be a bounded array";
      return false;
    }
    std::vector<batching::Token> tokens;
    for (const auto& id : ids) {
      if (!id.is_number_integer() ||
          (!id.is_number_unsigned() && id.get<int64_t>() < 0)) {
        failure = "prompt IDs must be nonnegative integers";
        return false;
      }
      tokens.push_back(id.get<uint64_t>());
    }
    known_positions += tokens.size();
    prompt.segments.emplace_back(std::move(tokens));
    return !cancelled();
  };
  const auto append_image = [&](const nlohmann::json& encoded) {
    if (has_image || !spec->has_vision) {
      failure = "only one image is supported, and requires a vision artifact";
      return false;
    }
    if (!encoded.is_object() || encoded.size() != 3 ||
        !encoded.contains("encoding") || encoded["encoding"] != "base64" ||
        !encoded.contains("mime_type") ||
        (encoded["mime_type"] != "image/png" &&
         encoded["mime_type"] != "image/jpeg") ||
        !encoded.contains("data") || !encoded["data"].is_string()) {
      failure =
          "image must contain base64 encoding, PNG/JPEG mime_type, and data";
      return false;
    }
    auto bytes = decode_muse_glimmer_base64_strict(
        encoded["data"].get_ref<const std::string&>(),
        spec->image_limits.max_encoded_bytes);
    if (!bytes.ok() || bytes->empty() || cancelled()) {
      failure = "invalid, oversized, or cancelled image data";
      return false;
    }
    const bool png =
        bytes->size() >= 8 && (*bytes)[0] == 0x89 && (*bytes)[1] == 'P';
    if ((encoded["mime_type"] == "image/png") != png) {
      failure = "image signature does not match mime_type";
      return false;
    }
    auto decoded = decode_muse_glimmer_image(
        runtime::Span<const uint8_t>(bytes->data(), bytes->size()),
        spec->image_limits);
    if (!decoded.ok() || cancelled()) {
      failure = "invalid, oversized, or cancelled decoded image";
      return false;
    }
    auto computed = muse_glimmer_image_grid(
        decoded->width, decoded->height, spec->max_soft_tokens);
    if (!computed.ok() ||
        static_cast<size_t>(computed->soft_tokens) > limit - known_positions) {
      failure = "image grid exceeds vision patch or prompt limit";
      return false;
    }
    // The shared Image contract is CHW; the decoder returns interleaved HWC.
    const size_t pixels = decoded->rgb.size() / 3;
    std::vector<uint8_t> chw(decoded->rgb.size());
    for (size_t p = 0; p < pixels; ++p)
      for (size_t c = 0; c < 3; ++c)
        chw[c * pixels + p] = decoded->rgb[p * 3 + c];
    prompt.segments.emplace_back(
        Image(std::move(chw), decoded->width, decoded->height, 3));
    known_positions += static_cast<size_t>(computed->soft_tokens);
    has_image = true;
    return !cancelled();
  };

  if (request.contains("prompt")) {
    if (!request["prompt"].is_string())
      return invalid("prompt must be text");
    const auto& text = request["prompt"].get_ref<const std::string&>();
    // Direct solo input retains the legacy one-marker convenience format.
    if (request.contains("image") && !request["image"].is_null()) {
      const size_t marker = text.find("<img>");
      if (marker == std::string::npos ||
          text.find("<img>", marker + 5) != std::string::npos) {
        return invalid("one image requires exactly one <img> marker");
      }
      if (!append_text(text.substr(0, marker)) ||
          !append_image(request["image"]) ||
          !append_text(text.substr(marker + 5)))
        return invalid(failure);
    } else if (!append_text(text)) {
      return invalid(failure);
    }
  } else {
    if (request.contains("image"))
      return invalid("image must be an ordered prompt segment");
    const auto& segments = request["prompt_segments"];
    if (!segments.is_array() || segments.empty())
      return invalid("prompt_segments must be a nonempty array");
    for (const auto& segment : segments) {
      if (!segment.is_object() || segment.size() != 1)
        return invalid(
            "each segment needs exactly one text, ids, or image field");
      bool ok = false;
      if (segment.contains("text") && segment["text"].is_string()) {
        ok = append_text(segment["text"].get_ref<const std::string&>());
      } else if (segment.contains("ids")) {
        ok = append_ids(segment["ids"]);
      } else if (segment.contains("image")) {
        ok = append_image(segment["image"]);
      }
      if (!ok || cancelled())
        return invalid(failure);
    }
  }
  if (cancelled())
    return invalid("prompt preparation cancelled");
  return prompt;
}

serving::ModelPreparationResult prepare_muse_glimmer_input(
    const serving::PromptPreparationContext& context,
    serving::PromptInput input,
    std::shared_ptr<const MuseGlimmerPreparationSpec> spec) {
  if (!spec || spec->max_context_length <= 1 || spec->vocab_size <= 0 ||
      input.segments.empty())
    return invalid("invalid preparation specification or empty input");
  const auto cancelled = [&] {
    return context.cancelled && context.cancelled();
  };
  const size_t limit = std::min(
      context.max_prompt_positions,
      static_cast<size_t>(spec->max_context_length - 1));
  std::vector<batching::Token> tokens;
  MuseGlimmerRGBImage image;
  MuseGlimmerImageGrid grid;
  MuseGlimmerImageSpan span{0, 0};
  for (const auto& segment : input.segments) {
    if (cancelled())
      return invalid("prompt preparation cancelled");
    std::vector<batching::Token> encoded;
    const auto* ids = segment.try_get_tokens();
    if (const auto* text = segment.try_get_text()) {
      auto result = context.tokenizer.encode(*text, 0, 0);
      if (!result.ok())
        return invalid("prompt tokenization failed");
      encoded = std::move(*result);
      ids = &encoded;
    }
    if (ids) {
      if (ids->size() > limit - tokens.size() || !valid_tokens(*ids, *spec) ||
          std::find(ids->begin(), ids->end(), spec->patch_id) != ids->end())
        return invalid("invalid, oversized, or unbound patch token segment");
      tokens.insert(tokens.end(), ids->begin(), ids->end());
    } else if (const auto* source = segment.try_get_image()) {
      if (span.size || !spec->has_vision || !source->is_uint8() ||
          source->channels() != 3 ||
          !valid_image(
              source->width(),
              source->height(),
              source->get_uint8_data().size(),
              *spec))
        return invalid("only one bounded uint8 RGB image is supported");
      auto computed = muse_glimmer_image_grid(
          source->width(), source->height(), spec->max_soft_tokens);
      if (!computed.ok() ||
          static_cast<size_t>(computed->soft_tokens) > limit - tokens.size())
        return invalid("image grid exceeds vision patch or prompt limit");
      image.width = source->width();
      image.height = source->height();
      const auto& chw = source->get_uint8_data();
      image.rgb.resize(chw.size());
      const size_t pixels = chw.size() / 3;
      for (size_t p = 0; p < pixels; ++p)
        for (size_t c = 0; c < 3; ++c)
          image.rgb[p * 3 + c] = chw[c * pixels + p];
      grid = *computed;
      span = {tokens.size(), static_cast<size_t>(grid.soft_tokens)};
      tokens.insert(tokens.end(), span.size, spec->patch_id);
    } else {
      return invalid("unsupported prompt modality");
    }
  }
  if (tokens.empty() || tokens.front() != spec->bos_id) {
    tokens.insert(tokens.begin(), spec->bos_id);
    if (span.size)
      ++span.offset;
  }
  if (tokens.size() > limit || !valid_tokens(tokens, *spec) || cancelled()) {
    return invalid(
        "prompt exceeds context/vocabulary or preparation cancelled");
  }
  if (!span.size) {
    return batching::PreparedInputPtr(
        std::make_shared<serving::detail::TokenPreparedInput>(
            std::make_shared<const std::vector<batching::Token>>(
                std::move(tokens))));
  }
  auto backing = std::make_shared<MuseGlimmerPreparedInput>(
      spec, std::move(tokens), std::move(image), grid, span);
  if (!backing->compatible(*spec))
    return invalid("invalid image token layout");
  if (cancelled())
    return invalid("prompt preparation cancelled");
  return batching::PreparedInputPtr(std::move(backing));
}

} // namespace executorch::extension::llm
