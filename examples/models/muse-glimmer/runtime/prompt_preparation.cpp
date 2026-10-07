/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/prompt_preparation.h>

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
} // namespace

MuseGlimmerPreparedInput::MuseGlimmerPreparedInput(
    std::shared_ptr<const MuseGlimmerPreparationSpec> spec,
    std::vector<batching::Token> tokens,
    MuseGlimmerRGBImage image,
    MuseGlimmerImageGrid grid,
    MuseGlimmerImageSpan image_span)
    : spec_(std::move(spec)),
      tokens_(std::move(tokens)),
      image_(std::move(image)),
      grid_(grid),
      image_span_(image_span),
      valid_(spec_ && validate_structure()) {}

const void* MuseGlimmerPreparedInput::kind_tag() {
  return &prepared_kind;
}
const void* MuseGlimmerPreparedInput::kind() const {
  return kind_tag();
}

bool MuseGlimmerPreparedInput::compatible(
    const MuseGlimmerPreparationSpec& spec) const {
  return spec_.get() == &spec && valid_;
}

bool MuseGlimmerPreparedInput::validate_structure() const {
  const auto& spec = *spec_;
  const auto& image = *image_;
  if (!spec.has_vision || spec.hidden_dim <= 0 ||
      (spec.activation_dtype != aten::ScalarType::Half &&
       spec.activation_dtype != aten::ScalarType::BFloat16) ||
      tokens_.empty() ||
      tokens_.size() >= static_cast<size_t>(spec.max_context_length) ||
      image_span_.offset > tokens_.size() || image_span_.size == 0 ||
      image_span_.size > tokens_.size() - image_span_.offset ||
      grid_.soft_tokens != static_cast<int64_t>(image_span_.size) ||
      grid_.soft_tokens > spec.max_soft_tokens || image.width <= 0 ||
      image.height <= 0 ||
      image.width > spec.image_limits.max_image_dimension ||
      image.height > spec.image_limits.max_image_dimension ||
      static_cast<int64_t>(image.width) * image.height >
          spec.image_limits.max_image_pixels ||
      image.rgb.size() != static_cast<size_t>(image.width) * image.height * 3) {
    return false;
  }
  auto grid =
      muse_glimmer_image_grid(image.width, image.height, spec.max_soft_tokens);
  if (!grid.ok() || grid->height != grid_.height ||
      grid->width != grid_.width || grid->soft_tokens != grid_.soft_tokens ||
      !valid_tokens(tokens_, spec)) {
    return false;
  }
  for (size_t i = 0; i < tokens_.size(); ++i) {
    const bool in_image =
        i >= image_span_.offset && i - image_span_.offset < image_span_.size;
    if ((tokens_[i] == spec.patch_id) != in_image) {
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
  std::vector<batching::Token> tokens;
  MuseGlimmerRGBImage image;
  MuseGlimmerImageGrid grid;
  MuseGlimmerImageSpan span{0, 0};
  const char* failure = "invalid prompt segment";
  const auto append_text = [&](const std::string& text) {
    auto encoded = context.tokenizer.encode(text, 0, 0);
    if (!encoded.ok() || encoded->size() > limit - tokens.size() ||
        cancelled()) {
      failure =
          "prompt tokenization failed, exceeded context, or was cancelled";
      return false;
    }
    tokens.insert(tokens.end(), encoded->begin(), encoded->end());
    return true;
  };
  const auto append_ids = [&](const nlohmann::json& ids) {
    if (!ids.is_array() || ids.size() > limit - tokens.size()) {
      failure = "prompt IDs must be a bounded array";
      return false;
    }
    for (const auto& id : ids) {
      if (!id.is_number_integer() ||
          (!id.is_number_unsigned() && id.get<int64_t>() < 0)) {
        failure = "prompt IDs must be nonnegative integers";
        return false;
      }
      tokens.push_back(id.get<uint64_t>());
    }
    return true;
  };
  const auto append_image = [&](const nlohmann::json& encoded) {
    if (span.size || !spec->has_vision) {
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
        static_cast<size_t>(computed->soft_tokens) > limit - tokens.size()) {
      failure = "image grid exceeds vision patch or prompt limit";
      return false;
    }
    image = std::move(*decoded);
    grid = *computed;
    span = {tokens.size(), static_cast<size_t>(grid.soft_tokens)};
    tokens.insert(tokens.end(), span.size, spec->patch_id);
    return true;
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
    if (std::find(tokens.begin(), tokens.end(), spec->patch_id) !=
        tokens.end()) {
      return invalid("patch tokens require an image");
    }
    serving::PromptInput prompt;
    prompt.segments.emplace_back(std::move(tokens));
    return serving::GenerationPrompt{std::move(prompt)};
  }
  auto backing = std::make_shared<MuseGlimmerPreparedInput>(
      spec, std::move(tokens), std::move(image), grid, span);
  if (!backing->compatible(*spec))
    return invalid("invalid image token layout");
  const auto previous = backing->previous_token();
  return serving::GenerationPrompt{
      serving::PreparedPromptInput{std::move(backing), previous}};
}

} // namespace executorch::extension::llm
