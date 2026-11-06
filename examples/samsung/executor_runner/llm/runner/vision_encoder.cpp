/*
 * Copyright (c) 2026 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/samsung/executor_runner/llm/runner/vision_encoder.h>

#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace example {

namespace runtime = ::executorch::runtime;
namespace extension = ::executorch::extension;

VisionEncoder::VisionEncoder(const std::string& model_path)
    : model_path_(model_path) {}

runtime::Error VisionEncoder::load() {
  if (loaded_) {
    return runtime::Error::Ok;
  }
  if (model_path_.empty()) {
    ET_LOG(Error, "Vision model path is empty");
    return runtime::Error::InvalidArgument;
  }

  module_ = std::make_unique<extension::Module>(
      model_path_, extension::Module::LoadMode::File);

  auto err = module_->load_method("vision_encoder");
  if (err != runtime::Error::Ok) {
    ET_LOG(
        Error,
        "Failed to load vision_encoder method from %s",
        model_path_.c_str());
    return err;
  }

  auto method_meta = module_->method_meta("vision_encoder");
  ET_CHECK_OK_OR_RETURN_ERROR(method_meta.error());
  ET_CHECK_OR_RETURN_ERROR(
      method_meta->num_inputs() >= 2 && method_meta->num_outputs() >= 2,
      InvalidProgram,
      "vision_encoder expects 2 inputs / 2 outputs, got %zu / %zu",
      method_meta->num_inputs(),
      method_meta->num_outputs());

  auto pixel_meta = method_meta->input_tensor_meta(0);
  ET_CHECK_OK_OR_RETURN_ERROR(pixel_meta.error());
  ET_CHECK_OR_RETURN_ERROR(
      pixel_meta->scalar_type() == executorch::aten::ScalarType::Float,
      InvalidProgram,
      "vision_encoder pixel_values must be float32, got dtype %d",
      static_cast<int>(pixel_meta->scalar_type()));
  const auto pixel_sizes = pixel_meta->sizes();
  ET_CHECK_OR_RETURN_ERROR(
      pixel_sizes.size() == 3 && pixel_sizes[2] == kVisionPatchDim &&
          pixel_sizes[1] % (kVisionPoolingKernel * kVisionPoolingKernel) == 0,
      InvalidProgram,
      "Unexpected vision_encoder pixel_values shape");
  max_soft_tokens_ =
      pixel_sizes[1] / (kVisionPoolingKernel * kVisionPoolingKernel);

  auto pos_meta = method_meta->input_tensor_meta(1);
  ET_CHECK_OK_OR_RETURN_ERROR(pos_meta.error());
  position_ids_dtype_ = pos_meta->scalar_type();
  ET_CHECK_OR_RETURN_ERROR(
      position_ids_dtype_ == executorch::aten::ScalarType::Long ||
          position_ids_dtype_ == executorch::aten::ScalarType::Int,
      InvalidProgram,
      "vision_encoder pixel_position_ids must be int64/int32, got dtype %d",
      static_cast<int>(position_ids_dtype_));

  auto out_meta = method_meta->output_tensor_meta(0);
  if (out_meta.ok() && out_meta->sizes().size() >= 3) {
    hidden_size_ = out_meta->sizes()[2];
  }
  ET_LOG(
      Info,
      "Vision encoder: max_soft_tokens %d, position_ids dtype %d, hidden %lld",
      max_soft_tokens_,
      static_cast<int>(position_ids_dtype_),
      static_cast<long long>(hidden_size_));

  loaded_ = true;
  ET_LOG(Info, "Vision encoder loaded: %s", model_path_.c_str());
  return runtime::Error::Ok;
}

runtime::Error VisionEncoder::encode(const std::string& image_path) {
  if (!loaded_) {
    ET_LOG(Error, "Vision encoder is not loaded");
    return runtime::Error::InvalidState;
  }

  // Load and patchify the image.
  int32_t width = 0;
  int32_t height = 0;
  std::vector<uint8_t> rgb;
  try {
    rgb = load_image_rgb(image_path, &width, &height);
  } catch (const std::exception& e) {
    ET_LOG(Error, "Failed to load image: %s", e.what());
    return runtime::Error::InvalidArgument;
  }

  ImageData image =
      patchify_rgb_image(rgb.data(), width, height, max_soft_tokens_);

  extension::TensorPtr position_ids = image.pixel_position_ids;
  if (position_ids_dtype_ == executorch::aten::ScalarType::Int) {
    position_ids = extension::empty(
        {1, static_cast<executorch::aten::SizesType>(position_ids->size(1)), 2},
        executorch::aten::ScalarType::Int);
    std::copy_n(
        image.pixel_position_ids->const_data_ptr<int64_t>(),
        position_ids->numel(),
        position_ids->mutable_data_ptr<int32_t>());
  }

  std::vector<runtime::EValue> input_vec = {image.pixel_values, position_ids};

  auto results = module_->execute("vision_encoder", input_vec);
  ET_CHECK_OK_OR_RETURN_ERROR(results.error());

  auto& outputs = results.get();
  if (outputs.size() < 2 || !outputs[0].isTensor() || !outputs[1].isTensor()) {
    ET_LOG(Error, "Unexpected vision encoder outputs (%zu)", outputs.size());
    return runtime::Error::InvalidProgram;
  }

  const auto& embeds = outputs[0].toTensor();
  const auto& mask = outputs[1].toTensor();

  if (embeds.scalar_type() != executorch::aten::ScalarType::Float) {
    ET_LOG(Error, "Vision embeddings are not float32");
    return runtime::Error::InvalidProgram;
  }

  const float* embeds_data = embeds.const_data_ptr<float>();
  const auto pooled = embeds.size(1);
  const auto hidden = embeds.size(2);
  hidden_size_ = hidden;

  ET_CHECK_OR_RETURN_ERROR(
      mask.numel() == pooled,
      InvalidProgram,
      "Vision mask numel %lld != pooled tokens %lld",
      static_cast<long long>(mask.numel()),
      static_cast<long long>(pooled));

  // The delegate may not preserve the BOOL dtype of the exported mask.
  std::vector<bool> valid(pooled);
  switch (mask.scalar_type()) {
    case executorch::aten::ScalarType::Bool:
      std::copy_n(mask.const_data_ptr<bool>(), pooled, valid.begin());
      break;
    case executorch::aten::ScalarType::Byte:
      std::copy_n(mask.const_data_ptr<uint8_t>(), pooled, valid.begin());
      break;
    case executorch::aten::ScalarType::Int:
      std::copy_n(mask.const_data_ptr<int32_t>(), pooled, valid.begin());
      break;
    case executorch::aten::ScalarType::Long:
      std::copy_n(mask.const_data_ptr<int64_t>(), pooled, valid.begin());
      break;
    case executorch::aten::ScalarType::Float:
      std::copy_n(mask.const_data_ptr<float>(), pooled, valid.begin());
      break;
    default:
      ET_LOG(
          Error,
          "Unsupported vision mask dtype %d",
          static_cast<int>(mask.scalar_type()));
      return runtime::Error::InvalidProgram;
  }
  num_valid_tokens_ = std::count(valid.begin(), valid.end(), true);

  const int64_t expected_tokens =
      compute_vision_num_tokens(image.num_valid_patches);
  if (num_valid_tokens_ != expected_tokens) {
    ET_LOG(
        Error,
        "Vision mask reports %lld valid tokens, expected %lld for %dx%d resized image",
        static_cast<long long>(num_valid_tokens_),
        static_cast<long long>(expected_tokens),
        image.original_width,
        image.original_height);
    return runtime::Error::InvalidProgram;
  }

  // Keep only valid tokens' embeddings, packed contiguously.
  vision_embeds_.clear();
  vision_embeds_.reserve(static_cast<size_t>(num_valid_tokens_) * hidden);
  for (int64_t i = 0; i < pooled; ++i) {
    if (valid[i]) {
      vision_embeds_.insert(
          vision_embeds_.end(),
          embeds_data + i * hidden,
          embeds_data + (i + 1) * hidden);
    }
  }

  ET_LOG(
      Info,
      "Vision encoded: %lld valid tokens out of %lld",
      static_cast<long long>(num_valid_tokens_),
      static_cast<long long>(pooled));

  return runtime::Error::Ok;
}

} // namespace example
