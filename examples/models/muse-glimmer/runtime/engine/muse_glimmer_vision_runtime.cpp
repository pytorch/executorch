/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/muse-glimmer/runtime/engine/muse_glimmer_vision_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/portable_type/device.h>
#include <executorch/runtime/platform/log.h>

#ifdef EXECUTORCH_BUILD_CUDA
#include <cuda_runtime.h>
#endif

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <executorch/examples/models/muse-glimmer/vision/preprocess.h>

namespace executorch::extension::llm {
namespace {

using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;
namespace muse_glimmer_vision = ::executorch::examples::muse_glimmer_vision;

constexpr size_t kPngSignatureSize = 8;
constexpr uint8_t kPngSignature[kPngSignatureSize] =
    {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};

bool has_jpeg_signature(Span<const uint8_t> bytes) {
  return bytes.size() >= 3 && bytes[0] == 0xff && bytes[1] == 0xd8 &&
      bytes[2] == 0xff;
}

bool has_png_signature(Span<const uint8_t> bytes) {
  return bytes.size() >= kPngSignatureSize &&
      std::equal(
             kPngSignature, kPngSignature + kPngSignatureSize, bytes.data());
}

bool validate_image_dimensions(
    int width,
    int height,
    const MuseGlimmerVisionRuntimeConfig& config) {
  if (width <= 0 || height <= 0) {
    ET_LOG(Error, "Muse Glimmer image dimensions must be positive");
    return false;
  }
  if (width > config.max_image_dimension ||
      height > config.max_image_dimension) {
    ET_LOG(
        Error,
        "Muse Glimmer image dimensions %dx%d exceed the limit of %d",
        width,
        height,
        config.max_image_dimension);
    return false;
  }
  if (config.max_image_pixels <= 0 ||
      static_cast<int64_t>(width) > config.max_image_pixels / height) {
    ET_LOG(
        Error,
        "Muse Glimmer image dimensions %dx%d exceed the pixel limit",
        width,
        height);
    return false;
  }
  return true;
}

Error copy_tensor_to_host(
    const executorch::aten::Tensor& tensor,
    void* destination,
    size_t num_bytes) {
#ifdef EXECUTORCH_BUILD_CUDA
  cudaPointerAttributes attributes{};
  const bool on_device =
      cudaPointerGetAttributes(&attributes, tensor.const_data_ptr()) ==
          cudaSuccess &&
      attributes.type == cudaMemoryTypeDevice;
  if (on_device) {
    return cudaMemcpy(
               destination,
               tensor.const_data_ptr(),
               num_bytes,
               cudaMemcpyDeviceToHost) == cudaSuccess
        ? Error::Ok
        : Error::Internal;
  }
#endif
  std::memcpy(destination, tensor.const_data_ptr(), num_bytes);
  return Error::Ok;
}

int decode_base64_character(char character) {
  if (character >= 'A' && character <= 'Z') {
    return character - 'A';
  }
  if (character >= 'a' && character <= 'z') {
    return character - 'a' + 26;
  }
  if (character >= '0' && character <= '9') {
    return character - '0' + 52;
  }
  if (character == '+') {
    return 62;
  }
  if (character == '/') {
    return 63;
  }
  return -1;
}

Result<std::vector<uint8_t>> read_file_signature_with_limit(
    const std::string& path,
    size_t max_bytes) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    ET_LOG(Error, "Cannot open Muse Glimmer image: %s", path.c_str());
    return Error::AccessFailed;
  }
  const std::streamoff file_size = file.tellg();
  if (file_size <= 0) {
    ET_LOG(
        Error,
        "Muse Glimmer image file is empty or unreadable: %s",
        path.c_str());
    return Error::InvalidExternalData;
  }
  if (static_cast<uintmax_t>(file_size) > max_bytes) {
    ET_LOG(
        Error,
        "Muse Glimmer image file exceeds the encoded byte limit: %s",
        path.c_str());
    return Error::InvalidArgument;
  }
  std::vector<uint8_t> signature(kPngSignatureSize);
  file.seekg(0, std::ios::beg);
  file.read(
      reinterpret_cast<char*>(signature.data()),
      static_cast<std::streamsize>(signature.size()));
  signature.resize(static_cast<size_t>(file.gcount()));
  return signature;
}

} // namespace

MuseGlimmerVisionRuntime::MuseGlimmerVisionRuntime(
    MuseGlimmerVisionRuntimeConfig config)
    : config_(std::move(config)) {
  if (config_.module == nullptr || config_.execution_mutex == nullptr) {
    throw std::invalid_argument(
        "MuseGlimmerVisionRuntime requires a module and execution mutex");
  }
  if (config_.pos_embed_path.empty()) {
    throw std::invalid_argument(
        "MuseGlimmerVisionRuntime requires an explicit pos_embed_path");
  }
  if (config_.activation_dtype != executorch::aten::ScalarType::BFloat16 &&
      config_.activation_dtype != executorch::aten::ScalarType::Half) {
    throw std::invalid_argument(
        "MuseGlimmerVisionRuntime activation dtype must be bfloat16 or float16");
  }
  if (config_.expected_hidden_dim <= 0 || config_.max_image_tokens <= 0 ||
      config_.max_encoded_bytes == 0 || config_.max_image_dimension <= 0 ||
      config_.max_image_pixels <= 0) {
    throw std::invalid_argument(
        "MuseGlimmerVisionRuntime limits must be positive");
  }
  pos_embed_table_ =
      muse_glimmer_vision::load_pos_embed_table(config_.pos_embed_path);
}

Result<PreparedMuseGlimmerImage>
MuseGlimmerVisionRuntime::prepare_image_from_file(
    const std::string& image_path) const {
  auto signature =
      read_file_signature_with_limit(image_path, config_.max_encoded_bytes);
  if (!signature.ok()) {
    return signature.error();
  }
  const Span<const uint8_t> signature_span(
      signature->data(), signature->size());
  if (!has_jpeg_signature(signature_span) &&
      !has_png_signature(signature_span)) {
    ET_LOG(Error, "Muse Glimmer image must have a JPEG or PNG signature");
    return Error::InvalidExternalData;
  }

  int width = 0;
  int height = 0;
  int channels = 0;
  if (stbi_info(image_path.c_str(), &width, &height, &channels) == 0) {
    ET_LOG(
        Error,
        "Muse Glimmer image metadata decode failed: %s",
        stbi_failure_reason());
    return Error::InvalidExternalData;
  }
  if (!validate_image_dimensions(width, height, config_)) {
    return Error::InvalidArgument;
  }

  using StbiImage = std::unique_ptr<uint8_t, decltype(&stbi_image_free)>;
  StbiImage rgb(
      stbi_load(image_path.c_str(), &width, &height, &channels, 3),
      &stbi_image_free);
  if (rgb == nullptr) {
    ET_LOG(
        Error, "Muse Glimmer image decode failed: %s", stbi_failure_reason());
    return Error::InvalidExternalData;
  }
  if (!validate_image_dimensions(width, height, config_)) {
    return Error::InvalidExternalData;
  }
  return prepare_decoded_image(rgb.get(), width, height);
}

Result<PreparedMuseGlimmerImage>
MuseGlimmerVisionRuntime::prepare_image_from_bytes(
    Span<const uint8_t> encoded_image) const {
  if (encoded_image.empty()) {
    ET_LOG(Error, "Muse Glimmer encoded image is empty");
    return Error::InvalidArgument;
  }
  if (encoded_image.size() > config_.max_encoded_bytes ||
      encoded_image.size() >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    ET_LOG(Error, "Muse Glimmer encoded image exceeds the byte limit");
    return Error::InvalidArgument;
  }
  if (!has_jpeg_signature(encoded_image) && !has_png_signature(encoded_image)) {
    ET_LOG(Error, "Muse Glimmer image must have a JPEG or PNG signature");
    return Error::InvalidExternalData;
  }

  int width = 0;
  int height = 0;
  int channels = 0;
  const int encoded_size = static_cast<int>(encoded_image.size());
  if (stbi_info_from_memory(
          encoded_image.data(), encoded_size, &width, &height, &channels) ==
      0) {
    ET_LOG(
        Error,
        "Muse Glimmer image metadata decode failed: %s",
        stbi_failure_reason());
    return Error::InvalidExternalData;
  }
  if (!validate_image_dimensions(width, height, config_)) {
    return Error::InvalidArgument;
  }

  using StbiImage = std::unique_ptr<uint8_t, decltype(&stbi_image_free)>;
  StbiImage rgb(
      stbi_load_from_memory(
          encoded_image.data(), encoded_size, &width, &height, &channels, 3),
      &stbi_image_free);
  if (rgb == nullptr) {
    ET_LOG(
        Error, "Muse Glimmer image decode failed: %s", stbi_failure_reason());
    return Error::InvalidExternalData;
  }
  if (!validate_image_dimensions(width, height, config_)) {
    return Error::InvalidExternalData;
  }
  return prepare_decoded_image(rgb.get(), width, height);
}

Result<PreparedMuseGlimmerImage>
MuseGlimmerVisionRuntime::prepare_decoded_image(
    const uint8_t* rgb,
    int32_t width,
    int32_t height) const {
  muse_glimmer_vision::VisionInputs inputs;
  try {
    inputs = muse_glimmer_vision::preprocess_image(
        rgb, width, height, pos_embed_table_, config_.max_image_tokens);
  } catch (const std::exception& exception) {
    ET_LOG(
        Error,
        "Muse Glimmer vision preprocessing failed: %s",
        exception.what());
    return Error::InvalidExternalData;
  }

  std::vector<EValue> encoder_inputs;
  encoder_inputs.reserve(9);
#ifdef EXECUTORCH_BUILD_CUDA
  std::vector<::executorch::extension::TensorPtr> device_inputs;
  device_inputs.reserve(9);
  const executorch::aten::Device cuda_device(
      executorch::aten::DeviceType::CUDA, 0);
  const auto stage_input = [&](const auto& input) {
    device_inputs.push_back(
        ::executorch::extension::clone_tensor_ptr_to(input, cuda_device));
    encoder_inputs.emplace_back(device_inputs.back());
  };
  stage_input(inputs.patches);
  stage_input(inputs.pos_emb);
  stage_input(inputs.cos_2d);
  stage_input(inputs.sin_2d);
  stage_input(inputs.sparse_perm);
  stage_input(inputs.inv_perm);
  stage_input(inputs.global_mask);
  stage_input(inputs.sparse_mask);
  stage_input(inputs.pixel_perm);
#else
  encoder_inputs.emplace_back(inputs.patches);
  encoder_inputs.emplace_back(inputs.pos_emb);
  encoder_inputs.emplace_back(inputs.cos_2d);
  encoder_inputs.emplace_back(inputs.sin_2d);
  encoder_inputs.emplace_back(inputs.sparse_perm);
  encoder_inputs.emplace_back(inputs.inv_perm);
  encoder_inputs.emplace_back(inputs.global_mask);
  encoder_inputs.emplace_back(inputs.sparse_mask);
  encoder_inputs.emplace_back(inputs.pixel_perm);
#endif

  std::lock_guard<std::mutex> guard(*config_.execution_mutex);
  const auto encoder_start = std::chrono::steady_clock::now();
  auto outputs = config_.module->execute("vision_encoder", encoder_inputs);
  if (!outputs.ok()) {
    ET_LOG(Error, "Muse Glimmer vision_encoder execution failed");
    return outputs.error();
  }
  const double vision_encoder_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - encoder_start)
          .count();
  if (outputs->size() != 1 || !(*outputs)[0].isTensor()) {
    ET_LOG(Error, "Muse Glimmer vision_encoder must return exactly one tensor");
    return Error::InvalidProgram;
  }

  const auto& embeddings = (*outputs)[0].toTensor();
  if (embeddings.dim() != 3 || embeddings.size(0) != 1 ||
      embeddings.size(1) != inputs.num_soft_tokens ||
      embeddings.size(2) != config_.expected_hidden_dim) {
    ET_LOG(
        Error,
        "Muse Glimmer vision_encoder output must be [1, %lld, %lld]",
        static_cast<long long>(inputs.num_soft_tokens),
        static_cast<long long>(config_.expected_hidden_dim));
    return Error::InvalidProgram;
  }
  if (embeddings.scalar_type() != config_.activation_dtype ||
      (embeddings.scalar_type() != executorch::aten::ScalarType::BFloat16 &&
       embeddings.scalar_type() != executorch::aten::ScalarType::Half)) {
    ET_LOG(
        Error,
        "Muse Glimmer vision_encoder returned an unexpected activation dtype");
    return Error::InvalidProgram;
  }

  const int64_t num_values =
      inputs.num_soft_tokens * config_.expected_hidden_dim;
  if (num_values <= 0 ||
      static_cast<uint64_t>(num_values) >
          std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
    ET_LOG(Error, "Muse Glimmer vision_encoder output size is invalid");
    return Error::InvalidProgram;
  }
  PreparedMuseGlimmerImage prepared;
  prepared.num_soft_tokens = inputs.num_soft_tokens;
  prepared.hidden_dim = config_.expected_hidden_dim;
  prepared.vision_encoder_ms = vision_encoder_ms;
  prepared.embeddings.resize(static_cast<size_t>(num_values));
  ET_CHECK_OK_OR_RETURN_ERROR(copy_tensor_to_host(
      embeddings,
      prepared.embeddings.data(),
      prepared.embeddings.size() * sizeof(uint16_t)));
  return prepared;
}

Result<std::vector<uint8_t>> decode_muse_glimmer_base64_strict(
    std::string_view encoded,
    size_t max_decoded_bytes) {
  if (encoded.empty()) {
    return std::vector<uint8_t>{};
  }
  if (encoded.size() % 4 != 0) {
    ET_LOG(Error, "Muse Glimmer base64 length must be a multiple of four");
    return Error::InvalidArgument;
  }

  size_t padding = 0;
  if (encoded.back() == '=') {
    padding = 1;
    if (encoded.size() >= 2 && encoded[encoded.size() - 2] == '=') {
      padding = 2;
    }
  }
  const size_t quartet_count = encoded.size() / 4;
  if (quartet_count > (std::numeric_limits<size_t>::max() - 2) / 3) {
    return Error::OutOfResources;
  }
  const size_t decoded_size = quartet_count * 3 - padding;
  if (decoded_size > max_decoded_bytes) {
    ET_LOG(Error, "Muse Glimmer base64 payload exceeds the decoded byte limit");
    return Error::InvalidArgument;
  }

  std::vector<uint8_t> decoded;
  decoded.reserve(decoded_size);
  for (size_t offset = 0; offset < encoded.size(); offset += 4) {
    const bool final_quartet = offset + 4 == encoded.size();
    const char c0 = encoded[offset];
    const char c1 = encoded[offset + 1];
    const char c2 = encoded[offset + 2];
    const char c3 = encoded[offset + 3];
    const int v0 = decode_base64_character(c0);
    const int v1 = decode_base64_character(c1);
    const int v2 = c2 == '=' ? 0 : decode_base64_character(c2);
    const int v3 = c3 == '=' ? 0 : decode_base64_character(c3);
    if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0 ||
        (!final_quartet && (c2 == '=' || c3 == '=')) ||
        (c2 == '=' && c3 != '=') || (c2 == '=' && (v1 & 0x0f) != 0) ||
        (c3 == '=' && c2 != '=' && (v2 & 0x03) != 0)) {
      ET_LOG(
          Error, "Muse Glimmer base64 payload has invalid alphabet or padding");
      return Error::InvalidArgument;
    }
    decoded.push_back(static_cast<uint8_t>((v0 << 2) | (v1 >> 4)));
    if (c2 != '=') {
      decoded.push_back(static_cast<uint8_t>((v1 << 4) | (v2 >> 2)));
    }
    if (c3 != '=') {
      decoded.push_back(static_cast<uint8_t>((v2 << 6) | v3));
    }
  }
  if (decoded.size() != decoded_size) {
    return Error::Internal;
  }
  return decoded;
}

} // namespace executorch::extension::llm
