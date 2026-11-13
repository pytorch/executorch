/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/diffusers/runner/metadata.h>

#include <executorch/runtime/platform/log.h>

#include <cmath>

namespace executorch::extension::diffusers {

using executorch::aten::ScalarType;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;

namespace {

Result<EValue> read(Module& module, const char* key) {
  auto value = module.get(std::string("get_diffusion_") + key);
  if (!value.ok()) {
    ET_LOG(Error, "Missing diffusion metadata get_diffusion_%s", key);
  }
  return value;
}

Result<std::string> read_string(Module& module, const char* key) {
  const auto value = ET_UNWRAP(read(module, key));
  return std::string(
      ET_UNWRAP(value.tryToString(), "get_diffusion_%s is not a string", key));
}

Result<int64_t> read_int(Module& module, const char* key) {
  const auto value = ET_UNWRAP(read(module, key));
  return ET_UNWRAP(value.tryToInt(), "get_diffusion_%s is not an int", key);
}

Result<bool> read_bool(Module& module, const char* key) {
  return ET_UNWRAP(read_int(module, key)) != 0;
}

Result<double> read_double(Module& module, const char* key) {
  const auto value = ET_UNWRAP(read(module, key));
  if (value.isInt()) {
    return static_cast<double>(value.toInt());
  }
  return ET_UNWRAP(
      value.tryToDouble(), "get_diffusion_%s is not a number", key);
}

Result<ScalarType> read_dtype(Module& module, const char* key) {
  const auto name = ET_UNWRAP(read_string(module, key));
  if (name == "float32") {
    return ScalarType::Float;
  }
  if (name == "float16") {
    return ScalarType::Half;
  }
  if (name == "bfloat16") {
    return ScalarType::BFloat16;
  }
  ET_LOG(Error, "Unsupported get_diffusion_%s %s", key, name.c_str());
  return Error::NotSupported;
}

Result<std::vector<int64_t>> read_lengths(Module& module, const char* key) {
  const auto value = ET_UNWRAP(read(module, key));
  const auto tensor =
      ET_UNWRAP(value.tryToTensor(), "get_diffusion_%s is not a tensor", key);
  ET_CHECK_OR_RETURN_ERROR(
      tensor.dim() == 1 && tensor.scalar_type() == ScalarType::Int &&
          tensor.numel() > 0,
      InvalidProgram,
      "get_diffusion_%s must be a non-empty rank-one int32 tensor",
      key);
  const auto* data = tensor.const_data_ptr<int32_t>();
  std::vector<int64_t> lengths(data, data + tensor.numel());
  for (int64_t length : lengths) {
    ET_CHECK_OR_RETURN_ERROR(
        length > 0,
        InvalidProgram,
        "get_diffusion_%s must contain positive values",
        key);
  }
  return lengths;
}

} // namespace

Result<DiffusionMetadata> read_diffusion_metadata(Module& module) {
  DiffusionMetadata metadata;
  metadata.pipeline = ET_UNWRAP(read_string(module, "pipeline"));
  metadata.scheduler = ET_UNWRAP(read_string(module, "scheduler"));
  metadata.scheduler_prediction_type =
      ET_UNWRAP(read_string(module, "scheduler_prediction_type"));
  metadata.supports_cfg = ET_UNWRAP(read_bool(module, "supports_cfg"));
  metadata.accepts_guidance_input =
      ET_UNWRAP(read_bool(module, "accepts_guidance_input"));
  metadata.supports_negative_prompt =
      ET_UNWRAP(read_bool(module, "supports_negative_prompt"));
  metadata.default_cfg_scale =
      ET_UNWRAP(read_double(module, "default_cfg_scale"));
  // The export encodes an absent default guidance scale as NaN.
  const double guidance_scale =
      ET_UNWRAP(read_double(module, "default_guidance_scale"));
  if (!std::isnan(guidance_scale)) {
    metadata.default_guidance_scale = guidance_scale;
  }
  metadata.guidance_input_dim =
      ET_UNWRAP(read_int(module, "guidance_input_dim"));
  metadata.default_inference_steps =
      ET_UNWRAP(read_int(module, "default_inference_steps"));
  metadata.text_encoder_dtype =
      ET_UNWRAP(read_dtype(module, "text_encoder_dtype"));
  metadata.denoiser_dtype = ET_UNWRAP(read_dtype(module, "denoiser_dtype"));
  metadata.vae_dtype = ET_UNWRAP(read_dtype(module, "vae_dtype"));
  metadata.zero_negative_prompt =
      ET_UNWRAP(read_bool(module, "zero_negative_prompt"));
  metadata.max_sequence_lengths =
      ET_UNWRAP(read_lengths(module, "max_sequence_lengths"));
  metadata.height = ET_UNWRAP(read_int(module, "height"));
  metadata.width = ET_UNWRAP(read_int(module, "width"));
  metadata.dynamic_shapes = ET_UNWRAP(read_bool(module, "dynamic_shapes"));
  metadata.min_size = ET_UNWRAP(read_int(module, "min_size"));
  metadata.max_size = ET_UNWRAP(read_int(module, "max_size"));
  metadata.dynamic_spatial_multiple =
      ET_UNWRAP(read_int(module, "dynamic_spatial_multiple"));
  metadata.vae_scale_factor = ET_UNWRAP(read_int(module, "vae_scale_factor"));
  return metadata;
}

} // namespace executorch::extension::diffusers
