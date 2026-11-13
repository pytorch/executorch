/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/diffusers/runner/pipelines.h>

#include <executorch/runtime/platform/assert.h>
#include <executorch/runtime/platform/log.h>

#include <nlohmann/json.hpp>

#include <algorithm>

namespace executorch::extension::diffusers {

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::runtime::Error;
using executorch::runtime::Result;
using json = nlohmann::json;

namespace {

double config_number(const json& config, const char* key, double fallback) {
  const auto it = config.find(key);
  return it == config.end() || it->is_null() ? fallback : it->get<double>();
}

TensorPtr scalar_tensor(float value, ScalarType dtype) {
  return to_tensor(HostTensor{{1}, {value}}, dtype);
}

TensorPtr image_position_ids(int64_t rows, int64_t columns) {
  std::vector<int64_t> ids;
  ids.reserve(rows * columns * 3);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < columns; ++column) {
      ids.insert(ids.end(), {0, row, column});
    }
  }
  return make_tensor_ptr(
      {static_cast<SizesType>(rows * columns), 3}, std::move(ids));
}

class StableDiffusionBehavior : public PipelineBehavior {
 public:
  HostTensor create_latents(
      const std::vector<int64_t>& exported_shape,
      const ImageGeometry& geometry,
      TorchGenerator& generator,
      ScalarType dtype,
      const Scheduler& scheduler) const override {
    HostTensor latents;
    latents.sizes = {
        static_cast<SizesType>(exported_shape[0]),
        static_cast<SizesType>(exported_shape[1]),
        static_cast<SizesType>(geometry.height / geometry.vae_scale_factor),
        static_cast<SizesType>(geometry.width / geometry.vae_scale_factor)};
    latents.data = randn(
        generator,
        dtype,
        static_cast<size_t>(
            latents.sizes[0] * latents.sizes[1] * latents.sizes[2] *
            latents.sizes[3]));
    const float sigma = scheduler.init_noise_sigma();
    std::transform(
        latents.data.begin(),
        latents.data.end(),
        latents.data.begin(),
        [sigma](float value) { return value * sigma; });
    round_to(latents.data, dtype);
    return latents;
  }

  Result<HostTensor> prepare_encoded_latents(
      const HostTensor& mean,
      const HostTensor& std,
      ScalarType dtype,
      TorchGenerator& generator) const override {
    const auto noise = randn(generator, dtype, mean.data.size());
    HostTensor latents{mean.sizes, std::vector<float>(mean.data.size())};
    for (size_t i = 0; i < latents.data.size(); ++i) {
      latents.data[i] = mean.data[i] + std.data[i] * noise[i];
    }
    round_to(latents.data, dtype);
    return latents;
  }

  Result<size_t> prepare_img2img_latents(
      Scheduler& scheduler,
      HostTensor& latents,
      int steps,
      double strength,
      TorchGenerator& generator,
      ScalarType dtype) const override {
    const int kept = std::min(static_cast<int>(steps * strength), steps);
    const auto start = static_cast<size_t>(std::max(steps - kept, 0));
    ET_CHECK_OR_RETURN_ERROR(
        start < scheduler.timesteps().size(),
        InvalidArgument,
        "strength %f leaves no denoising steps",
        strength);
    const auto noise = randn(generator, dtype, latents.data.size());
    ET_CHECK_OK_OR_RETURN_ERROR(
        scheduler.add_noise(latents.data, noise, start));
    round_to(latents.data, dtype);
    return start;
  }
};

class SD15Behavior : public StableDiffusionBehavior {
 public:
  std::vector<std::string> tokenizer_slots() const override {
    return {"tokenizer"};
  }

  std::vector<std::vector<std::string>> text_encoder_input_names()
      const override {
    return {{"input_ids"}};
  }

  std::vector<HostTensor> assemble_conditioning(
      std::vector<std::vector<HostTensor>> encoder_outputs) const override {
    return {std::move(encoder_outputs[0][0])};
  }

  std::vector<TensorPtr> prepare_denoiser_inputs(
      const HostTensor& latents,
      float timestep,
      const std::vector<TensorPtr>& conditioning,
      const ImageGeometry& geometry,
      ScalarType dtype,
      std::optional<double> guidance_scale) const override {
    std::vector<TensorPtr> inputs = {
        to_tensor(latents, dtype),
        scalar_tensor(timestep, ScalarType::Float),
        conditioning[0]};
    if (guidance_scale) {
      inputs.push_back(scalar_tensor(*guidance_scale, ScalarType::Float));
    }
    return inputs;
  }
};

class SDXLBehavior : public StableDiffusionBehavior {
 public:
  std::vector<std::string> tokenizer_slots() const override {
    return {"tokenizer", "tokenizer_2"};
  }

  std::vector<std::vector<std::string>> text_encoder_input_names()
      const override {
    return {{"input_ids"}, {"input_ids"}};
  }

  // Concatenates both hidden states on the feature dimension and keeps the
  // pooled embedding of the second encoder.
  std::vector<HostTensor> assemble_conditioning(
      std::vector<std::vector<HostTensor>> encoder_outputs) const override {
    const auto& first = encoder_outputs[0][0];
    const auto& second = encoder_outputs[1][0];
    const size_t first_width = first.sizes.back();
    const size_t second_width = second.sizes.back();
    const size_t rows = first.data.size() / first_width;
    HostTensor hidden{first.sizes, {}};
    hidden.sizes.back() = static_cast<SizesType>(first_width + second_width);
    hidden.data.reserve(rows * (first_width + second_width));
    for (size_t row = 0; row < rows; ++row) {
      const auto first_row = first.data.begin() + row * first_width;
      const auto second_row = second.data.begin() + row * second_width;
      hidden.data.insert(hidden.data.end(), first_row, first_row + first_width);
      hidden.data.insert(
          hidden.data.end(), second_row, second_row + second_width);
    }
    return {std::move(hidden), std::move(encoder_outputs[1][1])};
  }

  std::vector<TensorPtr> prepare_denoiser_inputs(
      const HostTensor& latents,
      float timestep,
      const std::vector<TensorPtr>& conditioning,
      const ImageGeometry& geometry,
      ScalarType dtype,
      std::optional<double> guidance_scale) const override {
    // time_ids: original size, crop coordinates and target size.
    const auto height = static_cast<float>(geometry.height);
    const auto width = static_cast<float>(geometry.width);
    std::vector<TensorPtr> inputs = {
        to_tensor(latents, dtype),
        scalar_tensor(timestep, ScalarType::Float),
        conditioning[0],
        conditioning[1],
        to_tensor(
            HostTensor{{1, 6}, {height, width, 0, 0, height, width}}, dtype)};
    if (guidance_scale) {
      inputs.push_back(scalar_tensor(*guidance_scale, ScalarType::Float));
    }
    return inputs;
  }
};

class FluxBehavior : public PipelineBehavior {
 public:
  std::vector<std::string> tokenizer_slots() const override {
    return {"tokenizer", "tokenizer_2"};
  }

  std::vector<std::vector<std::string>> text_encoder_input_names()
      const override {
    return {{"input_ids"}, {"input_ids"}};
  }

  // The CLIP encoder provides the pooled embedding and T5 the prompt.
  std::vector<HostTensor> assemble_conditioning(
      std::vector<std::vector<HostTensor>> encoder_outputs) const override {
    return {std::move(encoder_outputs[1][0]), std::move(encoder_outputs[0][0])};
  }

  bool should_use_cfg(double cfg_scale, bool has_negative_prompt)
      const override {
    return cfg_scale > 1 && has_negative_prompt;
  }

  Error prepare_schedule(
      Scheduler& scheduler,
      const json& scheduler_config,
      int steps,
      const ImageGeometry& geometry) const override {
    const double sequence_length =
        static_cast<double>(geometry.height / geometry.vae_scale_factor / 2) *
        static_cast<double>(geometry.width / geometry.vae_scale_factor / 2);
    const double base_length =
        config_number(scheduler_config, "base_image_seq_len", 256);
    const double max_length =
        config_number(scheduler_config, "max_image_seq_len", 4096);
    const double base_shift =
        config_number(scheduler_config, "base_shift", 0.5);
    const double max_shift = config_number(scheduler_config, "max_shift", 1.15);
    const double mu = base_shift +
        (sequence_length - base_length) * (max_shift - base_shift) /
            (max_length - base_length);
    const auto use_flow_sigmas = scheduler_config.find("use_flow_sigmas");
    const bool flow_sigmas = use_flow_sigmas != scheduler_config.end() &&
        use_flow_sigmas->is_boolean() && use_flow_sigmas->get<bool>();
    const auto sigmas = flow_sigmas
        ? std::vector<double>()
        : torch_linspace<double>(1.0, 1.0 / steps, steps);
    return scheduler.set_timesteps(steps, sigmas, mu);
  }

  HostTensor create_latents(
      const std::vector<int64_t>& exported_shape,
      const ImageGeometry& geometry,
      TorchGenerator& generator,
      ScalarType dtype,
      const Scheduler& scheduler) const override {
    HostTensor noise;
    noise.sizes = {
        1,
        static_cast<SizesType>(exported_shape.back() / 4),
        static_cast<SizesType>(geometry.height / geometry.vae_scale_factor),
        static_cast<SizesType>(geometry.width / geometry.vae_scale_factor)};
    noise.data = randn(
        generator,
        dtype,
        static_cast<size_t>(noise.sizes[1] * noise.sizes[2] * noise.sizes[3]));
    return pack_latents(noise);
  }

  Result<HostTensor> prepare_encoded_latents(
      const HostTensor&,
      const HostTensor&,
      ScalarType,
      TorchGenerator&) const override {
    ET_LOG(Error, "FLUX img2img is not supported");
    return Error::NotSupported;
  }

  Result<size_t> prepare_img2img_latents(
      Scheduler&,
      HostTensor&,
      int,
      double,
      TorchGenerator&,
      ScalarType) const override {
    ET_LOG(Error, "FLUX img2img is not supported");
    return Error::NotSupported;
  }

  std::vector<TensorPtr> prepare_denoiser_inputs(
      const HostTensor& latents,
      float timestep,
      const std::vector<TensorPtr>& conditioning,
      const ImageGeometry& geometry,
      ScalarType dtype,
      std::optional<double> guidance_scale) const override {
    std::vector<float> scaled = {timestep};
    round_to(scaled, dtype);
    const auto& prompt = conditioning[0];
    const auto text_length = static_cast<SizesType>(prompt->size(1));
    std::vector<TensorPtr> inputs = {
        to_tensor(latents, dtype),
        scalar_tensor(scaled[0] / 1000, dtype),
        prompt,
        conditioning[1],
        make_tensor_ptr(
            {text_length, 3}, std::vector<int64_t>(text_length * 3, 0)),
        image_position_ids(
            geometry.height / geometry.vae_scale_factor / 2,
            geometry.width / geometry.vae_scale_factor / 2)};
    if (guidance_scale) {
      inputs.push_back(scalar_tensor(*guidance_scale, dtype));
    }
    return inputs;
  }

  HostTensor prepare_decode_latents(
      HostTensor latents,
      const ImageGeometry& geometry) const override {
    return unpack_latents(
        latents,
        geometry.height / geometry.vae_scale_factor,
        geometry.width / geometry.vae_scale_factor);
  }
};

} // namespace

HostTensor PipelineBehavior::prepare_model_input(
    const Scheduler& scheduler,
    const HostTensor& latents,
    size_t index,
    ScalarType dtype) const {
  HostTensor input = latents;
  scheduler.scale_model_input(input.data, index);
  round_to(input.data, dtype);
  return input;
}

Result<std::unique_ptr<PipelineBehavior>> make_pipeline_behavior(
    const std::string& name) {
  if (name == "sd15") {
    return std::unique_ptr<PipelineBehavior>(new SD15Behavior());
  }
  if (name == "sdxl") {
    return std::unique_ptr<PipelineBehavior>(new SDXLBehavior());
  }
  if (name == "flux") {
    return std::unique_ptr<PipelineBehavior>(new FluxBehavior());
  }
  ET_LOG(Error, "Unsupported diffusion pipeline %s", name.c_str());
  return Error::NotSupported;
}

HostTensor pack_latents(const HostTensor& latents) {
  ET_CHECK_MSG(latents.sizes.size() == 4, "Expected (B, C, H, W) latents");
  const int64_t batch = latents.sizes[0];
  const int64_t channels = latents.sizes[1];
  const int64_t height = latents.sizes[2];
  const int64_t width = latents.sizes[3];
  ET_CHECK_MSG(
      height % 2 == 0 && width % 2 == 0,
      "FLUX latent height and width must be even");
  HostTensor packed;
  packed.sizes = {
      static_cast<SizesType>(batch),
      static_cast<SizesType>(height * width / 4),
      static_cast<SizesType>(channels * 4)};
  packed.data.resize(latents.data.size());
  size_t out = 0;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t y = 0; y < height / 2; ++y) {
      for (int64_t x = 0; x < width / 2; ++x) {
        for (int64_t c = 0; c < channels; ++c) {
          for (int64_t dy = 0; dy < 2; ++dy) {
            for (int64_t dx = 0; dx < 2; ++dx) {
              packed.data[out++] =
                  latents.data
                      [((b * channels + c) * height + 2 * y + dy) * width +
                       2 * x + dx];
            }
          }
        }
      }
    }
  }
  return packed;
}

HostTensor unpack_latents(
    const HostTensor& latents,
    int64_t latent_height,
    int64_t latent_width) {
  ET_CHECK_MSG(latents.sizes.size() == 3, "Expected (B, L, C) latents");
  const int64_t batch = latents.sizes[0];
  const int64_t channels = latents.sizes[2] / 4;
  ET_CHECK_MSG(
      latents.sizes[2] % 4 == 0,
      "FLUX packed latent channels must be divisible by four");
  HostTensor unpacked;
  unpacked.sizes = {
      static_cast<SizesType>(batch),
      static_cast<SizesType>(channels),
      static_cast<SizesType>(latent_height),
      static_cast<SizesType>(latent_width)};
  unpacked.data.resize(latents.data.size());
  size_t in = 0;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t y = 0; y < latent_height / 2; ++y) {
      for (int64_t x = 0; x < latent_width / 2; ++x) {
        for (int64_t c = 0; c < channels; ++c) {
          for (int64_t dy = 0; dy < 2; ++dy) {
            for (int64_t dx = 0; dx < 2; ++dx) {
              unpacked.data
                  [((b * channels + c) * latent_height + 2 * y + dy) *
                       latent_width +
                   2 * x + dx] = latents.data[in++];
            }
          }
        }
      }
    }
  }
  return unpacked;
}

} // namespace executorch::extension::diffusers
