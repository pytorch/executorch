/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/diffusers/runner/diffusion_runner.h>

#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <sstream>

namespace executorch::extension::diffusers {

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;
using json = nlohmann::json;

DiffusionRunner::DiffusionRunner(
    std::unique_ptr<Module> module,
    DiffusionMetadata metadata,
    std::unique_ptr<PipelineBehavior> behavior,
    std::vector<std::unique_ptr<PromptTokenizer>> tokenizers,
    std::unique_ptr<Scheduler> scheduler,
    json scheduler_config)
    : module_(std::move(module)),
      metadata_(std::move(metadata)),
      behavior_(std::move(behavior)),
      tokenizers_(std::move(tokenizers)),
      scheduler_(std::move(scheduler)),
      scheduler_config_(std::move(scheduler_config)) {}

Result<std::unique_ptr<DiffusionRunner>> DiffusionRunner::load(
    const std::string& model_path,
    const std::string& model_dir,
    const std::string& data_path,
    const runtime::LoadBackendOptionsMap* backend_options) {
  auto module = data_path.empty()
      ? std::make_unique<Module>(model_path, Module::LoadMode::Mmap)
      : std::make_unique<Module>(model_path, data_path, Module::LoadMode::Mmap);
  auto metadata = ET_UNWRAP(read_diffusion_metadata(*module));
  auto behavior = ET_UNWRAP(make_pipeline_behavior(metadata.pipeline));

  const auto slots = behavior->tokenizer_slots();
  ET_CHECK_OR_RETURN_ERROR(
      slots.size() == metadata.max_sequence_lengths.size() &&
          slots.size() == behavior->text_encoder_input_names().size(),
      InvalidProgram,
      "%s requires %zu tokenizer(s)",
      metadata.pipeline.c_str(),
      metadata.max_sequence_lengths.size());
  std::vector<std::unique_ptr<PromptTokenizer>> tokenizers;
  for (size_t i = 0; i < slots.size(); ++i) {
    tokenizers.push_back(ET_UNWRAP(PromptTokenizer::load(
        model_dir + "/" + slots[i], metadata.max_sequence_lengths[i])));
  }

  const std::string config_path =
      model_dir + "/scheduler/scheduler_config.json";
  std::ifstream config_file(config_path);
  ET_CHECK_OR_RETURN_ERROR(
      config_file.good(), NotFound, "Cannot read %s", config_path.c_str());
  std::stringstream config_text;
  config_text << config_file.rdbuf();
  auto scheduler_config =
      json::parse(config_text.str(), nullptr, /*allow_exceptions=*/false);
  ET_CHECK_OR_RETURN_ERROR(
      scheduler_config.is_object(),
      InvalidArgument,
      "Invalid %s",
      config_path.c_str());
  if (!metadata.scheduler_prediction_type.empty()) {
    const auto it = scheduler_config.find("prediction_type");
    const std::string prediction_type =
        it == scheduler_config.end() || it->is_null() ? "epsilon"
                                                      : it->get<std::string>();
    ET_CHECK_OR_RETURN_ERROR(
        prediction_type == metadata.scheduler_prediction_type,
        InvalidArgument,
        "scheduler prediction_type %s does not match exported "
        "prediction_type %s",
        prediction_type.c_str(),
        metadata.scheduler_prediction_type.c_str());
  }
  auto scheduler =
      ET_UNWRAP(make_scheduler(metadata.scheduler, config_text.str()));

  std::vector<std::string> methods = {"denoise", "decode"};
  for (size_t i = 0; i < slots.size(); ++i) {
    methods.push_back("text_encoder_" + std::to_string(i));
  }
  const auto method_names = ET_UNWRAP(module->method_names());
  if (method_names.count("encode")) {
    methods.push_back("encode");
  }
  for (const auto& method : methods) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        module->load_method(method, nullptr, nullptr, backend_options),
        "Failed to load method %s",
        method.c_str());
  }

  return std::unique_ptr<DiffusionRunner>(new DiffusionRunner(
      std::move(module),
      std::move(metadata),
      std::move(behavior),
      std::move(tokenizers),
      std::move(scheduler),
      std::move(scheduler_config)));
}

Result<std::vector<HostTensor>> DiffusionRunner::execute(
    const std::string& method,
    const std::vector<TensorPtr>& inputs) {
  std::vector<EValue> values;
  values.reserve(inputs.size());
  for (const auto& input : inputs) {
    values.emplace_back(*input);
  }
  const auto outputs =
      ET_UNWRAP(module_->execute(method, values), "%s failed", method.c_str());
  std::vector<HostTensor> results;
  for (const auto& output : outputs) {
    const auto tensor = ET_UNWRAP(
        output.tryToTensor(), "%s returned a non-tensor", method.c_str());
    results.push_back(ET_UNWRAP(to_host(tensor)));
  }
  return results;
}

Result<std::vector<HostTensor>> DiffusionRunner::encode_prompt(
    const std::string& prompt) {
  const auto input_names = behavior_->text_encoder_input_names();
  std::vector<std::vector<HostTensor>> encoder_outputs;
  for (size_t i = 0; i < tokenizers_.size(); ++i) {
    const auto encoded = ET_UNWRAP(tokenizers_[i]->encode(prompt));
    const auto length = static_cast<SizesType>(encoded.input_ids.size());
    std::vector<TensorPtr> inputs;
    for (const auto& name : input_names[i]) {
      if (name == "input_ids") {
        inputs.push_back(make_tensor_ptr({1, length}, encoded.input_ids));
      } else if (name == "attention_mask") {
        inputs.push_back(make_tensor_ptr({1, length}, encoded.attention_mask));
      } else {
        ET_LOG(Error, "Unsupported text encoder input %s", name.c_str());
        return Error::NotSupported;
      }
    }
    encoder_outputs.push_back(
        ET_UNWRAP(execute("text_encoder_" + std::to_string(i), inputs)));
  }
  return behavior_->assemble_conditioning(std::move(encoder_outputs));
}

Result<ImageGeometry> DiffusionRunner::resolve_geometry(
    std::optional<int64_t> height,
    std::optional<int64_t> width) const {
  ET_CHECK_OR_RETURN_ERROR(
      height.value_or(1) > 0 && width.value_or(1) > 0,
      InvalidArgument,
      "height and width must be positive");
  const int64_t h = height.value_or(metadata_.height);
  const int64_t w = width.value_or(metadata_.width);
  if (!metadata_.dynamic_shapes) {
    ET_CHECK_OR_RETURN_ERROR(
        h == metadata_.height && w == metadata_.width,
        InvalidArgument,
        "program was exported with static size %" PRId64 "x%" PRId64
        ", but requested %" PRId64 "x%" PRId64
        "; use the exported height and width or re-export with dynamic shapes",
        metadata_.height,
        metadata_.width,
        h,
        w);
  } else {
    ET_CHECK_OR_RETURN_ERROR(
        metadata_.min_size <= h && h <= metadata_.max_size &&
            metadata_.min_size <= w && w <= metadata_.max_size,
        InvalidArgument,
        "height and width must be between %" PRId64 " and %" PRId64,
        metadata_.min_size,
        metadata_.max_size);
  }
  ET_CHECK_OR_RETURN_ERROR(
      h % metadata_.vae_scale_factor == 0 &&
          w % metadata_.vae_scale_factor == 0,
      InvalidArgument,
      "height and width must be divisible by %" PRId64,
      metadata_.vae_scale_factor);
  ET_CHECK_OR_RETURN_ERROR(
      !metadata_.dynamic_shapes ||
          (h % metadata_.dynamic_spatial_multiple == 0 &&
           w % metadata_.dynamic_spatial_multiple == 0),
      InvalidArgument,
      "height and width must be divisible by %" PRId64
      " for this dynamic program",
      metadata_.dynamic_spatial_multiple);
  return ImageGeometry{h, w, metadata_.vae_scale_factor};
}

Result<std::optional<double>> DiffusionRunner::resolve_guidance_scale(
    std::optional<double> guidance_scale) const {
  if (!guidance_scale) {
    guidance_scale = metadata_.default_guidance_scale;
  }
  if (!guidance_scale) {
    return guidance_scale;
  }
  ET_CHECK_OR_RETURN_ERROR(
      std::isfinite(*guidance_scale) && *guidance_scale >= 0,
      InvalidArgument,
      "guidance_scale must be finite and non-negative");
  if (!metadata_.accepts_guidance_input) {
    ET_CHECK_OR_RETURN_ERROR(
        *guidance_scale == 0,
        InvalidArgument,
        "this program does not accept a guidance input");
    return std::optional<double>();
  }
  return guidance_scale;
}

Result<GeneratedImage> DiffusionRunner::generate(
    const std::string& prompt,
    const GenerationConfig& config,
    const std::function<void(size_t, size_t)>& on_step) {
  const int steps = config.steps.value_or(
      static_cast<int>(metadata_.default_inference_steps));
  const double cfg_scale =
      config.cfg_scale.value_or(metadata_.default_cfg_scale);
  ET_CHECK_OR_RETURN_ERROR(
      steps > 0, InvalidArgument, "steps must be positive");
  ET_CHECK_OR_RETURN_ERROR(
      std::isfinite(cfg_scale) && cfg_scale >= 0,
      InvalidArgument,
      "cfg_scale must be finite and non-negative");
  ET_CHECK_OR_RETURN_ERROR(
      cfg_scale <= 1 || metadata_.supports_cfg,
      InvalidArgument,
      "this program does not support CFG");
  const auto guidance_scale =
      ET_UNWRAP(resolve_guidance_scale(config.guidance_scale));
  ET_CHECK_OR_RETURN_ERROR(
      config.strength > 0 && config.strength <= 1,
      InvalidArgument,
      "strength must be in (0, 1]");
  ET_CHECK_OR_RETURN_ERROR(
      !config.negative_prompt || metadata_.supports_negative_prompt,
      InvalidArgument,
      "this program does not support negative prompts");
  const auto geometry =
      ET_UNWRAP(resolve_geometry(config.height, config.width));
  const bool use_cfg = metadata_.supports_cfg &&
      behavior_->should_use_cfg(cfg_scale, config.negative_prompt.has_value());

  auto positive = ET_UNWRAP(encode_prompt(prompt));
  std::vector<HostTensor> negative;
  if (use_cfg) {
    if (config.negative_prompt) {
      negative = ET_UNWRAP(encode_prompt(*config.negative_prompt));
    } else if (metadata_.zero_negative_prompt) {
      for (const auto& tensor : positive) {
        negative.push_back(
            {tensor.sizes, std::vector<float>(tensor.data.size(), 0.0f)});
      }
    } else {
      negative = ET_UNWRAP(encode_prompt(""));
    }
  }
  TorchGenerator generator(config.seed);

  const auto denoise_meta =
      ET_UNWRAP(module_->method_meta("denoise"), "Missing denoise method");
  const auto latent_meta = ET_UNWRAP(denoise_meta.input_tensor_meta(0));
  const std::vector<int64_t> exported_shape(
      latent_meta.sizes().begin(), latent_meta.sizes().end());
  ET_CHECK_OK_OR_RETURN_ERROR(behavior_->prepare_schedule(
      *scheduler_, scheduler_config_, steps, geometry));
  const ScalarType dtype = metadata_.denoiser_dtype;

  HostTensor latents;
  size_t start = 0;
  if (!config.image) {
    latents = behavior_->create_latents(
        exported_shape, geometry, generator, dtype, *scheduler_);
  } else {
    const auto& image = *config.image;
    const bool expected_shape = image.sizes ==
        std::vector<SizesType>{
            1,
            3,
            static_cast<SizesType>(geometry.height),
            static_cast<SizesType>(geometry.width)};
    ET_CHECK_OR_RETURN_ERROR(
        expected_shape,
        InvalidArgument,
        "image shape must be (1, 3, %" PRId64 ", %" PRId64 ")",
        geometry.height,
        geometry.width);
    ET_CHECK_OR_RETURN_ERROR(
        std::all_of(
            image.data.begin(),
            image.data.end(),
            [](float value) { return value >= 0 && value <= 1; }),
        InvalidArgument,
        "image values must be in [0, 1]");
    const auto encode_outputs = ET_UNWRAP(module_->execute(
        "encode", std::vector<EValue>{*to_tensor(image, metadata_.vae_dtype)}));
    ET_CHECK_OR_RETURN_ERROR(
        encode_outputs.size() == 2,
        InvalidProgram,
        "encode must return a mean and a standard deviation");
    const auto mean_tensor = ET_UNWRAP(encode_outputs[0].tryToTensor());
    const auto mean = ET_UNWRAP(to_host(mean_tensor));
    const auto deviation =
        ET_UNWRAP(to_host(ET_UNWRAP(encode_outputs[1].tryToTensor())));
    latents = ET_UNWRAP(behavior_->prepare_encoded_latents(
        mean, deviation, mean_tensor.scalar_type(), generator));
    start = ET_UNWRAP(behavior_->prepare_img2img_latents(
        *scheduler_, latents, steps, config.strength, generator, dtype));
  }

  std::vector<std::vector<TensorPtr>> conditionings;
  for (const auto* conditioning : {&negative, &positive}) {
    if (conditioning->empty()) {
      continue;
    }
    std::vector<TensorPtr> tensors;
    for (const auto& tensor : *conditioning) {
      tensors.push_back(to_tensor(tensor, dtype));
    }
    conditionings.push_back(std::move(tensors));
  }

  const auto& timesteps = scheduler_->timesteps();
  const size_t calls = timesteps.size() - start;
  const size_t warmup = std::min(scheduler_->warmup_calls(), calls - 1);
  for (size_t index = start; index < timesteps.size(); ++index) {
    const auto model_input =
        behavior_->prepare_model_input(*scheduler_, latents, index, dtype);
    std::vector<HostTensor> predictions;
    for (const auto& conditioning : conditionings) {
      auto outputs = ET_UNWRAP(execute(
          "denoise",
          behavior_->prepare_denoiser_inputs(
              model_input,
              timesteps[index],
              conditioning,
              geometry,
              dtype,
              guidance_scale)));
      predictions.push_back(std::move(outputs[0]));
    }
    // CFG: unconditional + cfg_scale * (conditional - unconditional).
    HostTensor prediction = std::move(predictions.back());
    if (predictions.size() == 2) {
      const auto& unconditional = predictions[0].data;
      for (size_t i = 0; i < prediction.data.size(); ++i) {
        prediction.data[i] = unconditional[i] +
            static_cast<float>(cfg_scale) *
                (prediction.data[i] - unconditional[i]);
      }
      round_to(prediction.data, dtype);
    }
    scheduler_->step(prediction.data, latents.data, index);
    round_to(latents.data, dtype);
    const size_t completed = index - start + 1;
    if (completed <= warmup) {
      continue;
    }
    const size_t step = completed - warmup;
    const size_t total = calls - warmup;
    if (on_step) {
      on_step(step, total);
    }
    if (config.on_preview && config.preview_interval > 0 &&
        step % config.preview_interval == 0 && step < total) {
      config.on_preview(
          step, total, ET_UNWRAP(decode(latents, geometry, dtype)));
    }
  }
  return decode(latents, geometry, dtype);
}

Result<GeneratedImage> DiffusionRunner::decode(
    const HostTensor& latents,
    const ImageGeometry& geometry,
    ScalarType dtype) {
  const auto decode_latents =
      behavior_->prepare_decode_latents(latents, geometry);
  const auto decoded =
      ET_UNWRAP(execute("decode", {to_tensor(decode_latents, dtype)}));
  const auto& pixels = decoded[0];
  ET_CHECK_OR_RETURN_ERROR(
      pixels.sizes.size() == 4 && pixels.sizes[0] == 1 && pixels.sizes[1] == 3,
      InvalidProgram,
      "decode must return a (1, 3, height, width) image");
  GeneratedImage result{pixels.sizes[2], pixels.sizes[3], {}};
  const size_t plane = static_cast<size_t>(result.height * result.width);
  result.rgb.resize(plane * 3);
  for (size_t pixel = 0; pixel < plane; ++pixel) {
    for (size_t channel = 0; channel < 3; ++channel) {
      const float value =
          std::clamp(pixels.data[channel * plane + pixel], 0.0f, 1.0f);
      result.rgb[pixel * 3 + channel] = static_cast<uint8_t>(value * 255);
    }
  }
  return result;
}

} // namespace executorch::extension::diffusers
