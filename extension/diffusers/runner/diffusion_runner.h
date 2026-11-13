/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/diffusers/runner/metadata.h>
#include <executorch/extension/diffusers/runner/pipelines.h>
#include <executorch/extension/diffusers/runner/schedulers.h>
#include <executorch/extension/diffusers/runner/tensor_util.h>
#include <executorch/extension/diffusers/runner/text_inputs.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/result.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace executorch::extension::diffusers {

/** An RGB image in height, width, channel order. */
struct GeneratedImage {
  int64_t height;
  int64_t width;
  std::vector<uint8_t> rgb;
};

/** Options of one generation; unset values use the program's defaults. */
struct GenerationConfig {
  std::optional<std::string> negative_prompt;
  std::optional<int> steps;
  std::optional<double> cfg_scale;
  std::optional<double> guidance_scale;
  uint64_t seed = 0;
  std::optional<int64_t> height;
  std::optional<int64_t> width;
  // img2img input of shape (1, 3, height, width) with values in [0, 1].
  std::optional<HostTensor> image;
  double strength = 0.8;
  // Every preview_interval steps before the last, on_preview receives the
  // step, the step count and the decoded current latents.
  size_t preview_interval = 0;
  std::function<void(size_t, size_t, const GeneratedImage&)> on_preview;
};

/**
 * Runs a multi-method diffusion program exported from Diffusers, with the
 * tokenizers and scheduler config of the matching Hugging Face snapshot.
 */
class DiffusionRunner {
 public:
  /**
   * Loads a program and the assets of a Hugging Face snapshot directory.
   * data_path names an optional external weights file, and backend_options
   * are passed to every method the runner loads.
   */
  static runtime::Result<std::unique_ptr<DiffusionRunner>> load(
      const std::string& model_path,
      const std::string& model_dir,
      const std::string& data_path = "",
      const runtime::LoadBackendOptionsMap* backend_options = nullptr);

  /**
   * Generates one image. on_step receives the completed step and the step
   * count after every denoising step.
   */
  runtime::Result<GeneratedImage> generate(
      const std::string& prompt,
      const GenerationConfig& config,
      const std::function<void(size_t, size_t)>& on_step = {});

 private:
  DiffusionRunner(
      std::unique_ptr<Module> module,
      DiffusionMetadata metadata,
      std::unique_ptr<PipelineBehavior> behavior,
      std::vector<std::unique_ptr<PromptTokenizer>> tokenizers,
      std::unique_ptr<Scheduler> scheduler,
      nlohmann::json scheduler_config);

  runtime::Result<std::vector<HostTensor>> execute(
      const std::string& method,
      const std::vector<TensorPtr>& inputs);
  runtime::Result<std::vector<HostTensor>> encode_prompt(
      const std::string& prompt);
  runtime::Result<ImageGeometry> resolve_geometry(
      std::optional<int64_t> height,
      std::optional<int64_t> width) const;
  runtime::Result<std::optional<double>> resolve_guidance_scale(
      std::optional<double> guidance_scale) const;
  runtime::Result<GeneratedImage> decode(
      const HostTensor& latents,
      const ImageGeometry& geometry,
      executorch::aten::ScalarType dtype);

  std::unique_ptr<Module> module_;
  DiffusionMetadata metadata_;
  std::unique_ptr<PipelineBehavior> behavior_;
  std::vector<std::unique_ptr<PromptTokenizer>> tokenizers_;
  std::unique_ptr<Scheduler> scheduler_;
  nlohmann::json scheduler_config_;
};

} // namespace executorch::extension::diffusers
