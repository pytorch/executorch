/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/diffusers/runner/schedulers.h>
#include <executorch/extension/diffusers/runner/tensor_util.h>
#include <executorch/extension/diffusers/runner/torch_rng.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace executorch::extension::diffusers {

/** Output image size and the VAE downsampling factor. */
struct ImageGeometry {
  int64_t height;
  int64_t width;
  int64_t vae_scale_factor;
};

/**
 * Runtime behavior of one diffusion pipeline. Mirrors
 * DiffusionPipelineBehavior in extension/diffusers/pipeline_behaviors.
 */
class PipelineBehavior {
 public:
  virtual ~PipelineBehavior() = default;

  /** Snapshot directories of the tokenizers, one per text encoder. */
  virtual std::vector<std::string> tokenizer_slots() const = 0;

  /** Tokenizer outputs each text encoder method takes, in order. */
  virtual std::vector<std::vector<std::string>> text_encoder_input_names()
      const = 0;

  /** Converts the outputs of every text encoder into denoiser conditioning. */
  virtual std::vector<HostTensor> assemble_conditioning(
      std::vector<std::vector<HostTensor>> encoder_outputs) const = 0;

  /** Whether generation runs unconditional conditioning. */
  virtual bool should_use_cfg(double cfg_scale, bool has_negative_prompt)
      const {
    return cfg_scale > 1;
  }

  /** Configures the scheduler for the given number of steps. */
  virtual runtime::Error prepare_schedule(
      Scheduler& scheduler,
      const nlohmann::json& scheduler_config,
      int steps,
      const ImageGeometry& geometry) const {
    return scheduler.set_timesteps(steps);
  }

  /** Creates the initial noise in the denoiser's latent layout. */
  virtual HostTensor create_latents(
      const std::vector<int64_t>& exported_shape,
      const ImageGeometry& geometry,
      TorchGenerator& generator,
      executorch::aten::ScalarType dtype,
      const Scheduler& scheduler) const = 0;

  /** Samples encoded VAE latents and converts them to the denoiser layout. */
  virtual runtime::Result<HostTensor> prepare_encoded_latents(
      const HostTensor& mean,
      const HostTensor& std,
      executorch::aten::ScalarType dtype,
      TorchGenerator& generator) const = 0;

  /**
   * Noises encoded image latents in place for img2img and returns the index
   * of the first denoising step.
   */
  virtual runtime::Result<size_t> prepare_img2img_latents(
      Scheduler& scheduler,
      HostTensor& latents,
      int steps,
      double strength,
      TorchGenerator& generator,
      executorch::aten::ScalarType dtype) const = 0;

  /** Applies the scheduler's denoiser input scaling. */
  virtual HostTensor prepare_model_input(
      const Scheduler& scheduler,
      const HostTensor& latents,
      size_t index,
      executorch::aten::ScalarType dtype) const;

  /** Assembles the inputs of one call of the denoise method. */
  virtual std::vector<TensorPtr> prepare_denoiser_inputs(
      const HostTensor& latents,
      float timestep,
      const std::vector<TensorPtr>& conditioning,
      const ImageGeometry& geometry,
      executorch::aten::ScalarType dtype,
      std::optional<double> guidance_scale) const = 0;

  /** Converts denoiser latents to the VAE decoder layout. */
  virtual HostTensor prepare_decode_latents(
      HostTensor latents,
      const ImageGeometry& geometry) const {
    return latents;
  }
};

/** Creates the behavior of a pipeline name recorded in the program. */
runtime::Result<std::unique_ptr<PipelineBehavior>> make_pipeline_behavior(
    const std::string& name);

/** Packs 2x2 latent neighborhoods into channels: (1, C, H, W) to (1, HW/4, 4C).
 */
HostTensor pack_latents(const HostTensor& latents);

/** Restores packed latents to (1, C, latent_height, latent_width). */
HostTensor unpack_latents(
    const HostTensor& latents,
    int64_t latent_height,
    int64_t latent_width);

} // namespace executorch::extension::diffusers
