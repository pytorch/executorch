/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace executorch::extension::diffusers {

/**
 * Runtime contract embedded in an exported diffusion program as
 * get_diffusion_* constant methods. Mirrors DiffusionMetadata in
 * extension/diffusers/metadata.py.
 */
struct DiffusionMetadata {
  std::string pipeline;
  std::string scheduler;
  // Empty when the export did not record a prediction type.
  std::string scheduler_prediction_type;
  bool supports_cfg;
  bool accepts_guidance_input;
  bool supports_negative_prompt;
  double default_cfg_scale;
  std::optional<double> default_guidance_scale;
  int64_t guidance_input_dim;
  int64_t default_inference_steps;
  executorch::aten::ScalarType text_encoder_dtype;
  executorch::aten::ScalarType denoiser_dtype;
  executorch::aten::ScalarType vae_dtype;
  bool zero_negative_prompt;
  std::vector<int64_t> max_sequence_lengths;
  int64_t height;
  int64_t width;
  bool dynamic_shapes;
  int64_t min_size;
  int64_t max_size;
  int64_t dynamic_spatial_multiple;
  int64_t vae_scale_factor;
};

/** Reads the get_diffusion_* constant methods of a diffusion program. */
runtime::Result<DiffusionMetadata> read_diffusion_metadata(Module& module);

} // namespace executorch::extension::diffusers
