/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/diffusers/runner/metadata.h>

#include <gtest/gtest.h>

#include <cstdlib>

using executorch::aten::ScalarType;
using executorch::extension::Module;
using executorch::extension::diffusers::read_diffusion_metadata;
using executorch::runtime::Error;

namespace {

std::string fixture(const std::string& name) {
  return std::string(std::getenv("DIFFUSERS_PROGRAMS_DIR")) +
      "/DiffusionMetadata_" + name + ".pte";
}

} // namespace

TEST(DiffusionMetadataTest, ReadsFluxProgram) {
  Module module(fixture("flux"));
  const auto result = read_diffusion_metadata(module);
  ASSERT_TRUE(result.ok());
  const auto& metadata = result.get();
  EXPECT_EQ(metadata.pipeline, "flux");
  EXPECT_EQ(metadata.scheduler, "FlowMatchEulerDiscreteScheduler");
  EXPECT_EQ(metadata.scheduler_prediction_type, "");
  EXPECT_TRUE(metadata.supports_cfg);
  EXPECT_TRUE(metadata.accepts_guidance_input);
  EXPECT_TRUE(metadata.supports_negative_prompt);
  EXPECT_EQ(metadata.default_cfg_scale, 1.0);
  ASSERT_TRUE(metadata.default_guidance_scale.has_value());
  EXPECT_EQ(*metadata.default_guidance_scale, 3.5);
  EXPECT_EQ(metadata.guidance_input_dim, 0);
  EXPECT_EQ(metadata.default_inference_steps, 28);
  EXPECT_EQ(metadata.text_encoder_dtype, ScalarType::BFloat16);
  EXPECT_EQ(metadata.denoiser_dtype, ScalarType::BFloat16);
  EXPECT_EQ(metadata.vae_dtype, ScalarType::Float);
  EXPECT_FALSE(metadata.zero_negative_prompt);
  EXPECT_EQ(metadata.max_sequence_lengths, (std::vector<int64_t>{77, 512}));
  EXPECT_EQ(metadata.height, 1024);
  EXPECT_EQ(metadata.width, 768);
  EXPECT_TRUE(metadata.dynamic_shapes);
  EXPECT_EQ(metadata.min_size, 256);
  EXPECT_EQ(metadata.max_size, 1024);
  EXPECT_EQ(metadata.dynamic_spatial_multiple, 16);
  EXPECT_EQ(metadata.vae_scale_factor, 8);
}

TEST(DiffusionMetadataTest, ReadsStableDiffusionProgram) {
  Module module(fixture("sd15"));
  const auto result = read_diffusion_metadata(module);
  ASSERT_TRUE(result.ok());
  const auto& metadata = result.get();
  EXPECT_EQ(metadata.pipeline, "sd15");
  EXPECT_EQ(metadata.scheduler_prediction_type, "epsilon");
  EXPECT_FALSE(metadata.accepts_guidance_input);
  EXPECT_EQ(metadata.default_cfg_scale, 7.5);
  EXPECT_FALSE(metadata.default_guidance_scale.has_value());
  EXPECT_EQ(metadata.denoiser_dtype, ScalarType::Half);
  EXPECT_EQ(metadata.max_sequence_lengths, (std::vector<int64_t>{77}));
  EXPECT_FALSE(metadata.dynamic_shapes);
}

TEST(DiffusionMetadataTest, FailsWithoutMetadata) {
  Module module(fixture("bare"));
  EXPECT_FALSE(read_diffusion_metadata(module).ok());
}
