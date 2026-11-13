/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/diffusers/runner/pipelines.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>

using executorch::aten::ScalarType;
using executorch::extension::diffusers::HostTensor;
using executorch::extension::diffusers::ImageGeometry;
using executorch::extension::diffusers::make_pipeline_behavior;
using executorch::extension::diffusers::make_scheduler;
using executorch::extension::diffusers::pack_latents;
using executorch::extension::diffusers::TorchGenerator;
using executorch::extension::diffusers::unpack_latents;
using executorch::runtime::Error;
using json = nlohmann::json;

namespace {

json pipeline_goldens() {
  std::ifstream file(std::getenv("DIFFUSERS_GOLDENS_PATH"));
  return json::parse(file).at("pipelines");
}

std::vector<int32_t> sizes_of(const json& sizes) {
  return sizes.get<std::vector<int32_t>>();
}

std::vector<float> floats_of(const json& values) {
  const auto doubles = values.get<std::vector<double>>();
  return std::vector<float>(doubles.begin(), doubles.end());
}

// Latents come from torch-compatible noise, which matches torch.randn within
// two ulps, so they are compared within two ulps of their dtype.
void expect_within_two_ulps(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    ScalarType dtype) {
  ASSERT_EQ(actual.size(), expected.size());
  const auto ordered = [dtype](float value) -> int64_t {
    const uint16_t bits = dtype == ScalarType::Half
        ? executorch::aten::Half(value).x
        : executorch::aten::BFloat16(value).x;
    const int64_t magnitude = bits & 0x7FFF;
    return bits & 0x8000 ? -magnitude : magnitude;
  };
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_LE(std::llabs(ordered(actual[i]) - ordered(expected[i])), 2)
        << "at " << i;
  }
}

} // namespace

TEST(PipelineTest, FluxLatentsMatchPython) {
  const auto golden = pipeline_goldens().at("flux_latents");
  auto behavior = make_pipeline_behavior("flux");
  ASSERT_TRUE(behavior.ok());
  auto scheduler = make_scheduler("FlowMatchEulerDiscreteScheduler", "{}");
  ASSERT_TRUE(scheduler.ok());
  TorchGenerator generator(golden.at("seed").get<uint64_t>());
  const auto latents = behavior.get()->create_latents(
      {1, 16, 64},
      ImageGeometry{64, 64, 8},
      generator,
      ScalarType::BFloat16,
      *scheduler.get());
  EXPECT_EQ(latents.sizes, sizes_of(golden.at("sizes")));
  expect_within_two_ulps(
      latents.data, floats_of(golden.at("values")), ScalarType::BFloat16);
}

TEST(PipelineTest, StableDiffusionLatentsMatchPython) {
  const auto golden = pipeline_goldens().at("sd_latents");
  auto behavior = make_pipeline_behavior("sdxl");
  ASSERT_TRUE(behavior.ok());
  auto scheduler =
      make_scheduler("EulerDiscreteScheduler", golden.at("config").dump());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_EQ(
      scheduler.get()->set_timesteps(golden.at("steps").get<int>()), Error::Ok);
  TorchGenerator generator(golden.at("seed").get<uint64_t>());
  const auto latents = behavior.get()->create_latents(
      {1, 4, 8, 8},
      ImageGeometry{64, 64, 8},
      generator,
      ScalarType::Half,
      *scheduler.get());
  EXPECT_EQ(latents.sizes, sizes_of(golden.at("sizes")));
  expect_within_two_ulps(
      latents.data, floats_of(golden.at("values")), ScalarType::Half);
}

TEST(PipelineTest, FluxScheduleMatchesPython) {
  const auto golden = pipeline_goldens().at("flux_schedule");
  auto behavior = make_pipeline_behavior("flux");
  ASSERT_TRUE(behavior.ok());
  auto scheduler = make_scheduler(
      "FlowMatchEulerDiscreteScheduler", golden.at("config").dump());
  ASSERT_TRUE(scheduler.ok());
  ASSERT_EQ(
      behavior.get()->prepare_schedule(
          *scheduler.get(),
          golden.at("config"),
          golden.at("steps").get<int>(),
          ImageGeometry{
              golden.at("height").get<int64_t>(),
              golden.at("width").get<int64_t>(),
              8}),
      Error::Ok);
  const auto expected = floats_of(golden.at("timesteps"));
  const auto& timesteps = scheduler.get()->timesteps();
  ASSERT_EQ(timesteps.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(timesteps[i], expected[i], 1e-6 * expected[i]) << "at " << i;
  }
}

TEST(PipelineTest, FluxDenoiserInputsMatchPython) {
  const auto golden = pipeline_goldens().at("flux_denoiser_inputs");
  auto behavior = make_pipeline_behavior("flux");
  ASSERT_TRUE(behavior.ok());
  HostTensor latents{{1, 16, 64}, std::vector<float>(16 * 64)};
  const std::vector<executorch::extension::TensorPtr> conditioning = {
      executorch::extension::diffusers::to_tensor(
          HostTensor{{1, 7, 8}, std::vector<float>(56)}, ScalarType::BFloat16),
      executorch::extension::diffusers::to_tensor(
          HostTensor{{1, 4}, std::vector<float>(4)}, ScalarType::BFloat16)};
  const auto inputs = behavior.get()->prepare_denoiser_inputs(
      latents,
      golden.at("timestep").get<float>(),
      conditioning,
      ImageGeometry{64, 64, 8},
      ScalarType::BFloat16,
      3.5);
  ASSERT_EQ(inputs.size(), 7);
  EXPECT_EQ(inputs[1]->scalar_type(), ScalarType::BFloat16);
  EXPECT_EQ(
      static_cast<float>(
          inputs[1]->const_data_ptr<executorch::aten::BFloat16>()[0]),
      golden.at("scaled_timestep")[0].get<float>());
  EXPECT_EQ(inputs[4]->size(0), golden.at("text_ids_sizes")[0].get<int64_t>());
  EXPECT_EQ(inputs[4]->size(1), golden.at("text_ids_sizes")[1].get<int64_t>());
  const auto image_ids =
      golden.at("image_ids").get<std::vector<std::vector<int64_t>>>();
  ASSERT_EQ(inputs[5]->numel(), image_ids.size() * 3);
  const int64_t* ids = inputs[5]->const_data_ptr<int64_t>();
  for (size_t i = 0; i < image_ids.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(ids[i * 3 + j], image_ids[i][j]);
    }
  }
  EXPECT_EQ(
      static_cast<float>(
          inputs[6]->const_data_ptr<executorch::aten::BFloat16>()[0]),
      golden.at("guidance")[0].get<float>());
}

TEST(PipelineTest, PackAndUnpackAreInverse) {
  HostTensor latents{{1, 3, 4, 6}, std::vector<float>(72)};
  for (size_t i = 0; i < latents.data.size(); ++i) {
    latents.data[i] = static_cast<float>(i);
  }
  const auto packed = pack_latents(latents);
  EXPECT_EQ(packed.sizes, (std::vector<int32_t>{1, 6, 12}));
  // The first packed token holds the top-left 2x2 block of every channel.
  EXPECT_EQ(
      std::vector<float>(packed.data.begin(), packed.data.begin() + 4),
      (std::vector<float>{0, 1, 6, 7}));
  const auto unpacked = unpack_latents(packed, 4, 6);
  EXPECT_EQ(unpacked.sizes, latents.sizes);
  EXPECT_EQ(unpacked.data, latents.data);
}

TEST(PipelineTest, SDXLConcatenatesHiddenStates) {
  auto behavior = make_pipeline_behavior("sdxl");
  ASSERT_TRUE(behavior.ok());
  const auto conditioning = behavior.get()->assemble_conditioning(
      {{HostTensor{{1, 2, 2}, {1, 2, 3, 4}}},
       {HostTensor{{1, 2, 1}, {5, 6}}, HostTensor{{1, 3}, {7, 8, 9}}}});
  ASSERT_EQ(conditioning.size(), 2);
  EXPECT_EQ(conditioning[0].sizes, (std::vector<int32_t>{1, 2, 3}));
  EXPECT_EQ(conditioning[0].data, (std::vector<float>{1, 2, 5, 3, 4, 6}));
  EXPECT_EQ(conditioning[1].data, (std::vector<float>{7, 8, 9}));
}

TEST(PipelineTest, RejectsUnknownPipelines) {
  EXPECT_EQ(make_pipeline_behavior("flux2").error(), Error::NotSupported);
}
