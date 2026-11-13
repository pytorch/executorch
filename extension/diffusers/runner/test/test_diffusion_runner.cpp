/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/diffusers/runner/diffusion_runner.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <string>

using executorch::extension::diffusers::DiffusionRunner;
using executorch::extension::diffusers::GenerationConfig;
using executorch::extension::diffusers::HostTensor;
using json = nlohmann::json;

// Repeats the Python runner's generations on the programs of
// export_test_programs.py. A pixel may differ by one level, since the noise
// matches torch.randn within two ulps.
TEST(DiffusionRunnerTest, MatchesPythonRunner) {
  std::ifstream file(std::getenv("DIFFUSERS_GOLDENS_PATH"));
  const auto goldens = json::parse(file).at("runner");
  const std::string directory = std::getenv("DIFFUSERS_PROGRAMS_DIR");
  for (const auto& run : goldens.at("cases")) {
    const auto program = directory + "/" + run.at("program").get<std::string>();
    const auto prompt = run.at("prompt").get<std::string>();
    SCOPED_TRACE(program + ": " + prompt);
    auto runner = DiffusionRunner::load(program + ".pte", program);
    ASSERT_TRUE(runner.ok());

    GenerationConfig config;
    config.seed = run.at("seed").get<uint64_t>();
    config.steps = run.at("steps").get<int>();
    if (run.contains("cfg_scale")) {
      config.cfg_scale = run.at("cfg_scale").get<double>();
    }
    if (run.contains("guidance_scale")) {
      config.guidance_scale = run.at("guidance_scale").get<double>();
    }
    if (run.contains("negative_prompt")) {
      config.negative_prompt = run.at("negative_prompt").get<std::string>();
    }
    if (run.contains("strength")) {
      config.image = HostTensor{
          {1, 3, 16, 16}, goldens.at("image").get<std::vector<float>>()};
      config.strength = run.at("strength").get<double>();
    }
    // img2img runs the last int(steps * strength) steps.
    const auto steps = config.image
        ? static_cast<size_t>(*config.steps * config.strength)
        : static_cast<size_t>(*config.steps);
    size_t completed = 0;
    auto image =
        runner.get()->generate(prompt, config, [&](size_t step, size_t total) {
          EXPECT_EQ(step, ++completed);
          EXPECT_EQ(total, steps);
        });
    ASSERT_TRUE(image.ok());
    EXPECT_EQ(completed, steps);

    const auto expected = run.at("pixels").get<std::vector<int>>();
    ASSERT_EQ(image->rgb.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_LE(std::abs(image->rgb[i] - expected[i]), 1) << "at " << i;
    }
  }
}
