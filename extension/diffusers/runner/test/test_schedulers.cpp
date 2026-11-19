/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/diffusers/runner/schedulers.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>

using executorch::extension::diffusers::make_scheduler;
using executorch::extension::diffusers::Scheduler;
using executorch::runtime::Error;
using json = nlohmann::json;

namespace {

constexpr double kTolerance = 1e-6;

json load_goldens() {
  std::ifstream file(std::getenv("DIFFUSERS_GOLDENS_PATH"));
  return json::parse(file);
}

std::vector<float> to_floats(const json& values) {
  const auto doubles = values.get<std::vector<double>>();
  return std::vector<float>(doubles.begin(), doubles.end());
}

void expect_close(const std::vector<float>& actual, const json& expected) {
  const auto values = expected.get<std::vector<double>>();
  ASSERT_EQ(actual.size(), values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(actual[i], values[i], kTolerance * (1 + std::abs(values[i])))
        << "at " << i;
  }
}

// The golden model output of a step: the initial sample rolled by the step
// index and halved, as generate_goldens.py builds it.
std::vector<float> model_output(const std::vector<float>& sample, size_t step) {
  std::vector<float> output(sample.size());
  for (size_t i = 0; i < sample.size(); ++i) {
    output[i] =
        sample[(i + sample.size() - step % sample.size()) % sample.size()] *
        0.5f;
  }
  return output;
}

std::unique_ptr<Scheduler> scheduler_for(const json& scheduler_case) {
  auto result = make_scheduler(
      scheduler_case.at("class_name").get<std::string>(),
      scheduler_case.at("config").dump());
  EXPECT_TRUE(result.ok());
  return result.ok() ? std::move(result.get()) : nullptr;
}

Error set_timesteps(Scheduler& scheduler, const json& run) {
  std::optional<double> mu;
  if (!run.at("mu").is_null()) {
    mu = run.at("mu").get<double>();
  }
  return scheduler.set_timesteps(
      run.at("steps").get<int>(),
      run.at("sigmas").get<std::vector<double>>(),
      mu);
}

} // namespace

TEST(SchedulerTest, MatchesDiffusers) {
  const auto goldens = load_goldens();
  const auto initial_sample = to_floats(goldens.at("sample"));
  const auto noise = to_floats(goldens.at("noise"));

  for (const auto& scheduler_case : goldens.at("schedulers")) {
    for (const auto& run : scheduler_case.at("runs")) {
      SCOPED_TRACE(
          scheduler_case.at("name").get<std::string>() + " with " +
          std::to_string(run.at("steps").get<int>()) + " steps");
      auto scheduler = scheduler_for(scheduler_case);
      ASSERT_NE(scheduler, nullptr);
      ASSERT_EQ(set_timesteps(*scheduler, run), Error::Ok);
      expect_close(scheduler->timesteps(), run.at("timesteps"));
      EXPECT_NEAR(
          scheduler->init_noise_sigma(),
          run.at("init_noise_sigma").get<double>(),
          kTolerance * run.at("init_noise_sigma").get<double>());

      auto sample = initial_sample;
      const auto& scaled_inputs = run.at("scaled_inputs");
      const auto& samples = run.at("samples");
      for (size_t index = 0; index < scheduler->timesteps().size(); ++index) {
        auto scaled = sample;
        scheduler->scale_model_input(scaled, index);
        scheduler->step(model_output(initial_sample, index), sample, index);
        if (index < samples.size()) {
          expect_close(scaled, scaled_inputs[index]);
          expect_close(sample, samples[index]);
        }
      }
      expect_close(sample, run.at("final_sample"));

      if (run.contains("add_noise")) {
        auto noisy_scheduler = scheduler_for(scheduler_case);
        ASSERT_EQ(set_timesteps(*noisy_scheduler, run), Error::Ok);
        auto noisy = initial_sample;
        ASSERT_EQ(
            noisy_scheduler->add_noise(
                noisy, noise, run.at("add_noise_index").get<size_t>()),
            Error::Ok);
        expect_close(noisy, run.at("add_noise"));
      }
    }
  }
}

TEST(SchedulerTest, RejectsUnsupportedSettings) {
  EXPECT_EQ(
      make_scheduler("DPMSolverMultistepScheduler", "{}").error(),
      Error::NotSupported);
  EXPECT_EQ(
      make_scheduler("EulerDiscreteScheduler", R"({"use_karras_sigmas": true})")
          .error(),
      Error::InvalidArgument);
  EXPECT_EQ(
      make_scheduler("PNDMScheduler", R"({"skip_prk_steps": false})").error(),
      Error::InvalidArgument);
}

TEST(SchedulerTest, IgnoresKeysOfOtherSchedulers) {
  EXPECT_TRUE(make_scheduler(
                  "EulerDiscreteScheduler",
                  R"({"skip_prk_steps": true, "clip_sample": false})")
                  .ok());
}

TEST(SchedulerTest, DynamicShiftingRequiresMu) {
  auto scheduler = make_scheduler(
      "FlowMatchEulerDiscreteScheduler", R"({"use_dynamic_shifting": true})");
  ASSERT_TRUE(scheduler.ok());
  EXPECT_EQ(scheduler.get()->set_timesteps(4), Error::InvalidArgument);
}

TEST(SchedulerTest, FlowMatchingHasNoAddNoise) {
  auto scheduler = make_scheduler("FlowMatchEulerDiscreteScheduler", "{}");
  ASSERT_TRUE(scheduler.ok());
  ASSERT_EQ(scheduler.get()->set_timesteps(4), Error::Ok);
  std::vector<float> sample(4), noise(4);
  EXPECT_EQ(scheduler.get()->add_noise(sample, noise, 0), Error::NotSupported);
}

TEST(SchedulerTest, OnlyPlmsHasAWarmupCall) {
  auto plms = make_scheduler("PNDMScheduler", R"({"skip_prk_steps": true})");
  ASSERT_TRUE(plms.ok());
  ASSERT_EQ(plms.get()->set_timesteps(20), Error::Ok);
  EXPECT_EQ(plms.get()->timesteps().size(), 21);
  EXPECT_EQ(plms.get()->warmup_calls(), 1);
  for (const char* name :
       {"EulerDiscreteScheduler", "FlowMatchEulerDiscreteScheduler"}) {
    auto scheduler = make_scheduler(name, "{}");
    ASSERT_TRUE(scheduler.ok());
    EXPECT_EQ(scheduler.get()->warmup_calls(), 0) << name;
  }
}
