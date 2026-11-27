/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Ports of the Diffusers schedulers in
// https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers
// (scheduling_euler_discrete.py, scheduling_pndm.py,
// scheduling_flow_match_euler_discrete.py).

namespace executorch::extension::diffusers {

/**
 * Host-side port of a Diffusers scheduler. Samples are fp32 buffers, and every
 * call takes the index of the current entry in timesteps().
 */
class Scheduler {
 public:
  virtual ~Scheduler() = default;

  /**
   * Builds the inference schedule, as Diffusers' set_timesteps. sigmas and mu
   * are only accepted by flow matching schedulers.
   */
  virtual runtime::Error set_timesteps(
      int steps,
      const std::vector<double>& sigmas = {},
      std::optional<double> mu = std::nullopt) = 0;

  /** The denoiser timestep of every scheduler step. */
  const std::vector<float>& timesteps() const {
    return timesteps_;
  }

  /** Scale applied to the initial noise. */
  virtual float init_noise_sigma() const {
    return 1.0f;
  }

  /**
   * Calls at the start of a run that do not complete a step on their own,
   * such as the trial evaluation of a multistep method's first step.
   */
  virtual size_t warmup_calls() const {
    return 0;
  }

  /** Scales the denoiser input in place. */
  virtual void scale_model_input(std::vector<float>& sample, size_t index)
      const {}

  /** Advances sample in place by one step. */
  virtual void step(
      const std::vector<float>& model_output,
      std::vector<float>& sample,
      size_t index) = 0;

  /** Noises sample in place to the noise level of step index. */
  virtual runtime::Error add_noise(
      std::vector<float>& sample,
      const std::vector<float>& noise,
      size_t index) const = 0;

 protected:
  std::vector<float> timesteps_;
};

/**
 * Creates the scheduler named by a Diffusers scheduler class name from its
 * scheduler_config.json contents. Fails on settings the port does not
 * implement.
 */
runtime::Result<std::unique_ptr<Scheduler>> make_scheduler(
    const std::string& class_name,
    const std::string& config_json);

} // namespace executorch::extension::diffusers
