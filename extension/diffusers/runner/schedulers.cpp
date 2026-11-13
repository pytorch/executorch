/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/diffusers/runner/schedulers.h>

#include <executorch/extension/diffusers/runner/tensor_util.h>
#include <executorch/runtime/platform/log.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

namespace executorch::extension::diffusers {

using executorch::runtime::Error;
using executorch::runtime::Result;
using json = nlohmann::json;

namespace {

// Diffusers ignores config keys its scheduler does not take, so unknown keys
// are ignored here too. Keys a scheduler takes but this port does not
// implement must keep their Diffusers default.
template <typename T>
T get_or(const json& config, const char* key, T fallback) {
  const auto it = config.find(key);
  return it == config.end() || it->is_null() ? fallback : it->get<T>();
}

void require_default(const json& config, const char* key, const json& value) {
  const auto it = config.find(key);
  if (it != config.end() && !it->is_null() && *it != value) {
    throw std::invalid_argument(
        std::string("unsupported ") + key + "=" + it->dump());
  }
}

// np.linspace, computed in float64.
std::vector<double> numpy_linspace(double start, double end, int steps) {
  std::vector<double> values(steps, start);
  if (steps > 1) {
    const double step = (end - start) / (steps - 1);
    for (int i = 0; i < steps; ++i) {
      values[i] = static_cast<double>(i) * step + start;
    }
    values.back() = end;
  }
  return values;
}

// np.arange(start, stop, step) for float64, which fills from the difference
// of its first two values rather than from step.
std::vector<double> numpy_arange(double start, double stop, double step) {
  const auto count = static_cast<int64_t>(std::ceil((stop - start) / step));
  const double delta = (start + step) - start;
  std::vector<double> values(std::max<int64_t>(count, 0));
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = start + static_cast<double>(i) * delta;
  }
  return values;
}

// Inference timesteps for Diffusers' linspace, leading and trailing spacing,
// before the per-scheduler cast.
std::vector<double> spaced_timesteps(
    const std::string& spacing,
    int num_train_timesteps,
    int steps,
    int steps_offset) {
  std::vector<double> timesteps;
  if (spacing == "linspace") {
    timesteps = numpy_linspace(0, num_train_timesteps - 1, steps);
    std::reverse(timesteps.begin(), timesteps.end());
  } else if (spacing == "leading") {
    const int ratio = num_train_timesteps / steps;
    for (int i = steps - 1; i >= 0; --i) {
      timesteps.push_back(static_cast<double>(i * ratio + steps_offset));
    }
  } else if (spacing == "trailing") {
    const double ratio = static_cast<double>(num_train_timesteps) / steps;
    timesteps = numpy_arange(num_train_timesteps, 0, -ratio);
    std::transform(
        timesteps.begin(), timesteps.end(), timesteps.begin(), [](double t) {
          return std::nearbyint(t) - 1;
        });
  } else {
    throw std::invalid_argument("unsupported timestep_spacing=" + spacing);
  }
  return timesteps;
}

// alphas_cumprod of a linear or scaled_linear beta schedule, in float32 as
// Diffusers computes it.
std::vector<float> alphas_cumprod(const json& config) {
  require_default(config, "trained_betas", nullptr);
  const int num_train_timesteps =
      get_or<int>(config, "num_train_timesteps", 1000);
  const double beta_start = get_or<double>(config, "beta_start", 0.0001);
  const double beta_end = get_or<double>(config, "beta_end", 0.02);
  const auto schedule = get_or<std::string>(config, "beta_schedule", "linear");

  std::vector<float> betas;
  if (schedule == "linear") {
    betas = torch_linspace<float>(beta_start, beta_end, num_train_timesteps);
  } else if (schedule == "scaled_linear") {
    betas = torch_linspace<float>(
        std::sqrt(beta_start), std::sqrt(beta_end), num_train_timesteps);
    std::transform(betas.begin(), betas.end(), betas.begin(), [](float beta) {
      return beta * beta;
    });
  } else {
    throw std::invalid_argument("unsupported beta_schedule=" + schedule);
  }

  // torch.cumprod accumulates float32 in float64.
  std::vector<float> cumprod(betas.size());
  double product = 1.0;
  for (size_t i = 0; i < betas.size(); ++i) {
    product *= 1.0f - betas[i];
    cumprod[i] = static_cast<float>(product);
  }
  return cumprod;
}

std::string prediction_type(const json& config) {
  const auto type = get_or<std::string>(config, "prediction_type", "epsilon");
  if (type != "epsilon" && type != "v_prediction") {
    throw std::invalid_argument("unsupported prediction_type=" + type);
  }
  return type;
}

runtime::Error reject_flow_arguments(
    const std::vector<double>& sigmas,
    std::optional<double> mu) {
  if (!sigmas.empty() || mu) {
    ET_LOG(Error, "sigmas and mu are only accepted by flow matching");
    return Error::InvalidArgument;
  }
  return Error::Ok;
}

class EulerDiscreteScheduler : public Scheduler {
 public:
  explicit EulerDiscreteScheduler(const json& config)
      : num_train_timesteps_(get_or<int>(config, "num_train_timesteps", 1000)),
        steps_offset_(get_or<int>(config, "steps_offset", 0)),
        spacing_(get_or<std::string>(config, "timestep_spacing", "linspace")),
        prediction_type_(prediction_type(config)) {
    for (const char* key :
         {"use_karras_sigmas",
          "use_exponential_sigmas",
          "use_beta_sigmas",
          "rescale_betas_zero_snr"}) {
      require_default(config, key, false);
    }
    require_default(config, "interpolation_type", "linear");
    require_default(config, "timestep_type", "discrete");
    require_default(config, "final_sigmas_type", "zero");
    const auto alphas = alphas_cumprod(config);
    std::transform(
        alphas.begin(),
        alphas.end(),
        std::back_inserter(train_sigmas_),
        [](float alpha) { return std::sqrt((1 - alpha) / alpha); });
  }

  Error set_timesteps(
      int steps,
      const std::vector<double>& sigmas,
      std::optional<double> mu) override {
    ET_CHECK_OK_OR_RETURN_ERROR(reject_flow_arguments(sigmas, mu));
    if (steps <= 0) {
      return Error::InvalidArgument;
    }
    const auto spaced =
        spaced_timesteps(spacing_, num_train_timesteps_, steps, steps_offset_);
    timesteps_.assign(spaced.begin(), spaced.end());
    sigmas_.clear();
    std::transform(
        timesteps_.begin(),
        timesteps_.end(),
        std::back_inserter(sigmas_),
        [this](float t) { return interpolate_sigma(t); });
    sigmas_.push_back(0.0f);
    return Error::Ok;
  }

  float init_noise_sigma() const override {
    const float max_sigma = *std::max_element(sigmas_.begin(), sigmas_.end());
    if (spacing_ == "linspace" || spacing_ == "trailing") {
      return max_sigma;
    }
    return std::sqrt(max_sigma * max_sigma + 1);
  }

  void scale_model_input(std::vector<float>& sample, size_t index)
      const override {
    const float sigma = sigmas_[index];
    const float scale = std::sqrt(sigma * sigma + 1);
    std::transform(
        sample.begin(), sample.end(), sample.begin(), [scale](float value) {
          return value / scale;
        });
  }

  void step(
      const std::vector<float>& model_output,
      std::vector<float>& sample,
      size_t index) override {
    const float sigma = sigmas_[index];
    const float dt = sigmas_[index + 1] - sigma;
    const float v_output_scale = -sigma / std::sqrt(sigma * sigma + 1);
    const float v_sample_scale = sigma * sigma + 1;
    for (size_t i = 0; i < sample.size(); ++i) {
      const float original = prediction_type_ == "epsilon"
          ? sample[i] - sigma * model_output[i]
          : model_output[i] * v_output_scale + sample[i] / v_sample_scale;
      const float derivative = (sample[i] - original) / sigma;
      sample[i] = sample[i] + derivative * dt;
    }
  }

  Error add_noise(
      std::vector<float>& sample,
      const std::vector<float>& noise,
      size_t index) const override {
    for (size_t i = 0; i < sample.size(); ++i) {
      sample[i] = sample[i] + noise[i] * sigmas_[index];
    }
    return Error::Ok;
  }

 private:
  // np.interp of the training sigmas at timestep t.
  float interpolate_sigma(double t) const {
    if (t <= 0) {
      return train_sigmas_.front();
    }
    const size_t last = train_sigmas_.size() - 1;
    if (t >= static_cast<double>(last)) {
      return train_sigmas_.back();
    }
    const auto low = static_cast<size_t>(t);
    if (static_cast<double>(low) == t) {
      return train_sigmas_[low];
    }
    const double slope =
        static_cast<double>(train_sigmas_[low + 1]) - train_sigmas_[low];
    return static_cast<float>(
        slope * (t - static_cast<double>(low)) + train_sigmas_[low]);
  }

  const int num_train_timesteps_;
  const int steps_offset_;
  const std::string spacing_;
  const std::string prediction_type_;
  std::vector<float> train_sigmas_;
  std::vector<float> sigmas_;
};

// PNDM in its PLMS form (skip_prk_steps=True), as Stable Diffusion uses it.
class PNDMScheduler : public Scheduler {
 public:
  explicit PNDMScheduler(const json& config)
      : num_train_timesteps_(get_or<int>(config, "num_train_timesteps", 1000)),
        steps_offset_(get_or<int>(config, "steps_offset", 0)),
        spacing_(get_or<std::string>(config, "timestep_spacing", "leading")),
        prediction_type_(prediction_type(config)),
        alphas_cumprod_(alphas_cumprod(config)),
        final_alpha_cumprod_(
            get_or<bool>(config, "set_alpha_to_one", false)
                ? 1.0f
                : alphas_cumprod_.front()) {
    if (!get_or<bool>(config, "skip_prk_steps", false)) {
      throw std::invalid_argument("unsupported skip_prk_steps=false");
    }
  }

  // The first call takes a trial step that the second call redoes with the
  // averaged model output.
  size_t warmup_calls() const override {
    return 1;
  }

  Error set_timesteps(
      int steps,
      const std::vector<double>& sigmas,
      std::optional<double> mu) override {
    ET_CHECK_OK_OR_RETURN_ERROR(reject_flow_arguments(sigmas, mu));
    if (steps <= 0) {
      return Error::InvalidArgument;
    }
    auto spaced =
        spaced_timesteps(spacing_, num_train_timesteps_, steps, steps_offset_);
    // spaced_timesteps returns descending values; PNDM's own spacing is
    // ascending and rounded to integers.
    std::vector<int64_t> ascending;
    for (auto it = spaced.rbegin(); it != spaced.rend(); ++it) {
      ascending.push_back(static_cast<int64_t>(std::nearbyint(*it)));
    }
    // PLMS repeats the second to last timestep.
    std::vector<int64_t> plms(ascending.begin(), ascending.end() - 1);
    if (ascending.size() >= 2) {
      plms.push_back(ascending[ascending.size() - 2]);
    }
    plms.push_back(ascending.back());
    std::reverse(plms.begin(), plms.end());

    step_ratio_ = num_train_timesteps_ / steps;
    integer_timesteps_ = plms;
    timesteps_.assign(plms.begin(), plms.end());
    ets_.clear();
    cur_sample_.clear();
    counter_ = 0;
    return Error::Ok;
  }

  void step(
      const std::vector<float>& model_output,
      std::vector<float>& sample,
      size_t index) override {
    int64_t timestep = integer_timesteps_[index];
    int64_t prev_timestep = timestep - step_ratio_;
    if (counter_ != 1) {
      if (ets_.size() == 4) {
        ets_.erase(ets_.begin());
      }
      ets_.push_back(model_output);
    } else {
      prev_timestep = timestep;
      timestep = timestep + step_ratio_;
    }

    std::vector<float> combined(sample.size());
    const auto& e1 = ets_.back();
    if (ets_.size() == 1 && counter_ == 0) {
      combined = model_output;
      cur_sample_ = sample;
    } else if (ets_.size() == 1 && counter_ == 1) {
      for (size_t i = 0; i < combined.size(); ++i) {
        combined[i] = (model_output[i] + e1[i]) / 2;
      }
      sample = cur_sample_;
      cur_sample_.clear();
    } else if (ets_.size() == 2) {
      const auto& e2 = ets_[0];
      for (size_t i = 0; i < combined.size(); ++i) {
        combined[i] = (3 * e1[i] - e2[i]) / 2;
      }
    } else if (ets_.size() == 3) {
      const auto& e2 = ets_[1];
      const auto& e3 = ets_[0];
      for (size_t i = 0; i < combined.size(); ++i) {
        combined[i] = (23 * e1[i] - 16 * e2[i] + 5 * e3[i]) / 12;
      }
    } else {
      const auto& e2 = ets_[2];
      const auto& e3 = ets_[1];
      const auto& e4 = ets_[0];
      const auto scale = static_cast<float>(1.0 / 24);
      for (size_t i = 0; i < combined.size(); ++i) {
        combined[i] =
            scale * (55 * e1[i] - 59 * e2[i] + 37 * e3[i] - 9 * e4[i]);
      }
    }
    previous_sample(sample, timestep, prev_timestep, combined);
    ++counter_;
  }

  Error add_noise(
      std::vector<float>& sample,
      const std::vector<float>& noise,
      size_t index) const override {
    const float alpha = alphas_cumprod_[integer_timesteps_[index]];
    const float sample_scale = std::sqrt(alpha);
    const float noise_scale = std::sqrt(1 - alpha);
    for (size_t i = 0; i < sample.size(); ++i) {
      sample[i] = sample_scale * sample[i] + noise_scale * noise[i];
    }
    return Error::Ok;
  }

 private:
  void previous_sample(
      std::vector<float>& sample,
      int64_t timestep,
      int64_t prev_timestep,
      std::vector<float>& model_output) const {
    const float alpha = alphas_cumprod_[timestep];
    const float alpha_prev = prev_timestep >= 0 ? alphas_cumprod_[prev_timestep]
                                                : final_alpha_cumprod_;
    const float beta = 1 - alpha;
    const float beta_prev = 1 - alpha_prev;
    if (prediction_type_ == "v_prediction") {
      for (size_t i = 0; i < sample.size(); ++i) {
        model_output[i] =
            std::sqrt(alpha) * model_output[i] + std::sqrt(beta) * sample[i];
      }
    }
    const float sample_coeff = std::sqrt(alpha_prev / alpha);
    const float denominator =
        alpha * std::sqrt(beta_prev) + std::sqrt(alpha * beta * alpha_prev);
    for (size_t i = 0; i < sample.size(); ++i) {
      sample[i] = sample_coeff * sample[i] -
          (alpha_prev - alpha) * model_output[i] / denominator;
    }
  }

  const int num_train_timesteps_;
  const int steps_offset_;
  const std::string spacing_;
  const std::string prediction_type_;
  const std::vector<float> alphas_cumprod_;
  const float final_alpha_cumprod_;
  int64_t step_ratio_ = 0;
  std::vector<int64_t> integer_timesteps_;
  std::vector<std::vector<float>> ets_;
  std::vector<float> cur_sample_;
  int counter_ = 0;
};

class FlowMatchEulerDiscreteScheduler : public Scheduler {
 public:
  explicit FlowMatchEulerDiscreteScheduler(const json& config)
      : num_train_timesteps_(get_or<int>(config, "num_train_timesteps", 1000)),
        shift_(get_or<double>(config, "shift", 1.0)),
        use_dynamic_shifting_(
            get_or<bool>(config, "use_dynamic_shifting", false)),
        shift_terminal_(get_or<double>(config, "shift_terminal", 0.0)) {
    for (const char* key :
         {"invert_sigmas",
          "use_karras_sigmas",
          "use_exponential_sigmas",
          "use_beta_sigmas",
          "stochastic_sampling"}) {
      require_default(config, key, false);
    }
    const auto time_shift =
        get_or<std::string>(config, "time_shift_type", "exponential");
    if (time_shift != "exponential" && time_shift != "linear") {
      throw std::invalid_argument("unsupported time_shift_type=" + time_shift);
    }
    exponential_time_shift_ = time_shift == "exponential";
  }

  Error set_timesteps(
      int steps,
      const std::vector<double>& sigmas,
      std::optional<double> mu) override {
    if (steps <= 0 || (!sigmas.empty() && sigmas.size() != size_t(steps))) {
      ET_LOG(Error, "sigmas must hold one value per step");
      return Error::InvalidArgument;
    }
    if (use_dynamic_shifting_ && !mu) {
      ET_LOG(Error, "mu is required when use_dynamic_shifting is set");
      return Error::InvalidArgument;
    }
    if (sigmas.empty()) {
      // Diffusers spans the shifted training range in float64.
      const float sigma_max = shift(1.0f, mu);
      const float sigma_min =
          shift(1.0f / static_cast<float>(num_train_timesteps_), mu);
      auto values = numpy_linspace(
          static_cast<double>(sigma_max) * num_train_timesteps_,
          static_cast<double>(sigma_min) * num_train_timesteps_,
          steps);
      std::transform(
          values.begin(), values.end(), values.begin(), [this](double value) {
            return value / num_train_timesteps_;
          });
      sigmas_ = shifted(values, mu);
    } else {
      sigmas_ = shifted(std::vector<float>(sigmas.begin(), sigmas.end()), mu);
    }
    timesteps_.clear();
    for (float sigma : sigmas_) {
      timesteps_.push_back(sigma * static_cast<float>(num_train_timesteps_));
    }
    sigmas_.push_back(0.0f);
    return Error::Ok;
  }

  void step(
      const std::vector<float>& model_output,
      std::vector<float>& sample,
      size_t index) override {
    const float dt = sigmas_[index + 1] - sigmas_[index];
    for (size_t i = 0; i < sample.size(); ++i) {
      sample[i] = sample[i] + dt * model_output[i];
    }
  }

  Error add_noise(std::vector<float>&, const std::vector<float>&, size_t)
      const override {
    ET_LOG(Error, "FlowMatchEulerDiscreteScheduler does not support add_noise");
    return Error::NotSupported;
  }

 private:
  // Applies the configured shift to one training sigma, in float32 as the
  // constructor of the Diffusers scheduler does.
  float shift(float sigma, std::optional<double> mu) const {
    if (use_dynamic_shifting_) {
      return sigma;
    }
    return static_cast<float>(shift_) * sigma /
        (1 + static_cast<float>(shift_ - 1) * sigma);
  }

  // Shifts and stretches sigmas in their own precision, as NumPy does for
  // float32 and float64 arrays with Python scalars.
  template <typename T>
  std::vector<float> shifted(std::vector<T> values, std::optional<double> mu)
      const {
    for (auto& t : values) {
      if (use_dynamic_shifting_) {
        const T m = exponential_time_shift_ ? static_cast<T>(std::exp(*mu))
                                            : static_cast<T>(*mu);
        t = m / (m + (1 / t - 1));
      } else {
        t = static_cast<T>(shift_) * t / (1 + static_cast<T>(shift_ - 1) * t);
      }
    }
    if (shift_terminal_ != 0.0) {
      const T scale = (1 - values.back()) / static_cast<T>(1 - shift_terminal_);
      std::transform(
          values.begin(), values.end(), values.begin(), [scale](T t) {
            return 1 - ((1 - t) / scale);
          });
    }
    return std::vector<float>(values.begin(), values.end());
  }

  const int num_train_timesteps_;
  const double shift_;
  const bool use_dynamic_shifting_;
  const double shift_terminal_;
  bool exponential_time_shift_;
  std::vector<float> sigmas_;
};

} // namespace

Result<std::unique_ptr<Scheduler>> make_scheduler(
    const std::string& class_name,
    const std::string& config_json) {
  try {
    const auto config = json::parse(config_json);
    if (class_name == "FlowMatchEulerDiscreteScheduler") {
      return std::unique_ptr<Scheduler>(
          new FlowMatchEulerDiscreteScheduler(config));
    }
    if (class_name == "EulerDiscreteScheduler") {
      return std::unique_ptr<Scheduler>(new EulerDiscreteScheduler(config));
    }
    if (class_name == "PNDMScheduler") {
      return std::unique_ptr<Scheduler>(new PNDMScheduler(config));
    }
  } catch (const std::exception& e) {
    ET_LOG(Error, "Invalid %s config: %s", class_name.c_str(), e.what());
    return Error::InvalidArgument;
  }
  ET_LOG(Error, "Unsupported scheduler %s", class_name.c_str());
  return Error::NotSupported;
}

} // namespace executorch::extension::diffusers
